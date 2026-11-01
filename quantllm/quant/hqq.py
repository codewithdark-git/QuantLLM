"""
Half-Quadratic Quantization (HQQ) for QuantLLM.

HQQ produces state-of-the-art quality at 2-4 bits without requiring
calibration data (unlike GPTQ). Algorithm: Badri & Shaji, 2023.

This module implements weight-only HQQ for linear layers:
    - Per-channel weight normalization
    - Quantization scale/shift optimization via half-quadratic splitting
    - Support for 2/3/4/8-bit
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


HQQ_QUANT_TYPES = {
    2: "q2_k",
    3: "q3_k_m",
    4: "q4_k_m",
    8: "q8_0",
}


@dataclass
class HQQConfig:
    bits: int = 4
    group_size: int = 128
    optimize: bool = True


class HQQLinear(nn.Module):
    """HQQ-quantized linear layer with half-quadratic optimization."""

    bias: Optional[torch.Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = False,
        optimize: bool = True,
    ):
        super().__init__()
        self.optimize = optimize
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.num_groups = (in_features + group_size - 1) // group_size

        dtype = torch.uint8
        qweight_shape = (
            (out_features, self.num_groups, group_size // 2)
            if bits == 4
            else (out_features, self.num_groups, group_size)
        )
        self.register_buffer("qweight", torch.empty(qweight_shape, dtype=dtype))
        self.register_buffer("scales", torch.empty(out_features, self.num_groups, 1, dtype=torch.float16))
        self.register_buffer("zeros", torch.empty(out_features, self.num_groups, 1, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def _pack_4bit(self, wq: torch.Tensor) -> torch.Tensor:
        return (wq[:, :, ::2] << 4) | wq[:, :, 1::2]

    def _unpack_4bit(self, wq: torch.Tensor) -> torch.Tensor:
        high = (wq >> 4) & 0xF
        low = wq & 0xF
        out = torch.stack([high, low], dim=-1).reshape(wq.shape[0], wq.shape[1], -1)
        return out[:, :, : self.group_size]

    def _group(self, w: torch.Tensor) -> torch.Tensor:
        """Reshape ``(out, in)`` -> ``(out, num_groups, group_size)``.

        When ``group_size`` does not divide ``in_features`` (or exceeds it),
        the last group is padded by repeating the final column so the padding
        does not widen the group's min/max range.
        """
        padded = self.num_groups * self.group_size
        pad = padded - self.in_features
        if pad > 0:
            w = torch.cat([w, w[:, -1:].expand(-1, pad)], dim=1)
        return w.reshape(self.out_features, self.num_groups, self.group_size)

    @staticmethod
    def _shrink_lp(x: torch.Tensor, beta: float, lp_norm: float) -> torch.Tensor:
        """Generalized soft-thresholding (proximal operator of the l_p norm, p < 1)."""
        return torch.sign(x) * torch.relu(torch.abs(x) - (1.0 / beta) * torch.abs(x).pow(lp_norm - 1))

    def _optimize_zero(
        self,
        w: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
        iters: int = 20,
        beta: float = 10.0,
        kappa: float = 1.01,
        lp_norm: float = 0.7,
    ) -> torch.Tensor:
        """Half-quadratic proximal optimization of the zero-point (HQQ, Badri & Shaji 2023).

        Works in the HQQ parametrisation ``w ~= (w_q - z) * s`` where ``z = -zero / s``
        and returns the optimized additive offset (``zero``) used by this layer.
        """
        qmax = 2**self.bits - 1
        z = -zero / scale
        best_z = z
        best_err = torch.abs(w - (torch.round(w / scale + z).clamp(0, qmax) - z) * scale).mean()
        for _ in range(iters):
            w_q = torch.round(w / scale + z).clamp(0, qmax)
            w_r = (w_q - z) * scale
            w_e = self._shrink_lp(w - w_r, beta, lp_norm)
            z = torch.mean(w_q - (w - w_e) / scale, dim=-1, keepdim=True)
            beta *= kappa

            err = torch.abs(w - (torch.round(w / scale + z).clamp(0, qmax) - z) * scale).mean()
            if err < best_err:
                best_err = err
                best_z = z
            else:
                break
        return -best_z * scale

    def quantize(self, weight: torch.Tensor) -> None:
        w = self._group(weight.float().reshape(self.out_features, self.in_features))

        # Min-max initialization
        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)
        scale = torch.clamp((w_max - w_min) / (2**self.bits - 1), min=1e-10)
        zero = w_min

        if self.optimize:
            zero = self._optimize_zero(w, scale, zero)

        wq = torch.round((w - zero) / scale).clamp(0, 2**self.bits - 1)

        if self.bits == 4:
            self.qweight = self._pack_4bit(wq.to(torch.uint8))
        else:
            self.qweight = wq.to(torch.uint8)
        self.scales = scale.to(torch.float16)
        self.zeros = zero.to(torch.float16)

    def dequantize(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        wq = self._unpack_4bit(self.qweight) if self.bits == 4 else self.qweight.float()
        w = wq.float().to(self.scales.device) * self.scales.float() + self.zeros.float()
        w = w.reshape(self.out_features, -1)[:, : self.in_features]
        return w.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in the input dtype (e.g., bfloat16) so it composes with unquantized layers like lm_head
        w = self.dequantize(dtype=x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, w, bias)


class HQQQuantizer:
    """Applies HQQ quantization to a full model."""

    def __init__(self, config: Optional[HQQConfig] = None):
        self.config = config or HQQConfig()

    def quantize_model(
        self,
        model: nn.Module,
        bits: Optional[int] = None,
        group_size: Optional[int] = None,
        skip_names: Optional[set] = None,
    ) -> nn.Module:
        bits = bits or self.config.bits
        group_size = group_size or self.config.group_size
        skip_names = skip_names or {"lm_head", "embed_tokens", "embed"}

        replacements = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not any(skip in name for skip in skip_names):
                replacements.append((name, module))

        for name, module in replacements:
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)

            hqq = HQQLinear(
                module.in_features,
                module.out_features,
                bits=bits,
                group_size=group_size,
                bias=module.bias is not None,
                optimize=self.config.optimize,
            )
            hqq.to(module.weight.device)
            hqq.quantize(module.weight.data)
            if module.bias is not None:
                hqq.bias = nn.Parameter(module.bias.data.half())
            setattr(parent, parts[-1], hqq)

        return model

    @staticmethod
    def estimate_size(original_params: int, bits: int) -> float:
        return original_params * bits / 8 * 1.05 / (1024**3)
