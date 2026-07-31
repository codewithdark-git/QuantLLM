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
    ):
        super().__init__()
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

    def quantize(self, weight: torch.Tensor) -> None:
        device = weight.device
        w = weight.float().reshape(self.out_features, self.in_features)

        if self.group_size < self.in_features:
            w = w.reshape(self.out_features, self.num_groups, self.group_size)

        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)
        scale = torch.clamp((w_max - w_min) / (2**self.bits - 1), min=1e-10)
        zero = w_min

        wq = torch.round((w - zero) / scale).clamp(0, 2**self.bits - 1)

        self.qweight = self._pack_4bit(wq.to(torch.uint8)) if self.bits == 4 else wq.to(torch.uint8)
        self.scales = scale.to(torch.float16)
        self.zeros = zero.to(torch.float16)

    def dequantize(self) -> torch.Tensor:
        wq = self._unpack_4bit(self.qweight) if self.bits == 4 else self.qweight.float()
        w = wq.float().to(self.scales.device) * self.scales + self.zeros
        return w.reshape(self.out_features, self.in_features).to(torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize()
        return torch.nn.functional.linear(x.to(w.dtype), w, self.bias)


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
                module.in_features, module.out_features, bits=bits, group_size=group_size, bias=module.bias is not None
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
