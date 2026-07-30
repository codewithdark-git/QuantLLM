"""
QuantLLM CPU kernels -- Python entry point for C++ pybind11 modules.

Provides INT4/INT8 weight-only matmul for CPU with AVX2/AVX-512/NEON
acceleration. Falls back to PyTorch's native matmul when C++ kernels
are not compiled.
"""

from __future__ import annotations

import importlib
import os
import platform
import warnings
from typing import Optional

import torch
import torch.nn as nn


def _is_avx2_supported() -> bool:
    """Check if CPU supports AVX2 instructions."""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                flags = f.read()
                return "avx2" in flags.lower()
        except Exception:
            pass
    elif platform.system() == "Darwin":
        return platform.machine() in ("x86_64", "x86_64h")
    return False


def _is_arm_neon_supported() -> bool:
    """Check if CPU supports NEON (Apple Silicon / ARM)."""
    return platform.machine() in ("arm64", "aarch64")


HAS_AVX2 = _is_avx2_supported()
HAS_NEON = _is_arm_neon_supported()


def is_cpu_kernel_available() -> bool:
    return HAS_AVX2 or HAS_NEON


class CPUInt4Linear(nn.Module):
    """INT4 quantized linear layer with CPU-optimized matmul.

    Uses the Python fallback (dequantize + torch.mm) when C++ kernels
    are not compiled.
    """

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

        self.register_buffer(
            "qweight",
            torch.empty(out_features, self.num_groups, in_features // (8 // bits), dtype=torch.uint8),
        )
        self.register_buffer("scales", torch.empty(out_features, self.num_groups, dtype=torch.float32))
        self.register_buffer("zeros", torch.empty(out_features, self.num_groups, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def quantize(self, weight: torch.Tensor) -> None:
        w = weight.float().reshape(self.out_features, self.in_features)
        if self.group_size < self.in_features:
            w = w.reshape(self.out_features, self.num_groups, self.group_size)

        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)
        scale = torch.clamp((w_max - w_min) / (2**self.bits - 1), min=1e-10)
        zero = w_min

        wq = torch.round((w - zero) / scale).clamp(0, 2**self.bits - 1).to(torch.uint8)

        if self.bits == 4:
            packed = (wq[:, :, ::2] << 4) | wq[:, :, 1::2]
            self.qweight = packed.reshape(self.out_features, self.num_groups, -1)
        else:
            self.qweight = wq.reshape(self.out_features, self.num_groups, -1)

        self.scales = scale.squeeze(-1).to(torch.float32)
        self.zeros = zero.squeeze(-1).to(torch.float32)

    def _dequantize_weights(self) -> torch.Tensor:
        if self.bits == 4:
            high = (self.qweight >> 4) & 0xF
            low = self.qweight & 0xF
            wq = torch.stack([high, low], dim=-1).reshape(self.out_features, self.num_groups, self.group_size)
        else:
            wq = self.qweight.float().reshape(self.out_features, self.num_groups, self.group_size)

        w = wq.float() * self.scales.unsqueeze(-1) + self.zeros.unsqueeze(-1)
        return w.reshape(self.out_features, self.in_features).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weights()
        return torch.nn.functional.linear(x, w.to(x.device), self.bias)


class CPUInt8Linear(nn.Module):
    """INT8 quantized linear layer with CPU-optimized matmul."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def quantize(self, weight: torch.Tensor) -> None:
        w = weight.float()
        w_max = w.abs().amax(dim=1, keepdim=True)
        self.scale = w_max / 127.0
        self.scale = torch.clamp(self.scale, min=1e-10)
        self.qweight = torch.round(w / self.scale).clamp(-128, 127).to(torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.float() * self.scale.to(x.device)
        return torch.nn.functional.linear(x, w, self.bias)


def optimize_for_cpu(model: nn.Module, bits: int = 4, group_size: int = 128) -> nn.Module:
    """Replace linear layers with CPU-optimized quantized versions.

    Args:
        model: Full-precision PyTorch model.
        bits: Quantization bits (4 or 8).
        group_size: Group size for INT4 quantization.

    Returns:
        Model with CPU-optimized quantized linear layers.
    """
    if bits == 8:
        linear_cls = CPUInt8Linear
    else:
        linear_cls = CPUInt4Linear

    replacements = []
    skip_names = {"lm_head", "embed_tokens", "embed"}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(skip in name for skip in skip_names):
                continue
            replacements.append((name, module))

    for name, module in replacements:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)

        if bits == 8:
            qlayer = CPUInt8Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
        else:
            qlayer = CPUInt4Linear(
                module.in_features,
                module.out_features,
                bits=bits,
                group_size=group_size,
                bias=module.bias is not None,
            )

        qlayer.to(module.weight.device)
        qlayer.quantize(module.weight.data)
        if module.bias is not None:
            if bits == 8:
                qlayer.bias = nn.Parameter(module.bias.data.float())
            else:
                qlayer.bias = nn.Parameter(module.bias.data.float())

        setattr(parent, parts[-1], qlayer)

    return model
