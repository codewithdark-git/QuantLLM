"""
QuantLLM Kernels Module

High-performance CUDA kernels using Triton for ultra-fast quantized operations.
"""

from .triton import (
    TritonQuantizedLinear,
    fused_dequant_matmul,
    is_triton_available,
)

__all__ = [
    "TritonQuantizedLinear",
    "fused_dequant_matmul",
    "is_triton_available",
]
