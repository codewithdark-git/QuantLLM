"""
Triton Kernels for QuantLLM

High-performance fused operations for quantized inference.
"""

from .quantized_linear import (
    TritonQuantizedLinear,
    fused_dequant_matmul,
    is_triton_available,
)

__all__ = [
    "TritonQuantizedLinear",
    "fused_dequant_matmul",
    "is_triton_available",
]
