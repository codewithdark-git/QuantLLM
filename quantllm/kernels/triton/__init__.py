"""
Triton Kernels for QuantLLM

High-performance fused operations for quantized inference.
"""

from .quantized_linear import (
    TritonQuantizedLinear,
    fused_dequant_matmul,
    int4_matmul,
    is_triton_available,
    triton_q4_0_quantize,
    triton_q8_0_quantize,
)

__all__ = [
    "TritonQuantizedLinear",
    "fused_dequant_matmul",
    "int4_matmul",
    "is_triton_available",
    "triton_q4_0_quantize",
    "triton_q8_0_quantize",
]
