"""
QuantLLM Kernels Module

High-performance kernels for quantized operations:
    - CUDA kernels via Triton
    - CPU kernels via pybind11 (with Python fallback)
"""

from .triton import (
    TritonQuantizedLinear,
    fused_dequant_matmul,
    int4_matmul,
    is_triton_available,
    triton_q4_0_quantize,
    triton_q8_0_quantize,
)

from .cpu import (
    CPUInt4Linear,
    CPUInt8Linear,
    optimize_for_cpu,
    is_cpu_kernel_available,
)

__all__ = [
    "TritonQuantizedLinear",
    "fused_dequant_matmul",
    "int4_matmul",
    "is_triton_available",
    "triton_q4_0_quantize",
    "triton_q8_0_quantize",
    "CPUInt4Linear",
    "CPUInt8Linear",
    "optimize_for_cpu",
    "is_cpu_kernel_available",
]
