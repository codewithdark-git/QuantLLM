"""
QuantLLM Quantization Module v2.2

Provides multiple quantization backends:
    1. llama.cpp GGUF conversion & quantization
    2. Native HQQ (Half-Quadratic Quantization) implementation
"""

from .llama_cpp import (
    convert_to_gguf,
    quantize_gguf,
    check_llama_cpp,
    install_llama_cpp,
    ensure_llama_cpp_installed,
    detect_environment,
    get_llama_cpp_dir,
)

from .hqq import (
    HQQLinear,
    HQQQuantizer,
    HQQConfig,
    HQQ_QUANT_TYPES,
)

export_to_gguf = convert_to_gguf

GGUF_QUANT_TYPES = [
    "f32", "f16", "bf16", "q8_0", "q6_k",
    "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s",
    "q3_k_l", "q3_k_m", "q3_k_s", "q2_k",
    "iq4_nl", "iq4_xs", "iq3_xxs", "iq2_xxs", "iq1_s",
]

QUANT_RECOMMENDATIONS = {
    "quality": "q5_k_m",
    "balanced": "q4_k_m",
    "speed": "q4_k_s",
    "memory": "q3_k_m",
    "extreme": "q2_k",
}

__all__ = [
    "convert_to_gguf", "quantize_gguf", "export_to_gguf",
    "check_llama_cpp", "install_llama_cpp", "ensure_llama_cpp_installed",
    "get_llama_cpp_dir", "detect_environment",
    "GGUF_QUANT_TYPES", "QUANT_RECOMMENDATIONS",
    "HQQLinear", "HQQQuantizer", "HQQConfig", "HQQ_QUANT_TYPES",
]
