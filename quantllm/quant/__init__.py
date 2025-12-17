"""
QuantLLM Quantization Module v2.0
"""

from .llama_cpp import (
    convert_to_gguf,
    check_llama_cpp,
    install_llama_cpp,
    ensure_llama_cpp_installed,
    detect_environment,
)

# Export alias
export_to_gguf = convert_to_gguf

# Common GGUF Quantization Types (for reference/validation)
GGUF_QUANT_TYPES = [
    "f16", "q8_0", 
    "q4_k_m", "q4_k_s", 
    "q5_k_m", "q5_k_s", 
    "q6_k", 
    "q2_k", "q3_k_m", "q3_k_s", "q3_k_l"
]

__all__ = [
    "convert_to_gguf",
    "export_to_gguf",
    "check_llama_cpp",
    "install_llama_cpp",
    "ensure_llama_cpp_installed",
    "detect_environment",
    "GGUF_QUANT_TYPES",
]
