"""
QuantLLM Quantization Module v2.0

Provides GGUF conversion and quantization capabilities using llama.cpp.

Quantization Flow:
    1. convert_to_gguf: Convert HuggingFace model → F16/BF16 GGUF
    2. quantize_gguf: Quantize GGUF file → Q4_K_M, Q5_K_M, Q8_0, etc.
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

# Export alias
export_to_gguf = convert_to_gguf

# Common GGUF Quantization Types (for reference/validation)
# Ordered by quality (higher = better quality, larger size)
GGUF_QUANT_TYPES = [
    # Full precision
    "f32",      # 32-bit float (largest)
    "f16",      # 16-bit float
    "bf16",     # Brain float 16
    
    # High quality quantization
    "q8_0",     # 8-bit quantization
    
    # K-quants (recommended for most use cases)
    "q6_k",     # 6-bit K-quant
    "q5_k_m",   # 5-bit K-quant medium
    "q5_k_s",   # 5-bit K-quant small
    "q4_k_m",   # 4-bit K-quant medium (best balance)
    "q4_k_s",   # 4-bit K-quant small
    
    # Lower quality (for very constrained environments)
    "q3_k_l",   # 3-bit K-quant large
    "q3_k_m",   # 3-bit K-quant medium
    "q3_k_s",   # 3-bit K-quant small
    "q2_k",     # 2-bit K-quant (smallest)
    
    # I-quants (importance-based, newest)
    "iq4_nl",   # 4-bit importance quant
    "iq4_xs",   # 4-bit importance quant extra small
    "iq3_xxs",  # 3-bit importance quant
    "iq2_xxs",  # 2-bit importance quant
    "iq1_s",    # 1-bit importance quant
]

# Recommended quantization types for different use cases
QUANT_RECOMMENDATIONS = {
    "quality": "q5_k_m",      # Best quality-to-size ratio
    "balanced": "q4_k_m",     # Good balance (most popular)
    "speed": "q4_k_s",        # Faster inference
    "memory": "q3_k_m",       # Low memory usage
    "extreme": "q2_k",        # Minimum size
}

__all__ = [
    # Core functions
    "convert_to_gguf",
    "quantize_gguf",
    "export_to_gguf",
    
    # Installation
    "check_llama_cpp",
    "install_llama_cpp",
    "ensure_llama_cpp_installed",
    "get_llama_cpp_dir",
    
    # Utilities
    "detect_environment",
    
    # Constants
    "GGUF_QUANT_TYPES",
    "QUANT_RECOMMENDATIONS",
]
