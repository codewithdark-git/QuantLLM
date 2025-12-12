"""
QuantLLM Quantization Module v2.0

Pure Python GGUF export - No llama.cpp dependency!
"""

# Pure Python GGUF converter
from .gguf_writer import (
    GGUFWriter,
    GGUFConverter,
    convert_to_gguf,
    list_quant_types,
    QUANT_TYPES,
    QUANT_ALIASES,
    FastQuantizer,
)

# Core quantization engine
from .quantization_engine import (
    QuantizationConfig,
    QuantizedLinear,
    QuantizationEngine
)

# Convenience exports
export_to_gguf = convert_to_gguf
GGUF_QUANT_TYPES = QUANT_TYPES
ALLOWED_QUANTS = {name: info.description for name, info in QUANT_TYPES.items()}

__all__ = [
    # GGUF Export (Pure Python - No Dependencies!)
    "GGUFWriter",
    "GGUFConverter",
    "convert_to_gguf",
    "export_to_gguf",
    "list_quant_types",
    "QUANT_TYPES",
    "GGUF_QUANT_TYPES",
    "ALLOWED_QUANTS",
    "QUANT_ALIASES",
    "FastQuantizer",
    
    # Core quantization
    "QuantizationConfig",
    "QuantizedLinear", 
    "QuantizationEngine",
]


def print_quantization_info():
    """Print available quantization methods."""
    print("\n" + "="*60)
    print(" QuantLLM GGUF Quantization ".center(60, "="))
    print("="*60 + "\n")
    
    print("📦 Available Quantization Types:\n")
    for name, info in QUANT_TYPES.items():
        bpw = info.bytes_per_weight()
        print(f"  {name:12} - {info.description:35} ({bpw:.2f} B/W)")
    
    print("\n" + "="*60 + "\n")
