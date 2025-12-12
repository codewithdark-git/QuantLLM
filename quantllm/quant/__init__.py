"""GGUF Quantization for Large Language Models."""

from .quantization_engine import (
    QuantizationConfig,
    QuantizedLinear,
    QuantizationEngine
)

# New modern GGUF converter (recommended)
from .gguf_converter import (
    GGUFExporter,
    export_to_gguf,
    print_quantization_methods,
    ALLOWED_QUANTS,
)

# Legacy - kept for backwards compatibility
try:
    from .gguf import GGUFQuantizer
except ImportError:
    GGUFQuantizer = None

# Aliases for backwards compatibility
convert_to_gguf = export_to_gguf
GGUF_QUANT_TYPES = ALLOWED_QUANTS

__all__ = [
    # Core quantization
    "QuantizationConfig",
    "QuantizedLinear", 
    "QuantizationEngine",
    # GGUF conversion (new - recommended)
    "GGUFExporter",
    "export_to_gguf",
    "print_quantization_methods",
    "ALLOWED_QUANTS",
    # Backwards compatibility
    "convert_to_gguf",
    "GGUF_QUANT_TYPES",
    "GGUFQuantizer",
]
