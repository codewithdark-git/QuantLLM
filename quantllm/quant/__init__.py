"""GGUF Quantization for Large Language Models."""

from .quantization_engine import (
    QuantizationConfig,
    QuantizedLinear,
    QuantizationEngine
)
from .gguf import GGUFQuantizer

# New modern GGUF converter
from .gguf_converter import (
    GGUFConverter,
    convert_to_gguf,
    MODEL_TYPE_MAPPING,
    GGUF_QUANT_TYPES,
)

__all__ = [
    # Core quantization
    "QuantizationConfig",
    "QuantizedLinear", 
    "QuantizationEngine",
    # GGUF conversion (legacy)
    "GGUFQuantizer",
    # GGUF conversion (new - recommended)
    "GGUFConverter",
    "convert_to_gguf",
    "MODEL_TYPE_MAPPING",
    "GGUF_QUANT_TYPES",
]
