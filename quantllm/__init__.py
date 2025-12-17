"""
QuantLLM v2.0 - Ultra-fast LLM Quantization

The simplest way to load, quantize, fine-tune, and export LLMs.

Example:
    >>> from quantllm import turbo
    >>> model = turbo("meta-llama/Llama-2-7b")
    >>> model.generate("Hello, world!")
    >>> model.export("gguf", "model.gguf")
"""

# ====== MAIN API (Recommended) ======
from .core import (
    turbo,
    TurboModel,
    SmartConfig,
    HardwareProfiler,
    ModelAnalyzer,
)

# ====== GGUF Export ======
from .quant import (
    convert_to_gguf,
    export_to_gguf,
    check_llama_cpp,
    install_llama_cpp,
    GGUF_QUANT_TYPES,
)

# ====== Hub Integration ======
from .hub import QuantLLMHubManager

# ====== Utilities ======
from .utils import (
    configure_logging,
    enable_logging,
    MemoryTracker,
)

# Configure logging
configure_logging()

__version__ = "2.0.0"
__title__ = "QuantLLM"
__description__ = "Ultra-fast LLM Quantization - GGUF Export"
__author__ = "QuantLLM Team"

__all__ = [
    # Main API
    "turbo",
    "TurboModel",
    "SmartConfig",
    "HardwareProfiler",
    "ModelAnalyzer",
    
    # GGUF Export
    "convert_to_gguf",
    "export_to_gguf",
    "check_llama_cpp",
    "install_llama_cpp",
    "GGUF_QUANT_TYPES",
    
    # Hub
    "QuantLLMHubManager",
    
    # Utils
    "configure_logging",
    "enable_logging",
    "MemoryTracker",
]
