"""
QuantLLM v2.0 - Ultra-fast LLM Quantization & GGUF Export

The simplest way to load, quantize, fine-tune, and export LLMs.

Features:
    - Load any HuggingFace model with automatic quantization
    - Export to GGUF, ONNX, MLX formats with proper quantization
    - Fine-tune with LoRA
    - Push to HuggingFace Hub with auto-generated model cards

Example:
    >>> from quantllm import turbo
    >>> 
    >>> # Load any model (auto-quantizes to 4-bit)
    >>> model = turbo("meta-llama/Llama-3.2-3B")
    >>> 
    >>> # Generate text
    >>> model.generate("Hello, world!")
    >>> 
    >>> # Export to GGUF with Q4_K_M quantization
    >>> model.export("gguf", "model.Q4_K_M.gguf", quantization="Q4_K_M")
    >>> 
    >>> # Push to HuggingFace Hub
    >>> model.push("username/my-model", format="gguf", quantization="Q4_K_M")
"""

import os
import sys

# ====== MAIN API (Recommended) ======
from .core import (
    turbo,
    TurboModel,
    SmartConfig,
    HardwareProfiler,
    ModelAnalyzer,
)

# ====== GGUF Export & Quantization ======
from .quant import (
    convert_to_gguf,
    quantize_gguf,
    export_to_gguf,
    check_llama_cpp,
    install_llama_cpp,
    ensure_llama_cpp_installed,
    GGUF_QUANT_TYPES,
    QUANT_RECOMMENDATIONS,
)

# ====== Hub Integration ======
from .hub import QuantLLMHubManager, ModelCardGenerator, generate_model_card

# ====== Utilities ======
from .utils import (
    configure_logging,
    enable_logging,
    MemoryTracker,
    QuantLLMProgress,
    print_header,
    print_success,
    print_error,
    print_info,
    print_warning,
    print_banner,
    is_banner_shown,
    reset_banner,
    console,
    QUANTLLM_ORANGE,
)

# Configure logging (minimal by default)
configure_logging("WARNING")

__version__ = "2.0.0"
__title__ = "QuantLLM"
__description__ = "Ultra-fast LLM Quantization & Export (GGUF, ONNX, MLX)"
__author__ = "Dark Coder"


def _should_show_banner() -> bool:
    """Determine if banner should be shown on import."""
    # Check environment variable (can disable with QUANTLLM_BANNER=0)
    env_banner = os.environ.get("QUANTLLM_BANNER", "").lower()
    if env_banner == "0" or env_banner == "false":
        return False
    
    # Don't show in non-interactive environments
    if not sys.stdout.isatty():
        # Unless explicitly enabled
        if env_banner not in ("1", "true"):
            return False
    
    return True


# Show banner on import (only once, automatically)
if _should_show_banner():
    print_banner(__version__)


def show_banner(force: bool = False):
    """
    Display the QuantLLM banner.
    
    Args:
        force: If True, show banner even if already displayed
    """
    print_banner(__version__, force=force)


__all__ = [
    # Main API
    "turbo",
    "TurboModel",
    "SmartConfig",
    "HardwareProfiler",
    "ModelAnalyzer",
    
    # GGUF Export & Quantization
    "convert_to_gguf",
    "quantize_gguf",
    "export_to_gguf",
    "check_llama_cpp",
    "install_llama_cpp",
    "ensure_llama_cpp_installed",
    "GGUF_QUANT_TYPES",
    "QUANT_RECOMMENDATIONS",
    
    # Hub
    "QuantLLMHubManager",
    "ModelCardGenerator",
    "generate_model_card",
    
    # Utils
    "configure_logging",
    "enable_logging",
    "MemoryTracker",
    "QuantLLMProgress",
    "print_header",
    "print_success",
    "print_error",
    "print_info",
    "print_warning",
    "print_banner",
    "show_banner",
    "is_banner_shown",
    "reset_banner",
    "console",
    "QUANTLLM_ORANGE",
]
