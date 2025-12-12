"""
QuantLLM Core Module

Contains the high-performance turbo API for easy model loading,
quantization, fine-tuning, and export.
"""

from .hardware import HardwareProfiler
from .smart_config import SmartConfig
from .model_analyzer import ModelAnalyzer
from .turbo_model import TurboModel, turbo
from .compilation import (
    compile_model,
    compile_for_inference,
    compile_for_training,
    compile_for_max_speed,
    is_compile_supported,
    CompiledModelWrapper,
)
from .flash_attention import (
    flash_attention,
    is_flash_attention_available,
    enable_flash_attention_for_model,
    FlashAttentionWrapper,
)
from .memory import (
    MemoryManager,
    DynamicOffloader,
    GradientCheckpointManager,
    CPUOffloadOptimizer,
    setup_memory_efficient_training,
)
from .training import (
    AutoBatchSizeFinder,
    LoRAAutoConfig,
    TrainingConfig,
    TrainingCallbacks,
    auto_configure_training,
    load_training_data,
)
from .export import (
    UniversalExporter,
    ExportFormat,
    export_model,
)

__all__ = [
    # Main API
    "HardwareProfiler",
    "SmartConfig", 
    "ModelAnalyzer",
    "TurboModel",
    "turbo",
    # Compilation
    "compile_model",
    "compile_for_inference",
    "compile_for_training",
    "compile_for_max_speed",
    "is_compile_supported",
    "CompiledModelWrapper",
    # Flash Attention
    "flash_attention",
    "is_flash_attention_available",
    "enable_flash_attention_for_model",
    "FlashAttentionWrapper",
    # Memory Optimization
    "MemoryManager",
    "DynamicOffloader",
    "GradientCheckpointManager",
    "CPUOffloadOptimizer",
    "setup_memory_efficient_training",
    # Training
    "AutoBatchSizeFinder",
    "LoRAAutoConfig",
    "TrainingConfig",
    "TrainingCallbacks",
    "auto_configure_training",
    "load_training_data",
    # Export
    "UniversalExporter",
    "ExportFormat",
    "export_model",
]
