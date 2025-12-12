
from .data import (
    LoadDataset,
    DatasetPreprocessor,
    DatasetSplitter,
    DataLoader
)
from .trainer import (
    FineTuningTrainer,
    ModelEvaluator
)
from .hub import HubManager, CheckpointManager
from .utils import (
    get_optimal_training_settings,
    configure_logging,
    enable_logging,
    QuantizationBenchmark,
    MemoryTracker,
)
from .api import QuantLLM

from .quant import (
    QuantizationConfig, 
    QuantizationEngine, 
    QuantizedLinear, 
    GGUFQuantizer
)


from .config import (
    ModelConfig,
    DatasetConfig,
    TrainingConfig
)

# NEW: Turbo API - The ultra-simple way to use QuantLLM
from .core import (
    turbo,
    TurboModel,
    SmartConfig,
    HardwareProfiler,
    ModelAnalyzer,
)

# Configure package-wide logging
configure_logging()

__version__ = "2.0.0"

# Package metadata
__title__ = "QuantLLM"
__description__ = "Ultra-fast LLM Quantization - Faster and Simpler than Unsloth"
__author__ = "QuantLLM Team"

__all__ = [
    # ====== NEW TURBO API (Recommended) ======
    "turbo",           # One-liner model loading
    "TurboModel",      # Full-featured model class
    "SmartConfig",     # Auto-configuration
    "HardwareProfiler", # Hardware detection
    "ModelAnalyzer",   # Model architecture analysis
    
    # ====== Legacy API ======
    # Dataset
    "DataLoader",
    "DatasetPreprocessor",
    "DatasetSplitter",
    "LoadDataset",
    
    # Training
    "FineTuningTrainer",
    "ModelEvaluator",
    
    # Hub and Checkpoint
    "HubManager",
    "CheckpointManager",
    
    # Configuration
    "ModelConfig",
    "DatasetConfig",
    "TrainingConfig",
    "QuantizationBenchmark",
    
    # Utilities
    "get_optimal_training_settings",
    "configure_logging",
    "enable_logging",
    "MemoryTracker",

    # Quantization
    "QuantizationConfig",
    "QuantizationEngine",
    "QuantizedLinear",
    "GGUFQuantizer",

    # API
    "QuantLLM",
]
