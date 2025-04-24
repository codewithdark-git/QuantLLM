from .model import Model
from .data import (
    LoadDataset,
    DatasetPreprocessor,
    DatasetSplitter,
    DataLoader
)
from .trainer import (
    FineTuningTrainer,
    ModelEvaluator,
    TrainingLogger
)
from .hub import HubManager, CheckpointManager
from .utils.optimizations import get_optimal_training_settings
from .utils.log_config import configure_logging, enable_logging

from .config import (
    ModelConfig,
    DatasetConfig,
    TrainingConfig
)

# Configure package-wide logging
configure_logging()

__version__ = "0.1.0"

# Package metadata
__title__ = "QuantLLM"
__description__ = "Efficient Quantized LLM Fine-Tuning Library"
__author__ = "QuantLLM Team"

__all__ = [
    # Model
    "Model",
    
    # Dataset
    "DataLoader",
    "DatasetPreprocessor",
    "DatasetSplitter",
    "LoadDataset",
    
    # Training
    "FineTuningTrainer",
    "ModelEvaluator",
    "TrainingLogger",
    
    # Hub and Checkpoint
    "HubManager",
    "CheckpointManager",
    
    # Configuration
    "ModelConfig",
    "DatasetConfig",
    "TrainingConfig",
    
    # Utilities
    "get_optimal_training_settings",
    "configure_logging",
    "enable_logging",
]

# Initialize package-level logger with fancy welcome message
logger = TrainingLogger()
logger.log_success(f"""
✨ QuantLLM v{__version__} initialized successfully ✨
🚀 Efficient Quantized Language Model Fine-Tuning
📚 Documentation: https://github.com/yourusername/QuantLLM
""")