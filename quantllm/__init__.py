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
from .utils import (
    get_optimal_training_settings,
    configure_logging,
    enable_logging,
    QuantizationBenchmark
)
from .api import QuantLLM

from .quant import (
    QuantizationConfig, 
    QuantizationEngine, 
    QuantizedLinear, 
    GGUFQuantizer, 
    GPTQQuantizer, 
    AWQQuantizer
)


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
    "QuantizationBenchmark",
    
    # Utilities
    "get_optimal_training_settings",
    "configure_logging",
    "enable_logging",

    # Quantization
    "QuantizationConfig",
    "QuantizationEngine",
    "QuantizedLinear",
    "GGUFQuantizer",
    "GPTQQuantizer",
    "AWQQuantizer",

    # API
    "QuantLLM"
]


# Initialize package-level logger with fancy welcome message
logger = TrainingLogger()
logger.log_welcome_message()