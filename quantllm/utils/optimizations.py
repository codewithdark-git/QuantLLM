import torch
from typing import Dict, Any
from ..trainer.logger import TrainingLogger

logger = TrainingLogger()

def get_optimal_training_settings() -> Dict[str, Any]:
    """
    Determine optimal training settings based on available hardware.
    
    Returns:
        Dict[str, Any]: Dictionary containing optimal training settings
    """
    logger.log_info("🔍 Detecting hardware capabilities...")
    
    if torch.cuda.is_available():
        device_info = torch.cuda.get_device_properties(0)
        logger.log_info(f"💪 Found CUDA GPU: {device_info.name} with {device_info.total_memory / 1024**3:.1f}GB memory")
        settings = {
            "load_in_4bit": True,
            "use_lora": True,
            "gradient_checkpointing": True,
            "bf16": True
        }
        logger.log_success("✓ Configuring for GPU training with 4-bit quantization and LoRA")
    elif torch.backends.mps.is_available():
        logger.log_info("🍎 Found Apple Silicon GPU")
        settings = {
            "cpu_offload": True,
            "gradient_checkpointing": True
        }
        logger.log_success("✓ Configuring for Apple Silicon with gradient checkpointing")
    else:
        logger.log_info("💻 No GPU detected, using CPU")
        settings = {
            "cpu_offload": True,
            "gradient_checkpointing": True
        }
        logger.log_success("✓ Configured CPU optimizations with gradient checkpointing")
    
    return settings