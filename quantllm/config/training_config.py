from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    weight_decay: float = 0.01
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "linear"
    lr_scheduler_kwargs: Dict[str, Any] = None
    
    # Training settings
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    # Logging and evaluation
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Paths
    output_dir: str = "outputs"
    logging_dir: str = "logs"
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}
            
        # Convert paths to Path objects
        self.output_dir = Path(self.output_dir)
        self.logging_dir = Path(self.logging_dir)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "lr_scheduler_kwargs": self.lr_scheduler_kwargs,
            "seed": self.seed,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "output_dir": str(self.output_dir),
            "logging_dir": str(self.logging_dir)
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
        
    def validate(self) -> bool:
        """Validate configuration values."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive")
        if self.warmup_steps < 0:
            raise ValueError("Warmup steps cannot be negative")
        if self.weight_decay < 0:
            raise ValueError("Weight decay cannot be negative")
        if self.early_stopping_patience < 0:
            raise ValueError("Early stopping patience cannot be negative")
        if self.early_stopping_threshold < 0:
            raise ValueError("Early stopping threshold cannot be negative")
        if self.logging_steps <= 0:
            raise ValueError("Logging steps must be positive")
        if self.eval_steps <= 0:
            raise ValueError("Evaluation steps must be positive")
        if self.save_steps <= 0:
            raise ValueError("Save steps must be positive")
        if self.save_total_limit < 0:
            raise ValueError("Save total limit cannot be negative")
            
        return True 