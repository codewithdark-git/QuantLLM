from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    
    # Dataset identification
    dataset_name: str
    dataset_type: str = "huggingface"  # huggingface, local, custom
    dataset_revision: str = "main"
    dataset_split: Optional[str] = None
    
    # Dataset columns
    text_column: str = "text"
    label_column: Optional[str] = None
    input_column: Optional[str] = None
    target_column: Optional[str] = None
    
    # Dataset processing
    max_length: int = 512
    truncation: bool = True
    padding: Union[str, bool] = "max_length"
    preprocessing_num_workers: int = 1
    overwrite_cache: bool = False
    remove_columns: Optional[List[str]] = None
    
    # Dataset splitting
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    shuffle: bool = True
    seed: int = 42
    
    # Data augmentation
    augmentation_config: Optional[Dict[str, Any]] = None
    use_augmentation: bool = False
    
    # Local dataset settings
    data_dir: Optional[str] = None
    data_files: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None
    file_format: str = "auto"  # auto, csv, json, text
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.remove_columns is None:
            self.remove_columns = []
            
        if self.augmentation_config is None:
            self.augmentation_config = {}
            
        if self.data_dir is not None:
            self.data_dir = Path(self.data_dir)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type,
            "dataset_revision": self.dataset_revision,
            "dataset_split": self.dataset_split,
            "text_column": self.text_column,
            "label_column": self.label_column,
            "input_column": self.input_column,
            "target_column": self.target_column,
            "max_length": self.max_length,
            "truncation": self.truncation,
            "padding": self.padding,
            "preprocessing_num_workers": self.preprocessing_num_workers,
            "overwrite_cache": self.overwrite_cache,
            "remove_columns": self.remove_columns,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "augmentation_config": self.augmentation_config,
            "use_augmentation": self.use_augmentation,
            "data_dir": str(self.data_dir) if self.data_dir else None,
            "data_files": self.data_files,
            "file_format": self.file_format
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DatasetConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
        
    def validate(self) -> bool:
        """Validate configuration values."""
        if not self.dataset_name:
            raise ValueError("Dataset name or path is required")
            
        if self.dataset_type not in ["huggingface", "local", "custom"]:
            raise ValueError("Invalid dataset type")
            
        if self.max_length <= 0:
            raise ValueError("Max length must be positive")
            
        if self.preprocessing_num_workers < 0:
            raise ValueError("Preprocessing number of workers cannot be negative")
            
        if self.train_size <= 0 or self.train_size >= 1:
            raise ValueError("Train size must be between 0 and 1")
        if self.val_size <= 0 or self.val_size >= 1:
            raise ValueError("Validation size must be between 0 and 1")
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("Test size must be between 0 and 1")
            
        total_size = self.train_size + self.val_size + self.test_size
        if abs(total_size - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test sizes must sum to 1")
            
        if isinstance(self.padding, str) and self.padding not in ["max_length", "longest", "do_not_pad"]:
            raise ValueError("Padding must be True, False, 'max_length', 'longest', or 'do_not_pad'")
            
        if self.file_format not in ["auto", "csv", "json", "text", "parquet"]:
            raise ValueError("File format must be 'auto', 'csv', 'json', 'text', or 'parquet'")
            
        if self.data_dir is not None and not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
            
        return True