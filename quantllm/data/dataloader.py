from typing import Optional, Union, Dict, Any
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset, TensorDataset
from datasets import Dataset as HFDataset
from .dataset_preprocessor import DatasetPreprocessor

class DataLoader:
    """
    Custom DataLoader class for QuantLLM that wraps torch.utils.data.DataLoader.
    """
    
    @staticmethod
    def validate_dataset(dataset, name: str):
        """Validate dataset."""
        if dataset is None:
            return
        if not isinstance(dataset, (Dataset, HFDataset)):
            raise ValueError(f"{name} must be a PyTorch Dataset or HuggingFace Dataset object, got {type(dataset)}")
            
    @classmethod
    def from_datasets(
        cls,
        train_dataset,
        val_dataset=None,
        test_dataset=None,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs
    ):
        """Create DataLoader instances from datasets."""
        try:
            # Validate inputs
            cls.validate_dataset(train_dataset, "train_dataset")
            cls.validate_dataset(val_dataset, "val_dataset")
            cls.validate_dataset(test_dataset, "test_dataset")
            
            if batch_size <= 0:
                raise ValueError(f"batch_size must be positive, got {batch_size}")
                
            def prepare_dataset(dataset):
                if dataset is None:
                    return None
                    
                if isinstance(dataset, HFDataset):
                    # Ensure all required features are present
                    required_features = ['input_ids', 'attention_mask', 'labels']
                    if not all(feature in dataset.features for feature in required_features):
                        raise ValueError(f"Dataset must contain all required features: {required_features}")
                    
                    # Get feature dimensions
                    sample_len = len(dataset[0]['input_ids'])
                    total_samples = len(dataset)
                    
                    # Pre-allocate tensors
                    input_ids = torch.zeros((total_samples, sample_len), dtype=torch.long)
                    attention_mask = torch.zeros((total_samples, sample_len), dtype=torch.long)
                    labels = torch.zeros((total_samples, sample_len), dtype=torch.long)
                    
                    # Fill tensors
                    for i in range(total_samples):
                        input_ids[i] = torch.tensor(dataset[i]['input_ids'])
                        attention_mask[i] = torch.tensor(dataset[i]['attention_mask'])
                        labels[i] = torch.tensor(dataset[i]['labels'])
                    
                    return TensorDataset(input_ids, attention_mask, labels)
                
                return dataset

            train_dataset = prepare_dataset(train_dataset)
            val_dataset = prepare_dataset(val_dataset)
            test_dataset = prepare_dataset(test_dataset)
            
            # Create DataLoaders with consistent batch sizes
            train_loader = TorchDataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=True,  # Drop last incomplete batch
                **kwargs
            ) if train_dataset is not None else None
            
            val_loader = TorchDataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=True,  # Drop last incomplete batch
                **kwargs
            ) if val_dataset is not None else None
            
            test_loader = TorchDataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=True,  # Drop last incomplete batch
                **kwargs
            ) if test_dataset is not None else None
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"Error creating data loaders: {str(e)}")
            raise