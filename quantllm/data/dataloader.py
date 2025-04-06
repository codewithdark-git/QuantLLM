from typing import Optional, Union, Dict, Any
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset, TensorDataset
from datasets import Dataset as HFDataset
from .dataset_preprocessor import DatasetPreprocessor

class DataLoader(TorchDataLoader):
    """
    Custom DataLoader class for QuantLLM that inherits from torch.utils.data.DataLoader.
    Provides additional functionality and easier integration with the QuantLLM package.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        """
        Initialize the QuantLLM DataLoader.
        
        Args:
            dataset (Dataset): The dataset to load
            batch_size (int): Number of samples per batch
            shuffle (bool): Whether to shuffle the data
            num_workers (int): Number of worker processes for data loading
            pin_memory (bool): Whether to pin memory for faster data transfer to GPU
            drop_last (bool): Whether to drop the last incomplete batch
            **kwargs: Additional arguments to pass to the DataLoader
        """
        self.loader = TorchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )
        self.dataset = dataset
        self.batch_size = batch_size

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
                
            # Convert HuggingFace Dataset to PyTorch Dataset if needed
            def convert_to_torch_dataset(hf_dataset):
                if hf_dataset is None:
                    return None
                if isinstance(hf_dataset, HFDataset):
                    return hf_dataset.with_format("torch")
                return hf_dataset

            train_dataset = convert_to_torch_dataset(train_dataset)
            val_dataset = convert_to_torch_dataset(val_dataset)
            test_dataset = convert_to_torch_dataset(test_dataset)
            
            train_loader = TorchDataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                **kwargs
            ) if train_dataset is not None else None
            
            val_loader = TorchDataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                **kwargs
            ) if val_dataset is not None else None
            
            test_loader = TorchDataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                **kwargs
            ) if test_dataset is not None else None
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"Error creating data loaders: {str(e)}")
            raise
            
    @classmethod
    def from_tensors(
        cls,
        input_ids,
        attention_mask,
        labels=None,
        batch_size: int = 8,
        **kwargs
    ):
        """Create DataLoader from tensor inputs."""
        try:
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask)
                
            if labels is not None:
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                dataset = TensorDataset(input_ids, attention_mask, labels)
            else:
                dataset = TensorDataset(input_ids, attention_mask)
                
            return TorchDataLoader(
                dataset,
                batch_size=batch_size,
                **kwargs
            )
            
        except Exception as e:
            raise RuntimeError(f"Error creating data loader from tensors: {str(e)}")
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        Get a single batch from the DataLoader.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the batch data
        """
        try:
            batch = next(iter(self.loader))
            return batch
        except StopIteration:
            raise RuntimeError("No more batches available in the DataLoader")
            
    def get_batch_size(self) -> int:
        """
        Get the current batch size of the DataLoader.
        
        Returns:
            int: Current batch size
        """
        return self.batch_size
        
    def get_dataset_size(self) -> int:
        """
        Get the size of the underlying dataset.
        
        Returns:
            int: Size of the dataset
        """
        return len(self.dataset)
        
    def get_num_batches(self) -> int:
        """
        Get the total number of batches in the DataLoader.
        
        Returns:
            int: Total number of batches
        """
        return len(self.loader)