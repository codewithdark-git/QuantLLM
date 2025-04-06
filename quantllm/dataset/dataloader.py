from typing import Optional, Union, Dict, Any
import torch
from torch.utils.data import DataLoader, Dataset
from .dataset_preprocessor import DatasetPreprocessor

class DataLoader(DataLoader):
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
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )
        
    @classmethod
    def from_datasets(
        cls,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        **kwargs
    ) -> Union['DataLoader', Dict[str, 'DataLoader']]:
        """
        Create DataLoader(s) from one or more datasets.
        
        Args:
            train_dataset (Dataset): Training dataset
            val_dataset (Optional[Dataset]): Validation dataset
            test_dataset (Optional[Dataset]): Test dataset
            batch_size (int): Number of samples per batch
            num_workers (int): Number of worker processes
            **kwargs: Additional arguments to pass to the DataLoader
            
        Returns:
            Union[QuantLLMDataLoader, Dict[str, QuantLLMDataLoader]]: 
                Single DataLoader if only train_dataset is provided,
                or dictionary of DataLoaders if multiple datasets are provided
        """
        train_loader = cls(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            **kwargs
        )
        
        if val_dataset is None and test_dataset is None:
            return train_loader
            
        loaders = {"train": train_loader}
        
        if val_dataset is not None:
            loaders["val"] = cls(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                **kwargs
            )
            
        if test_dataset is not None:
            loaders["test"] = cls(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                **kwargs
            )
            
        return loaders
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        Get a single batch from the DataLoader.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the batch data
        """
        try:
            batch = next(iter(self))
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
        return len(self) 