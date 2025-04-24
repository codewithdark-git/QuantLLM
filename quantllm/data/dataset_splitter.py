from datasets import Dataset, DatasetDict
from typing import Optional, Tuple, Union
import numpy as np
from ..trainer.logger import TrainingLogger
from tqdm.auto import tqdm
import logging

# Configure logging
logging.getLogger("datasets").setLevel(logging.WARNING)

class DatasetSplitter:
    def __init__(self, logger: Optional[TrainingLogger] = None):
        """Initialize dataset splitter."""
        self.logger = logger or TrainingLogger()

    def _get_dataset_from_dict(self, dataset: Union[Dataset, DatasetDict], split: str = "train") -> Dataset:
        """Extract dataset from DatasetDict if needed."""
        if isinstance(dataset, DatasetDict):
            if split in dataset:
                return dataset[split]
            raise ValueError(f"DatasetDict does not contain split '{split}'")
        return dataset

    def validate_split_params(self, train_size: float, val_size: float, test_size: float = None):
        """Validate split parameters."""
        if train_size <= 0 or train_size >= 1:
            raise ValueError(f"train_size must be between 0 and 1, got {train_size}")
        if val_size <= 0 or val_size >= 1:
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
        if test_size is not None and (test_size <= 0 or test_size >= 1):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
            
        total = train_size + val_size + (test_size or (1 - train_size - val_size))
        if not (0.99 <= total <= 1.01):  # Allow small floating point differences
            raise ValueError(f"Split sizes must sum to 1.0, got {total}")
            
    def train_test_split(
        self,
        dataset: Dataset,
        test_size: float = 0.2,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs
    ) -> Tuple[Dataset, Dataset]:
        """
        Split dataset into train and test sets.
        
        Args:
            dataset (Dataset): Dataset to split
            test_size (float): Size of test set
            shuffle (bool): Whether to shuffle
            seed (int): Random seed
            **kwargs: Additional splitting arguments
            
        Returns:
            Tuple[Dataset, Dataset]: Train and test datasets
        """
        try:
            self.logger.log_info("Splitting dataset into train and test sets")
            split_dataset = dataset.train_test_split(
                test_size=test_size,
                shuffle=shuffle,
                seed=seed,
                **kwargs
            )
            self.logger.log_info("Successfully split dataset")
            return split_dataset["train"], split_dataset["test"]
            
        except Exception as e:
            self.logger.log_error(f"Error splitting dataset: {str(e)}")
            raise
            
    def train_val_test_split(
        self,
        dataset: Union[Dataset, DatasetDict],
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
        split: str = "train"
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train, validation and test sets with progress indication.
        
        Args:
            dataset (Dataset or DatasetDict): Dataset to split
            train_size (float): Proportion of training set
            val_size (float): Proportion of validation set
            test_size (float): Proportion of test set
            shuffle (bool): Whether to shuffle the dataset
            seed (int): Random seed
            split (str): Which split to use if dataset is a DatasetDict
            
        Returns:
            Tuple[Dataset, Dataset, Dataset]: Train, validation and test datasets
        """
        try:
            # Get the actual dataset if we have a DatasetDict
            dataset = self._get_dataset_from_dict(dataset, split)
            
            # Validate split proportions
            total = train_size + val_size + test_size
            if not np.isclose(total, 1.0):
                raise ValueError(f"Split proportions must sum to 1, got {total}")
            
            # Calculate split sizes
            total_size = len(dataset)
            train_samples = int(total_size * train_size)
            val_samples = int(total_size * val_size)
            test_samples = total_size - train_samples - val_samples
            
            self.logger.log_info("Splitting dataset...")
            
            # Create indices
            indices = np.arange(total_size)
            if shuffle:
                with tqdm(total=1, desc="Shuffling dataset", unit="operation") as pbar:
                    rng = np.random.default_rng(seed)
                    rng.shuffle(indices)
                    pbar.update(1)
            
            # Split dataset using Hugging Face's built-in functionality
            with tqdm(total=2, desc="Creating splits", unit="split") as pbar:
                # First split: train vs rest
                train_val_split = dataset.train_test_split(
                    train_size=train_size,
                    seed=seed,
                    shuffle=False  # We already shuffled if needed
                )
                train_dataset = train_val_split["train"]
                rest_dataset = train_val_split["test"]
                pbar.update(1)
                
                # Second split: val vs test from the rest
                val_ratio = val_size / (val_size + test_size)
                val_test_split = rest_dataset.train_test_split(
                    train_size=val_ratio,
                    seed=seed,
                    shuffle=False
                )
                val_dataset = val_test_split["train"]
                test_dataset = val_test_split["test"]
                pbar.update(1)
            
            # Log split sizes
            self.logger.log_info(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            self.logger.log_error(f"Error splitting dataset: {str(e)}")
            raise

    def train_val_split(
        self,
        dataset: Union[Dataset, DatasetDict],
        train_size: float = 0.8,
        shuffle: bool = True,
        seed: int = 42,
        split: str = "train"
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and validation sets."""
        dataset = self._get_dataset_from_dict(dataset, split)
        return dataset.train_test_split(
            train_size=train_size,
            shuffle=shuffle,
            seed=seed
        ).values()

    def k_fold_split(self, dataset, n_splits: int = 5, shuffle: bool = True, seed: int = 42):
        """Create k-fold cross validation splits."""
        try:
            if not isinstance(dataset, Dataset):
                raise ValueError(f"Expected Dataset object, got {type(dataset)}")
                
            if n_splits < 2:
                raise ValueError(f"n_splits must be at least 2, got {n_splits}")
                
            # Convert to pandas for k-fold split
            df = dataset.to_pandas()
            
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
            
            folds = []
            for train_idx, val_idx in kf.split(df):
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
                
                train_dataset = Dataset.from_pandas(train_df)
                val_dataset = Dataset.from_pandas(val_df)
                
                folds.append((train_dataset, val_dataset))
                
            self.logger.log_info(f"Created {n_splits}-fold cross validation splits")
            return folds
            
        except Exception as e:
            self.logger.log_error(f"Error creating k-fold splits: {str(e)}")
            raise