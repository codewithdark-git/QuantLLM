from datasets import Dataset
from typing import Optional, Dict, Any, Tuple
from ..trainer.logger import TrainingLogger

class DatasetSplitter:
    def __init__(self, logger=None):
        self.logger = logger or TrainingLogger()
        
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
            
    def train_val_test_split(self, dataset, train_size: float, val_size: float, test_size: float = None):
        """Split dataset into train, validation and test sets."""
        try:
            if not isinstance(dataset, Dataset):
                if isinstance(dataset, dict) and 'train' in dataset:
                    dataset = dataset['train']
                else:
                    raise ValueError(f"Expected Dataset object or dict with 'train' key, got {type(dataset)}")
                    
            if test_size is None:
                test_size = 1.0 - train_size - val_size
                
            self.validate_split_params(train_size, val_size, test_size)
            
            # If dataset is already split
            if isinstance(dataset, dict) and all(k in dataset for k in ['train', 'validation', 'test']):
                self.logger.log_info("Dataset already contains train/validation/test splits")
                return dataset['train'], dataset['validation'], dataset['test']
            
            # Convert ratios to absolute sizes
            total_size = len(dataset)
            if total_size == 0:
                raise ValueError("Dataset is empty")
                
            train_end = int(total_size * train_size)
            val_end = train_end + int(total_size * val_size)
            
            # Shuffle dataset with seed for reproducibility
            dataset = dataset.shuffle(seed=42)
            
            # Split dataset
            train_dataset = dataset.select(range(train_end))
            val_dataset = dataset.select(range(train_end, val_end))
            test_dataset = dataset.select(range(val_end, total_size))
            
            # Validate split sizes
            if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
                raise ValueError("One or more splits are empty. Try adjusting split ratios.")
                
            self.logger.log_info(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            self.logger.log_error(f"Error splitting dataset: {str(e)}")
            raise
            
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