from datasets import Dataset
from typing import Optional, Dict, Any, Tuple
from ..finetune.logger import TrainingLogger

class DatasetSplitter:
    def __init__(self, logger: Optional[TrainingLogger] = None):
        """
        Initialize the dataset splitter.
        
        Args:
            logger (TrainingLogger, optional): Logger instance
        """
        self.logger = logger or TrainingLogger()
        
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
        dataset: Dataset,
        val_size: float = 0.1,
        test_size: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset (Dataset): Dataset to split
            val_size (float): Size of validation set
            test_size (float): Size of test set
            shuffle (bool): Whether to shuffle
            seed (int): Random seed
            **kwargs: Additional splitting arguments
            
        Returns:
            Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets
        """
        try:
            self.logger.log_info("Splitting dataset into train, validation, and test sets")
            
            # First split into train and temp
            train_size = 1 - val_size - test_size
            temp_size = val_size + test_size
            train_dataset, temp_dataset = dataset.train_test_split(
                test_size=temp_size,
                shuffle=shuffle,
                seed=seed,
                **kwargs
            )
            
            # Then split temp into validation and test
            val_ratio = val_size / temp_size
            val_dataset, test_dataset = temp_dataset.train_test_split(
                test_size=1 - val_ratio,
                shuffle=shuffle,
                seed=seed,
                **kwargs
            )
            
            self.logger.log_info("Successfully split dataset")
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            self.logger.log_error(f"Error splitting dataset: {str(e)}")
            raise
            
    def k_fold_split(
        self,
        dataset: Dataset,
        n_splits: int = 5,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs
    ) -> list:
        """
        Split dataset into k folds.
        
        Args:
            dataset (Dataset): Dataset to split
            n_splits (int): Number of folds
            shuffle (bool): Whether to shuffle
            seed (int): Random seed
            **kwargs: Additional splitting arguments
            
        Returns:
            list: List of (train, test) dataset tuples
        """
        try:
            self.logger.log_info(f"Splitting dataset into {n_splits} folds")
            
            # Convert to pandas for k-fold split
            df = dataset.to_pandas()
            
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
            
            folds = []
            for train_idx, test_idx in kf.split(df):
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                
                train_dataset = Dataset.from_pandas(train_df, **kwargs)
                test_dataset = Dataset.from_pandas(test_df, **kwargs)
                
                folds.append((train_dataset, test_dataset))
                
            self.logger.log_info("Successfully created k-fold splits")
            return folds
            
        except Exception as e:
            self.logger.log_error(f"Error creating k-fold splits: {str(e)}")
            raise 