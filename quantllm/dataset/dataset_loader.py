from datasets import load_dataset, Dataset
from typing import Optional, Dict, Any, Union
from ..finetune.logger import TrainingLogger

class LoadDataset:
    def __init__(self, logger: Optional[TrainingLogger] = None):
        """
        Initialize the dataset loader.
        
        Args:
            logger (TrainingLogger, optional): Logger instance
        """
        self.logger = logger or TrainingLogger()
        
    def load_hf_dataset(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        streaming: bool = False,
        **kwargs
    ) -> Dataset:
        """
        Load a dataset from HuggingFace.
        
        Args:
            dataset_name (str): Name of the dataset
            split (str, optional): Dataset split
            streaming (bool): Whether to use streaming
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            Dataset: Loaded dataset
        """
        try:
            self.logger.log_info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=streaming,
                **kwargs
            )
            self.logger.log_info(f"Successfully loaded dataset: {dataset_name}")
            return dataset
            
        except Exception as e:
            self.logger.log_error(f"Error loading dataset: {str(e)}")
            raise
            
    def load_local_dataset(
        self,
        file_path: str,
        file_type: str = "csv",
        **kwargs
    ) -> Dataset:
        """
        Load a dataset from local file.
        
        Args:
            file_path (str): Path to the dataset file
            file_type (str): Type of file (csv, json, etc.)
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            Dataset: Loaded dataset
        """
        try:
            self.logger.log_info(f"Loading local dataset: {file_path}")
            dataset = load_dataset(
                file_type,
                data_files=file_path,
                **kwargs
            )
            self.logger.log_info(f"Successfully loaded local dataset: {file_path}")
            return dataset
            
        except Exception as e:
            self.logger.log_error(f"Error loading local dataset: {str(e)}")
            raise
            
    def load_custom_dataset(
        self,
        data: Union[Dict, list],
        **kwargs
    ) -> Dataset:
        """
        Load a custom dataset from data.
        
        Args:
            data (Union[Dict, list]): Dataset data
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            Dataset: Created dataset
        """
        try:
            self.logger.log_info("Creating custom dataset")
            dataset = Dataset.from_dict(data, **kwargs)
            self.logger.log_info("Successfully created custom dataset")
            return dataset
            
        except Exception as e:
            self.logger.log_error(f"Error creating custom dataset: {str(e)}")
            raise 