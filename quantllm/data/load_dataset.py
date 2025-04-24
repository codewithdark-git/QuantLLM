from datasets import load_dataset, Dataset, disable_progress_bars
from typing import Optional, Dict, Any, Union
import os
from ..trainer.logger import TrainingLogger
from tqdm.auto import tqdm
import logging
import warnings

# Disable unnecessary logging
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

class LoadDataset:
    def __init__(self, logger: Optional[TrainingLogger] = None):
        """
        Initialize the dataset loader.
        
        Args:
            logger (TrainingLogger, optional): Logger instance
        """
        self.logger = logger or TrainingLogger()
        disable_progress_bars()  # Disable default progress bars
        
    def load_hf_dataset(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        streaming: bool = False,
        **kwargs
    ) -> Dataset:
        """
        Load a dataset from HuggingFace with custom progress bar.
        
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
            
            # Create progress bar
            with tqdm(total=1, desc=f"Downloading {dataset_name}", unit="dataset") as pbar:
                dataset = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=streaming,
                    **kwargs
                )
                pbar.update(1)
            
            self.logger.log_info(f"Successfully loaded dataset: {dataset_name}")
            return dataset
            
        except Exception as e:
            self.logger.log_error(f"Error loading dataset: {str(e)}")
            raise
            
    def load_local_dataset(
        self,
        file_path: str,
        file_type: str = "auto",
        **kwargs
    ) -> Dataset:
        """
        Load a dataset from local file with progress bar.
        
        Args:
            file_path (str): Path to the dataset file
            file_type (str): Type of file (auto, csv, json, text, parquet)
            **kwargs: Additional arguments for dataset loading
        """
        try:
            self.logger.log_info(f"Loading local dataset: {file_path}")
            
            if file_type == "auto":
                extension = os.path.splitext(file_path)[1][1:].lower()
                if extension in ["csv", "json", "txt", "parquet"]:
                    file_type = "text" if extension == "txt" else extension
                else:
                    raise ValueError(f"Unsupported file extension: {extension}")
            
            # Create progress bar
            with tqdm(total=1, desc=f"Loading {file_path}", unit="file") as pbar:
                dataset = load_dataset(
                    file_type,
                    data_files=file_path,
                    **kwargs
                )
                pbar.update(1)
            
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
        Load a custom dataset from data with progress indication.
        
        Args:
            data (Union[Dict, list]): Dataset data
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            Dataset: Created dataset
        """
        try:
            self.logger.log_info("Creating custom dataset")
            
            # Create progress bar
            with tqdm(total=1, desc="Creating dataset", unit="dataset") as pbar:
                dataset = Dataset.from_dict(data, **kwargs)
                pbar.update(1)
            
            self.logger.log_info("Successfully created custom dataset")
            return dataset
            
        except Exception as e:
            self.logger.log_error(f"Error creating custom dataset: {str(e)}")
            raise