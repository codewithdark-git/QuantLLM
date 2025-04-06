from datasets import Dataset
from typing import Optional, Dict, Any, Callable
from transformers import PreTrainedTokenizer
from ..finetune.logger import TrainingLogger

class DatasetPreprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        logger: Optional[TrainingLogger] = None
    ):
        """
        Initialize the dataset preprocessor.
        
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer for preprocessing
            logger (TrainingLogger, optional): Logger instance
        """
        self.tokenizer = tokenizer
        self.logger = logger or TrainingLogger()
        
    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "max_length",
        **kwargs
    ) -> Dataset:
        """
        Tokenize a dataset.
        
        Args:
            dataset (Dataset): Dataset to tokenize
            text_column (str): Column containing text
            max_length (int): Maximum sequence length
            truncation (bool): Whether to truncate
            padding (str): Padding strategy
            **kwargs: Additional tokenization arguments
            
        Returns:
            Dataset: Tokenized dataset
        """
        try:
            self.logger.log_info("Tokenizing dataset")
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples[text_column],
                    max_length=max_length,
                    truncation=truncation,
                    padding=padding,
                    **kwargs
                )
                
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            self.logger.log_info("Successfully tokenized dataset")
            return tokenized_dataset
            
        except Exception as e:
            self.logger.log_error(f"Error tokenizing dataset: {str(e)}")
            raise
            
    def apply_custom_preprocessing(
        self,
        dataset: Dataset,
        preprocessing_function: Callable,
        **kwargs
    ) -> Dataset:
        """
        Apply custom preprocessing to dataset.
        
        Args:
            dataset (Dataset): Dataset to preprocess
            preprocessing_function (Callable): Preprocessing function
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Dataset: Preprocessed dataset
        """
        try:
            self.logger.log_info("Applying custom preprocessing")
            processed_dataset = dataset.map(
                preprocessing_function,
                **kwargs
            )
            self.logger.log_info("Successfully applied custom preprocessing")
            return processed_dataset
            
        except Exception as e:
            self.logger.log_error(f"Error in custom preprocessing: {str(e)}")
            raise
            
    def filter_dataset(
        self,
        dataset: Dataset,
        filter_function: Callable,
        **kwargs
    ) -> Dataset:
        """
        Filter dataset based on criteria.
        
        Args:
            dataset (Dataset): Dataset to filter
            filter_function (Callable): Filter function
            **kwargs: Additional filtering arguments
            
        Returns:
            Dataset: Filtered dataset
        """
        try:
            self.logger.log_info("Filtering dataset")
            filtered_dataset = dataset.filter(
                filter_function,
                **kwargs
            )
            self.logger.log_info("Successfully filtered dataset")
            return filtered_dataset
            
        except Exception as e:
            self.logger.log_error(f"Error filtering dataset: {str(e)}")
            raise 