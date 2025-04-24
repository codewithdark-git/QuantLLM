from datasets import Dataset
from typing import Optional, Dict, Any, Callable, Tuple
from transformers import PreTrainedTokenizer
from ..trainer.logger import TrainingLogger
from tqdm.auto import tqdm
import logging
import warnings

# Disable unnecessary logging
logging.getLogger("tokenizers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

class DatasetPreprocessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, logger: Optional[TrainingLogger] = None):
        self.tokenizer = tokenizer
        self.logger = logger or TrainingLogger()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.log_info("Set pad token to eos token")

    def validate_datasets(self, datasets):
        """Validate input datasets."""
        for dataset in datasets:
            if dataset is not None and not isinstance(dataset, Dataset):
                raise ValueError(f"Expected Dataset object, got {type(dataset)}")

    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text:
            return ""
        text = str(text).strip()
        text = " ".join(text.split())  # Normalize whitespace
        return text

    def tokenize_dataset(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        max_length: int = 512,
        text_column: str = "text",
        label_column: Optional[str] = None,
        batch_size: int = 1000
    ) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        """Tokenize datasets with preprocessing and progress bars."""
        try:
            self.validate_datasets([train_dataset, val_dataset, test_dataset])
            
            def process_and_tokenize_batch(examples):
                # Get texts and preprocess with progress indication
                texts = examples[text_column]
                if not isinstance(texts, list):
                    texts = [texts]
                
                # Preprocess texts
                texts = [self.preprocess_text(text) for text in texts]
                
                try:
                    # Tokenize with padding and truncation
                    tokenized = self.tokenizer(
                        texts,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length + 1,  # Add 1 for shift
                        return_tensors=None
                    )
                    
                    # For causal language modeling, prepare shifted sequences
                    input_ids = tokenized["input_ids"]
                    attention_mask = tokenized["attention_mask"]
                    
                    # Prepare shifted sequences for input and labels
                    labels = [ids[1:] for ids in input_ids]
                    input_ids = [ids[:-1] for ids in input_ids]
                    attention_mask = [mask[:-1] for mask in attention_mask]
                    
                    # Verify sequence lengths
                    expected_length = max_length
                    assert all(len(seq) == expected_length for seq in input_ids), "Input sequence lengths don't match"
                    assert all(len(seq) == expected_length for seq in attention_mask), "Attention mask lengths don't match"
                    assert all(len(seq) == expected_length for seq in labels), "Label sequence lengths don't match"
                    
                    result = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
                    
                    return result
                    
                except Exception as e:
                    self.logger.log_error(f"Error tokenizing batch: {str(e)}")
                    raise
            
            # Process datasets with overall progress bars
            self.logger.log_info("Processing training dataset")
            train_tokenized = train_dataset.map(
                process_and_tokenize_batch,
                batched=True,
                batch_size=batch_size,
                remove_columns=train_dataset.column_names,
                desc="Tokenizing training set"
            )
            self.logger.log_success(f"Tokenized training dataset: {len(train_tokenized)} examples")
            
            val_tokenized = None
            if val_dataset is not None:
                self.logger.log_info("Processing validation dataset")
                val_tokenized = val_dataset.map(
                    process_and_tokenize_batch,
                    batched=True,
                    batch_size=batch_size,
                    remove_columns=val_dataset.column_names,
                    desc="Tokenizing validation set"
                )
                self.logger.log_success(f"Tokenized validation dataset: {len(val_tokenized)} examples")
                
            test_tokenized = None
            if test_dataset is not None:
                self.logger.log_info("Processing test dataset")
                test_tokenized = test_dataset.map(
                    process_and_tokenize_batch,
                    batched=True,
                    batch_size=batch_size,
                    remove_columns=test_dataset.column_names,
                    desc="Tokenizing test set"
                )
                self.logger.log_success(f"Tokenized test dataset: {len(test_tokenized)} examples")
            
            # Set format to PyTorch tensors
            train_tokenized.set_format("torch")
            if val_tokenized:
                val_tokenized.set_format("torch")
            if test_tokenized:
                test_tokenized.set_format("torch")
                
            return train_tokenized, val_tokenized, test_tokenized
            
        except Exception as e:
            self.logger.log_error(f"Error in tokenization: {str(e)}")
            raise