from datasets import Dataset
from typing import Optional, Dict, Any, Callable
from transformers import PreTrainedTokenizer
from ..trainer.logger import TrainingLogger

class DatasetPreprocessor:
    def __init__(self, tokenizer, logger=None):
        self.tokenizer = tokenizer
        self.logger = logger or TrainingLogger()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print("Set pad token to eos token")

    def validate_datasets(self, datasets):
        """Validate input datasets."""
        for dataset in datasets:
            if dataset is not None and not isinstance(dataset, Dataset):
                raise ValueError(f"Expected Dataset object, got {type(dataset)}")

    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if not text:
            return ""
        text = str(text).strip()
        text = " ".join(text.split())  # Normalize whitespace
        return text

    def tokenize_dataset(
        self,
        train_dataset,
        val_dataset=None,
        test_dataset=None,
        max_length: int = 512,
        text_column: str = "text",
        label_column: str = None,
        batch_size: int = 1000
    ):
        """Tokenize datasets with preprocessing."""
        try:
            self.validate_datasets([train_dataset, val_dataset, test_dataset])
            
            def process_and_tokenize_batch(examples):
                # Get texts and preprocess
                texts = examples[text_column]
                if not isinstance(texts, list):
                    texts = [texts]
                texts = [self.preprocess_text(text) for text in texts]
                
                try:
                    # Tokenize with padding and truncation
                    # Use max_length + 1 to account for the shift we'll do later
                    tokenized = self.tokenizer(
                        texts,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length + 1,  # Add 1 to account for shift
                        return_tensors=None
                    )
                    
                    input_ids = tokenized["input_ids"]
                    attention_mask = tokenized["attention_mask"]
                    
                    # Now shift to create inputs and labels
                    # inputs will be [:-1] and labels will be [1:]
                    labels = [ids[1:] for ids in input_ids]
                    input_ids = [ids[:-1] for ids in input_ids]
                    attention_mask = [mask[:-1] for mask in attention_mask]
                    
                    # Verify all sequences have the expected length
                    expected_length = max_length
                    if not all(len(seq) == expected_length for seq in input_ids):
                        raise ValueError(f"Input sequence lengths don't match. Expected {expected_length}")
                    if not all(len(seq) == expected_length for seq in attention_mask):
                        raise ValueError(f"Attention mask lengths don't match. Expected {expected_length}")
                    if not all(len(seq) == expected_length for seq in labels):
                        raise ValueError(f"Label sequence lengths don't match. Expected {expected_length}")
                    
                    result = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
                    
                    self.logger.log_info(f"Tokenized batch of {len(texts)} texts")
                    return result
                    
                except Exception as e:
                    self.logger.log_error(f"Error tokenizing batch: {str(e)}")
                    raise
            
            # Process datasets
            train_tokenized = train_dataset.map(
                process_and_tokenize_batch,
                batched=True,
                batch_size=batch_size,
                remove_columns=train_dataset.column_names,
                desc="Tokenizing training set"
            )
            self.logger.log_info(f"Tokenized training dataset: {len(train_tokenized)} examples")
            
            val_tokenized = None
            if val_dataset is not None:
                val_tokenized = val_dataset.map(
                    process_and_tokenize_batch,
                    batched=True,
                    batch_size=batch_size,
                    remove_columns=val_dataset.column_names,
                    desc="Tokenizing validation set"
                )
                self.logger.log_info(f"Tokenized validation dataset: {len(val_tokenized)} examples")
                
            test_tokenized = None
            if test_dataset is not None:
                test_tokenized = test_dataset.map(
                    process_and_tokenize_batch,
                    batched=True,
                    batch_size=batch_size,
                    remove_columns=test_dataset.column_names,
                    desc="Tokenizing test set"
                )
                self.logger.log_info(f"Tokenized test dataset: {len(test_tokenized)} examples")
            
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