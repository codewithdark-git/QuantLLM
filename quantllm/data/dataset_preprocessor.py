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
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added [PAD] token to tokenizer")

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
                texts = examples[text_column]
                if not isinstance(texts, list):
                    texts = [texts]
                
                # Preprocess texts
                texts = [self.preprocess_text(text) for text in texts]
                
                try:
                    tokenized = self.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    
                    result = {
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"]
                    }
                    
                    if label_column and label_column in examples:
                        result["labels"] = examples[label_column]
                    
                    print(f"Tokenized batch of {len(texts)} texts")  # User feedback
                    return result
                    
                except Exception as e:
                    print(f"Error tokenizing batch: {str(e)}")  # User feedback
                    raise
            
            # Process datasets
            train_tokenized = train_dataset.map(
                process_and_tokenize_batch,
                batched=True,
                batch_size=batch_size,
                remove_columns=train_dataset.column_names
            )
            print(f"Tokenized training dataset: {len(train_tokenized)} examples")  # User feedback
            
            val_tokenized = None
            if val_dataset is not None:
                val_tokenized = val_dataset.map(
                    process_and_tokenize_batch,
                    batched=True,
                    batch_size=batch_size,
                    remove_columns=val_dataset.column_names
                )
                print(f"Tokenized validation dataset: {len(val_tokenized)} examples")  # User feedback
                
            test_tokenized = None
            if test_dataset is not None:
                test_tokenized = test_dataset.map(
                    process_and_tokenize_batch,
                    batched=True,
                    batch_size=batch_size,
                    remove_columns=test_dataset.column_names
                )
                print(f"Tokenized test dataset: {len(test_tokenized)} examples")  # User feedback
                
            return train_tokenized, val_tokenized, test_tokenized
            
        except Exception as e:
            print(f"Error in tokenization: {str(e)}")  # User feedback
            raise