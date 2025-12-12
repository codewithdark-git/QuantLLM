"""
GPTQ Post-Quantization Fine-tuning Example

This example demonstrates how to use the GPTQ fine-tuning capabilities
with QLoRA integration for parameter-efficient fine-tuning.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from quantllm.backend.gptq_finetuning import (
        GPTQFineTuner,
        QLoRAConfig,
        GPTQFineTuningEvaluator,
        create_qlora_config
    )
    from quantllm.models.gptq_model import GPTQModel
    from quantllm.config.quantization_config import GPTQConfig
    from quantllm.models.base import ModelMetadata
    QUANTLLM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"QuantLLM not fully available: {e}")
    QUANTLLM_AVAILABLE = False

try:
    from transformers import PreTrainedTokenizer, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available")
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    logger.warning("PEFT not available")
    PEFT_AVAILABLE = False


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For language modeling
        }


class MockTokenizer:
    """Mock tokenizer for demonstration when transformers is not available."""
    
    def __call__(self, text, **kwargs):
        # Simple mock tokenization
        tokens = text.split()[:kwargs.get('max_length', 512)]
        input_ids = torch.randint(1, 1000, (len(tokens),))
        attention_mask = torch.ones(len(tokens))
        
        if kwargs.get('padding') == 'max_length':
            max_len = kwargs.get('max_length', 512)
            if len(input_ids) < max_len:
                pad_len = max_len - len(input_ids)
                input_ids = torch.cat([input_ids, torch.zeros(pad_len)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)])
        
        return {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.long()
        }
    
    def save_pretrained(self, path):
        """Mock save method."""
        logger.info(f"Mock tokenizer saved to {path}")


class MockModel(nn.Module):
    """Mock model for demonstration."""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            for _ in range(6)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Add common attention projection layers for LoRA targeting
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Mock config with get method
        class MockConfig:
            def __init__(self):
                self.name_or_path = 'mock-gpt-model'
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                self.model_type = 'gpt'
                
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        self.config = MockConfig()
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Simple forward pass
        x = self.embedding(input_ids)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Return object with attributes
        outputs = type('Outputs', (), {})()
        outputs.logits = logits
        outputs.loss = loss
        
        return outputs
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        logger.info("Gradient checkpointing enabled")
    
    def save_pretrained(self, path, **kwargs):
        """Save model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        logger.info(f"Model saved to {path}")


def create_mock_gptq_model() -> 'GPTQModel':
    """Create a mock GPTQ model for demonstration."""
    if not QUANTLLM_AVAILABLE:
        logger.error("QuantLLM not available - cannot create GPTQ model")
        return None
    
    # Create base model
    base_model = MockModel()
    
    # Create GPTQ config
    gptq_config = GPTQConfig(
        method="gptq",
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False
    )
    
    # Create metadata
    metadata = ModelMetadata(
        original_model_name="mock-gpt-model",
        quantization_method="GPTQ",
        bits=4,
        group_size=128,
        compression_ratio=4.0,
        model_size_mb=250.0,
        quantization_time_seconds=120.0,
        quality_metrics={"perplexity": 15.2},
        hardware_info={"gpu_memory_gb": 8},
        timestamp="2024-01-01T00:00:00"
    )
    
    # Create quantized state dict (mock)
    quantized_state_dict = {
        f"layer_{i}.weight": torch.randn(768, 768) 
        for i in range(6)
    }
    
    # Create GPTQ model
    gptq_model = GPTQModel(
        model=base_model,
        config=gptq_config,
        quantized_state_dict=quantized_state_dict,
        metadata=metadata
    )
    
    return gptq_model


def demonstrate_qlora_config():
    """Demonstrate QLoRA configuration creation."""
    logger.info("=== QLoRA Configuration Demo ===")
    
    # Create basic configuration
    basic_config = QLoRAConfig()
    logger.info(f"Basic config: r={basic_config.r}, alpha={basic_config.lora_alpha}")
    
    # Create custom configuration
    custom_config = QLoRAConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        learning_rate=2e-4,
        num_epochs=2,
        batch_size=4
    )
    logger.info(f"Custom config: r={custom_config.r}, modules={custom_config.target_modules}")
    
    # Use helper function
    helper_config = create_qlora_config(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        num_epochs=3
    )
    logger.info(f"Helper config: r={helper_config.r}, epochs={helper_config.num_epochs}")
    
    return custom_config


def demonstrate_fine_tuning():
    """Demonstrate the fine-tuning process."""
    logger.info("=== Fine-tuning Demo ===")
    
    if not QUANTLLM_AVAILABLE:
        logger.error("QuantLLM not available - skipping fine-tuning demo")
        return
    
    # Create mock GPTQ model
    gptq_model = create_mock_gptq_model()
    if gptq_model is None:
        return
    
    # Create QLoRA configuration
    qlora_config = QLoRAConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        learning_rate=2e-4,
        num_epochs=1,  # Short for demo
        batch_size=2
    )
    
    # Create fine-tuner
    try:
        fine_tuner = GPTQFineTuner(gptq_model, qlora_config)
        logger.info("Fine-tuner created successfully")
        
        # Prepare model for fine-tuning (this would normally require PEFT)
        if PEFT_AVAILABLE:
            peft_model = fine_tuner.prepare_model_for_finetuning()
            logger.info("Model prepared for fine-tuning")
        else:
            logger.warning("PEFT not available - cannot prepare model")
            
    except Exception as e:
        logger.error(f"Error creating fine-tuner: {e}")


def demonstrate_evaluation():
    """Demonstrate evaluation functionality."""
    logger.info("=== Evaluation Demo ===")
    
    if not QUANTLLM_AVAILABLE:
        logger.error("QuantLLM not available - skipping evaluation demo")
        return
    
    # Create mock model for evaluation
    model = MockModel()
    
    # Create evaluator
    evaluator = GPTQFineTuningEvaluator(model)
    
    # Create mock dataset
    tokenizer = MockTokenizer()
    test_texts = [
        "This is a test sentence for evaluation.",
        "Another sentence to test the model performance.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    # Evaluate perplexity
    try:
        perplexity = evaluator.evaluate_perplexity(test_texts, tokenizer)
        logger.info(f"Model perplexity: {perplexity:.2f}")
    except Exception as e:
        logger.error(f"Error evaluating perplexity: {e}")
    
    # Compare with base model
    try:
        base_model = MockModel()
        comparison = evaluator.compare_with_base_model(base_model, test_texts, tokenizer)
        logger.info(f"Comparison results: {comparison}")
    except Exception as e:
        logger.error(f"Error comparing models: {e}")


def demonstrate_adapter_operations():
    """Demonstrate adapter saving and merging."""
    logger.info("=== Adapter Operations Demo ===")
    
    if not QUANTLLM_AVAILABLE or not PEFT_AVAILABLE:
        logger.error("Required libraries not available - skipping adapter demo")
        return
    
    # This would normally involve actual fine-tuning
    logger.info("Adapter operations require actual fine-tuning with PEFT")
    logger.info("Key operations:")
    logger.info("1. Save adapter: fine_tuner._save_adapter(output_dir)")
    logger.info("2. Merge adapter: fine_tuner.merge_and_save(output_dir, tokenizer)")
    logger.info("3. Load merged model for inference")


def main():
    """Main demonstration function."""
    logger.info("GPTQ Post-Quantization Fine-tuning Example")
    logger.info("=" * 50)
    
    # Check availability
    logger.info(f"QuantLLM available: {QUANTLLM_AVAILABLE}")
    logger.info(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    logger.info(f"PEFT available: {PEFT_AVAILABLE}")
    logger.info("")
    
    # Demonstrate different aspects
    try:
        # Configuration
        config = demonstrate_qlora_config()
        logger.info("")
        
        # Fine-tuning
        demonstrate_fine_tuning()
        logger.info("")
        
        # Evaluation
        demonstrate_evaluation()
        logger.info("")
        
        # Adapter operations
        demonstrate_adapter_operations()
        logger.info("")
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()