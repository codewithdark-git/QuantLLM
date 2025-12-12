"""
Advanced Training Utilities for QuantLLM v2.0

Provides auto-configuration and optimization for fine-tuning
with minimal user input.
"""

import gc
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Callable
import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Auto-configured training parameters."""
    # Basic
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=list)
    
    # Optimization
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    
    # Memory
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = False
    
    # Output
    output_dir: str = "./output"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "gradient_checkpointing": self.gradient_checkpointing,
            "fp16": self.fp16,
            "bf16": self.bf16,
        }


class AutoBatchSizeFinder:
    """
    Automatically find the maximum batch size that fits in GPU memory.
    
    Uses binary search to efficiently find the optimal batch size.
    
    Example:
        >>> finder = AutoBatchSizeFinder(model, tokenizer)
        >>> max_batch = finder.find_max_batch_size()
        >>> print(f"Use batch size: {max_batch}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_seq_length: int = 512,
        start_batch_size: int = 32,
        min_batch_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.start_batch_size = start_batch_size
        self.min_batch_size = min_batch_size
    
    def find_max_batch_size(
        self,
        training: bool = True,
        safety_margin: float = 0.9,
    ) -> int:
        """
        Find maximum batch size using binary search.
        
        Args:
            training: Whether to test for training (includes gradients)
            safety_margin: Use this fraction of max to allow headroom
            
        Returns:
            Maximum recommended batch size
        """
        if not torch.cuda.is_available():
            return self.min_batch_size
        
        # Clear memory first
        gc.collect()
        torch.cuda.empty_cache()
        
        max_working = self.min_batch_size
        current = self.start_batch_size
        min_test = self.min_batch_size
        max_test = self.start_batch_size * 2
        
        # Binary search
        while min_test <= max_test:
            mid = (min_test + max_test) // 2
            
            if self._test_batch_size(mid, training):
                max_working = mid
                min_test = mid + 1
            else:
                max_test = mid - 1
            
            # Cleanup between tests
            gc.collect()
            torch.cuda.empty_cache()
        
        # Apply safety margin
        return max(self.min_batch_size, int(max_working * safety_margin))
    
    def _test_batch_size(self, batch_size: int, training: bool) -> bool:
        """Test if a batch size fits in memory."""
        try:
            # Create dummy input
            dummy_input = torch.randint(
                0, 
                self.tokenizer.vocab_size,
                (batch_size, self.max_seq_length),
                device=self.model.device if hasattr(self.model, 'device') else 'cuda'
            )
            
            if training:
                self.model.train()
                with torch.cuda.amp.autocast():
                    outputs = self.model(dummy_input, labels=dummy_input)
                    loss = outputs.loss
                    loss.backward()
                
                self.model.zero_grad()
            else:
                self.model.eval()
                with torch.inference_mode():
                    with torch.cuda.amp.autocast():
                        self.model(dummy_input)
            
            del dummy_input
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    
    def get_recommended_config(self, target_effective_batch: int = 32) -> Dict[str, int]:
        """
        Get recommended batch size and gradient accumulation.
        
        Args:
            target_effective_batch: Target effective batch size
            
        Returns:
            Dict with batch_size and gradient_accumulation_steps
        """
        max_batch = self.find_max_batch_size()
        
        if max_batch >= target_effective_batch:
            return {
                "batch_size": target_effective_batch,
                "gradient_accumulation_steps": 1,
            }
        
        grad_accum = (target_effective_batch + max_batch - 1) // max_batch
        
        return {
            "batch_size": max_batch,
            "gradient_accumulation_steps": grad_accum,
        }


class LoRAAutoConfig:
    """
    Automatic LoRA configuration based on model architecture.
    
    Detects the model type and selects appropriate target modules
    and hyperparameters.
    """
    
    # Target modules for different architectures
    TARGET_MODULES = {
        # Llama-style
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "qwen": ["c_attn", "c_proj", "w1", "w2"],
        "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "yi": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "internlm": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Phi-style
        "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        
        # Gemma
        "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Falcon
        "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        
        # GPT-NeoX
        "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        
        # BLOOM
        "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        
        # MPT
        "mpt": ["Wqkv", "out_proj", "up_proj", "down_proj"],
        
        # StarCoder
        "starcoder": ["c_attn", "c_proj", "c_fc"],
        "starcoder2": ["q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "c_proj"],
        
        # ChatGLM
        "chatglm": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        
        # OPT
        "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        
        # Default
        "default": ["q_proj", "v_proj"],
    }
    
    # Recommended LoRA rank by model size
    RANK_BY_SIZE = {
        "tiny": 8,      # < 1B
        "small": 16,    # 1-7B  
        "medium": 32,   # 7-13B
        "large": 64,    # 13-70B
        "huge": 128,    # 70B+
    }
    
    @classmethod
    def get_config(
        cls,
        model: nn.Module,
        task_type: str = "CAUSAL_LM",
    ) -> Dict[str, Any]:
        """
        Get optimal LoRA configuration for a model.
        
        Args:
            model: The model to configure LoRA for
            task_type: Task type (CAUSAL_LM, SEQ_CLS, etc.)
            
        Returns:
            Dict with LoRA configuration
        """
        # Detect model type
        model_type = cls._detect_model_type(model)
        
        # Get target modules
        target_modules = cls.TARGET_MODULES.get(
            model_type, 
            cls.TARGET_MODULES["default"]
        )
        
        # Validate target modules exist in model
        target_modules = cls._validate_modules(model, target_modules)
        
        # Get model size category
        num_params = sum(p.numel() for p in model.parameters())
        size_category = cls._get_size_category(num_params)
        
        # Get recommended rank
        r = cls.RANK_BY_SIZE[size_category]
        
        return {
            "r": r,
            "lora_alpha": r * 2,
            "target_modules": target_modules,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": task_type,
        }
    
    @classmethod
    def _detect_model_type(cls, model: nn.Module) -> str:
        """Detect the model architecture type."""
        if hasattr(model, 'config'):
            model_type = getattr(model.config, 'model_type', '').lower()
            if model_type in cls.TARGET_MODULES:
                return model_type
        
        # Try to infer from module names
        module_names = [name for name, _ in model.named_modules()]
        
        if any('query_key_value' in name for name in module_names):
            if any('bloom' in str(model.__class__).lower() for _ in [1]):
                return "bloom"
            return "falcon"
        
        if any('c_attn' in name for name in module_names):
            return "starcoder"
        
        if any('gate_proj' in name for name in module_names):
            return "llama"
        
        return "default"
    
    @classmethod
    def _validate_modules(cls, model: nn.Module, modules: List[str]) -> List[str]:
        """Validate and filter target modules that exist in the model."""
        model_modules = {name.split('.')[-1] for name, _ in model.named_modules()}
        
        valid_modules = []
        for m in modules:
            if m in model_modules:
                valid_modules.append(m)
            else:
                # Also check if it's a partial match
                for model_m in model_modules:
                    if m in model_m:
                        valid_modules.append(m)
                        break
        
        return valid_modules if valid_modules else ["q_proj", "v_proj"]
    
    @classmethod
    def _get_size_category(cls, num_params: int) -> str:
        """Get model size category."""
        if num_params < 1e9:
            return "tiny"
        elif num_params < 7e9:
            return "small"
        elif num_params < 13e9:
            return "medium"
        elif num_params < 70e9:
            return "large"
        else:
            return "huge"


class TrainingCallbacks:
    """Collection of training callbacks for progress monitoring."""
    
    @staticmethod
    def progress_callback() -> Callable:
        """Create a progress bar callback."""
        def callback(state, logs, **kwargs):
            if "loss" in logs:
                print(f"Step {state.global_step}: loss={logs['loss']:.4f}")
        return callback
    
    @staticmethod
    def memory_monitor_callback() -> Callable:
        """Create a memory monitoring callback."""
        def callback(state, logs, **kwargs):
            if torch.cuda.is_available() and state.global_step % 50 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        return callback
    
    @staticmethod
    def auto_save_callback(save_steps: int = 500) -> Callable:
        """Create an auto-save callback."""
        def callback(state, logs, model=None, output_dir="./checkpoints", **kwargs):
            if model and state.global_step > 0 and state.global_step % save_steps == 0:
                save_path = f"{output_dir}/checkpoint-{state.global_step}"
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")
        return callback


def auto_configure_training(
    model: nn.Module,
    tokenizer: Any,
    data_size: int,
    max_seq_length: int = 512,
    target_effective_batch: int = 32,
) -> TrainingConfig:
    """
    Automatically configure all training parameters.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        data_size: Number of training examples
        max_seq_length: Maximum sequence length
        target_effective_batch: Target effective batch size
        
    Returns:
        TrainingConfig with optimal settings
    """
    config = TrainingConfig()
    
    # Auto batch size
    if torch.cuda.is_available():
        finder = AutoBatchSizeFinder(model, tokenizer, max_seq_length)
        batch_config = finder.get_recommended_config(target_effective_batch)
        config.batch_size = batch_config["batch_size"]
        config.gradient_accumulation_steps = batch_config["gradient_accumulation_steps"]
    else:
        config.batch_size = 1
        config.gradient_accumulation_steps = target_effective_batch
    
    # Auto LoRA config
    lora_config = LoRAAutoConfig.get_config(model)
    config.lora_r = lora_config["r"]
    config.lora_alpha = lora_config["lora_alpha"]
    config.lora_target_modules = lora_config["target_modules"]
    
    # Auto epochs based on data size
    if data_size < 1000:
        config.epochs = 5
    elif data_size < 10000:
        config.epochs = 3
    else:
        config.epochs = 1
    
    # Auto learning rate based on model size
    num_params = sum(p.numel() for p in model.parameters())
    if num_params > 30e9:
        config.learning_rate = 1e-4
    elif num_params > 10e9:
        config.learning_rate = 2e-4
    else:
        config.learning_rate = 3e-4
    
    # Dtype based on hardware
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:  # Ampere+
            config.bf16 = True
        else:
            config.fp16 = True
    
    return config


def load_training_data(
    data: Union[str, List[Dict], Any],
    tokenizer: Any,
    max_length: int = 512,
) -> Any:
    """
    Load and prepare training data from various sources.
    
    Supports:
    - JSON/JSONL files
    - HuggingFace dataset names
    - List of dictionaries
    - Pre-loaded datasets
    
    Args:
        data: Training data source
        tokenizer: Tokenizer for processing
        max_length: Maximum sequence length
        
    Returns:
        Processed dataset ready for training
    """
    from datasets import Dataset, load_dataset
    
    # Load raw data
    if isinstance(data, str):
        if os.path.exists(data):
            # Local file
            if data.endswith('.json') or data.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=data)['train']
            elif data.endswith('.csv'):
                dataset = load_dataset('csv', data_files=data)['train']
            else:
                dataset = load_dataset(data)['train']
        else:
            # HuggingFace dataset
            dataset = load_dataset(data)['train']
    elif isinstance(data, list):
        dataset = Dataset.from_list(data)
    else:
        dataset = data
    
    # Determine text column
    text_column = _find_text_column(dataset)
    
    # Tokenize
    def tokenize_fn(examples):
        texts = examples[text_column]
        result = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized


def _find_text_column(dataset) -> str:
    """Find the text column in a dataset."""
    # Common text column names
    text_columns = ['text', 'content', 'input', 'prompt', 'question']
    
    for col in text_columns:
        if col in dataset.column_names:
            return col
    
    # Try to find instruction/output format
    if 'instruction' in dataset.column_names:
        if 'output' in dataset.column_names:
            # Need to combine - return special marker
            return '__instruction_output__'
        return 'instruction'
    
    # Return first string column
    for col in dataset.column_names:
        sample = dataset[0][col]
        if isinstance(sample, str):
            return col
    
    raise ValueError(f"Could not find text column. Available: {dataset.column_names}")
