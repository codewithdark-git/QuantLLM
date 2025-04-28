"""High-level API for QuantLLM - provides simple, user-friendly interfaces."""

import torch
from typing import Optional, Dict, Any, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..model.model import Model
from ..model.lora_config import LoraConfigManager
from ..config.model_config import ModelConfig

class QuantLLM:
    """Main interface for QuantLLM, providing simplified model loading and training."""
    
    @staticmethod
    def from_pretrained(
        model_name: str,
        *,
        quant_bits: int = 4,
        bnb_4bit_compute_dtype: str = "bfloat16",
        max_seq_len: Optional[int] = None,
        device_map: Union[str, Dict[str, str]] = "auto",
        max_memory: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pre-trained model with optional quantization.
        
        Args:
            model_name: Name or path of the model to load
            quant_bits: Number of bits for quantization (4 or 8)
            bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
            max_seq_len: Maximum sequence length
            device_map: Device mapping strategy or explicit mapping
            max_memory: Maximum memory allocation per device
            **kwargs: Additional arguments passed to from_pretrained
            
        Returns:
            Tuple of (model, tokenizer)
        """
        config = ModelConfig(
            model_name=model_name,
            load_in_4bit=(quant_bits == 4),
            load_in_8bit=(quant_bits == 8),
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            device_map=device_map,
            max_memory=max_memory,
            kwargs=kwargs
        )
        
        model_loader = Model(config)
        return model_loader.get_model(), model_loader.get_tokenizer()
    
    @staticmethod
    def get_adapter_model(
        base_model: PreTrainedModel,
        r: int = 16,
        target_modules: Optional[list] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        bias: str = "none"
    ) -> PreTrainedModel:
        """
        Attach LoRA adapters to a model for efficient fine-tuning.
        
        Args:
            base_model: Base model to attach adapters to
            r: LoRA attention dimension
            target_modules: List of module names to apply LoRA to
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            bias: Bias type ("none", "all", or "lora_only")
            
        Returns:
            Model with LoRA adapters attached
        """
        from peft import prepare_model_for_kbit_training, get_peft_model
        
        lora_config = LoraConfigManager().create_custom_config(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias
        )
        
        model = prepare_model_for_kbit_training(base_model)
        return get_peft_model(model, lora_config)