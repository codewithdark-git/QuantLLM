import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Dict, Any
from .lora_config import LoraConfigManager
from ..config.model_config import ModelConfig

class Model:
    def __init__(
        self,
        config: ModelConfig
    ):
        """
        Initialize the Model.
        
        Args:
            config (ModelConfig): Configuration object for the model

        """
        self.model_name = config.model_name
        self.quantization = '4bit' if config.load_in_4bit else '8bit'
        self.use_lora = config.use_lora
        self.device_map = config.device_map
        self.kwargs = config.kwargs
        
        self._setup_quantization()
        self._load_model()
        
    def _setup_quantization(self):
        """Setup quantization configuration"""
        if self.quantization == "4bit":
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        elif self.quantization == "8bit":
            self.quant_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            raise ValueError(f"Unsupported quantization mode: {self.quantization}")
            
    def _load_model(self):
        """Load the model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.quant_config,
            device_map=self.device_map,
            **self.kwargs
        )
        
        if self.use_lora:
            self._setup_lora()
            
    def _setup_lora(self):
        """Setup LoRA configuration"""
        lora_manager = LoraConfigManager()
        lora_config = lora_manager.get_default_config()
        
        from peft import prepare_model_for_kbit_training, get_peft_model
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
    def get_model(self):
        """Get the loaded model"""
        return self.model
        
    def get_tokenizer(self):
        """Get the loaded tokenizer"""
        return self.tokenizer 