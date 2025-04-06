import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Dict, Any
from .lora_config import LoraConfigManager

class ModelLoader:
    def __init__(
        self,
        model_name: str,
        quantization: str = "4bit",
        use_lora: bool = True,
        device_map: str = "auto",
        **kwargs
    ):
        """
        Initialize the model loader.
        
        Args:
            model_name (str): Name or path of the model
            quantization (str): Quantization mode ("4bit" or "8bit")
            use_lora (bool): Whether to use LoRA
            device_map (str): Device mapping strategy
            **kwargs: Additional model configuration
        """
        self.model_name = model_name
        self.quantization = quantization
        self.use_lora = use_lora
        self.device_map = device_map
        self.kwargs = kwargs
        
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