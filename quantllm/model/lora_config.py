from peft import LoraConfig
from typing import List, Optional, Dict

class LoraConfigManager:
    def __init__(self):
        """Initialize the LoRA configuration manager"""
        self.default_config = {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
        
    def get_default_config(self) -> LoraConfig:
        """Get the default LoRA configuration"""
        return LoraConfig(**self.default_config)
        
    def create_custom_config(
        self,
        r: int = 8,
        lora_alpha: int = 32,
        target_modules: List[str] = None,
        lora_dropout: float = 0.05,
        bias: str = "none",
        task_type: str = "CAUSAL_LM"
    ) -> LoraConfig:
        """
        Create a custom LoRA configuration.
        
        Args:
            r (int): LoRA attention dimension
            lora_alpha (int): LoRA alpha parameter
            target_modules (List[str]): Target modules for LoRA
            lora_dropout (float): LoRA dropout rate
            bias (str): Bias type
            task_type (str): Task type
            
        Returns:
            LoraConfig: Custom LoRA configuration
        """
        if target_modules is None:
            target_modules = self.default_config["target_modules"]
            
        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type
        ) 