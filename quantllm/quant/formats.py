"""Support for advanced quantization formats."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import torch

@dataclass
class AdvancedQuantConfig:
    """Configuration for advanced quantization formats."""
    format: Literal["gguf", "gptq", "awq"] = "gguf"
    bits: Literal[2, 4, 8] = 4
    group_size: int = 128
    desc_act: bool = True  # Group-wise quantization
    
    # GGUF specific
    gguf_mode: Optional[str] = None  # e.g. "llama", "falcon"
    
    # GPTQ specific
    block_size: int = 128
    percdamp: float = 0.01
    actorder: bool = False
    
    # AWQ specific 
    awq_alpha: float = 0.5
    awq_bits_groups: int = 128
    
    def __post_init__(self):
        """Validate configuration."""
        if self.bits not in [2, 4, 8]:
            raise ValueError(f"Bits must be 2, 4, or 8, got {self.bits}")
            
        if self.format == "gguf":
            if not self.gguf_mode:
                raise ValueError("GGUF mode must be specified")
        elif self.format == "gptq":
            if self.block_size <= 0:
                raise ValueError("Block size must be positive")
        elif self.format == "awq":
            if not 0 <= self.awq_alpha <= 1:
                raise ValueError("AWQ alpha must be between 0 and 1")

class GGUFQuantizer:
    """GGUF format quantizer using CTransformers."""
    
    def __init__(self, config: AdvancedQuantConfig):
        try:
            from ctransformers import AutoModelForCausalLM
            self.ctransformers = AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "CTransformers not found, please install with: pip install ctransformers"
            )
        self.config = config
    
    def quantize(self, model_path: str, output_path: str):
        """Quantize model to GGUF format."""
        model = self.ctransformers.from_pretrained(
            model_path,
            model_type=self.config.gguf_mode,
            lib="avx2",  # CPU optimization
            gpu_layers=0  # CPU only for quantization
        )
        
        model.save_pretrained(
            output_path,
            bits=self.config.bits,
            group_size=self.config.group_size
        )
        return output_path

class GPTQQuantizer:
    """GPTQ quantizer implementation."""
    
    def __init__(self, config: AdvancedQuantConfig):
        try:
            import auto_gptq
        except ImportError:
            raise ImportError(
                "AutoGPTQ not found, please install with: pip install auto-gptq"
            )
        self.config = config
    
    def quantize(
        self,
        model,
        dataloader,
        output_path: str
    ):
        """Quantize model using GPTQ algorithm."""
        from auto_gptq import AutoGPTQForCausalLM
        
        quantized = AutoGPTQForCausalLM.from_pretrained(
            model,
            bits=self.config.bits,
            group_size=self.config.group_size,
            desc_act=self.config.desc_act,
            block_size=self.config.block_size,
            percdamp=self.config.percdamp,
            actorder=self.config.actorder
        )
        
        # Calibrate and quantize
        quantized.quantize(dataloader)
        
        # Save quantized model
        quantized.save_pretrained(output_path)
        return output_path

class AWQQuantizer:
    """AWQ (Activation-aware Weight Quantization) implementation."""
    
    def __init__(self, config: AdvancedQuantConfig):
        try:
            import awq
        except ImportError:
            raise ImportError(
                "AWQ not found, please install with: pip install awq"
            )
        self.config = config
    
    def quantize(
        self,
        model,
        dataloader,
        output_path: str
    ):
        """Quantize model using AWQ algorithm."""
        from awq import AutoAWQForCausalLM
        
        quantized = AutoAWQForCausalLM.from_pretrained(
            model,
            bits=self.config.bits,
            group_size=self.config.group_size,
            alpha=self.config.awq_alpha,
            bits_groups=self.config.awq_bits_groups
        )
        
        # Calibrate and quantize
        quantized.quantize(dataloader)
        
        # Save quantized model
        quantized.save_pretrained(output_path)
        return output_path
