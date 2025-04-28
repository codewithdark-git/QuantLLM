"""Low-level API for QuantLLM - provides detailed control over model loading and quantization."""

import torch
from typing import Optional, Dict, Any, Tuple, Union, List
from transformers import PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig
from ..model.model import Model
from ..config.model_config import ModelConfig
from ..quant.quantization_engine import QuantizationEngine
from ..quant.kernels import TritonKernelManager

class LowLevelQuantLLM:
    """Low-level interface providing fine-grained control over model loading and quantization."""
    
    def __init__(self):
        self.quant_engine = QuantizationEngine()
        self.kernel_manager = TritonKernelManager()
    
    def load_model_advanced(
        self,
        model_name: str,
        *,
        quant_config: Optional[BitsAndBytesConfig] = None,
        device_map: Union[str, Dict[str, str]] = "auto",
        max_memory: Optional[Dict[str, str]] = None,
        use_triton_kernels: bool = False,
        optimize_layers: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model with detailed quantization and optimization controls.
        
        Args:
            model_name: Model name or path
            quant_config: Optional custom BitsAndBytes quantization config
            device_map: Device mapping strategy
            max_memory: Maximum memory per device
            use_triton_kernels: Whether to use optimized Triton kernels
            optimize_layers: List of layer names to optimize with Triton
            **kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        config = ModelConfig(
            model_name=model_name,
            device_map=device_map,
            max_memory=max_memory,
            kwargs=kwargs
        )
        
        if quant_config:
            config.quantization_config = quant_config.to_dict()
        
        model_loader = Model(config)
        model, tokenizer = model_loader.get_model(), model_loader.get_tokenizer()
        
        if use_triton_kernels:
            model = self.kernel_manager.optimize_model(
                model, 
                target_modules=optimize_layers
            )
            
        return model, tokenizer
    
    def quantize_model_weights(
        self,
        model: PreTrainedModel,
        bits: int = 4,
        group_size: int = 128,
        compute_dtype: torch.dtype = torch.bfloat16,
        use_double_quant: bool = True
    ) -> PreTrainedModel:
        """
        Apply quantization to an existing model's weights.
        
        Args:
            model: Model to quantize
            bits: Number of bits for quantization
            group_size: Size of quantization groups
            compute_dtype: Compute dtype for operations
            use_double_quant: Whether to use double quantization
            
        Returns:
            Quantized model
        """
        return self.quant_engine.quantize_weights(
            model,
            bits=bits,
            group_size=group_size,
            compute_dtype=compute_dtype,
            use_double_quant=use_double_quant
        )
    
    def replace_layer_with_triton(
        self,
        model: PreTrainedModel,
        layer_name: str,
        kernel_type: str = "auto"
    ) -> PreTrainedModel:
        """
        Replace a specific layer with its optimized Triton version.
        
        Args:
            model: Model to modify
            layer_name: Name of layer to replace
            kernel_type: Type of Triton kernel to use
            
        Returns:
            Model with replaced layer
        """
        return self.kernel_manager.replace_layer(
            model,
            layer_name=layer_name,
            kernel_type=kernel_type
        )
    
    def get_memory_stats(self, model: PreTrainedModel) -> Dict[str, Any]:
        """Get detailed memory statistics for model."""
        return self.quant_engine.get_memory_stats(model)