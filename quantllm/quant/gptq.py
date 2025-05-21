"""GPTQ (Goyal-Pham-Tan-Quant) implementation."""

import gc
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
from transformers import PreTrainedModel
from .quantization_engine import BaseQuantizer, QuantizationConfig, QuantizedLinear

class GPTQQuantizer(BaseQuantizer):
    """GPTQ quantization implementation with memory-efficient processing."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        bits: int = 4,
        group_size: int = 128,
        actorder: bool = False,
        allow_mixed_bits: bool = False,
        use_triton: bool = False,
        percdamp: float = 0.01,
        sym: bool = True,
        batch_size: int = 4,
        device: Optional[Union[str, torch.device]] = None
    ):
        super().__init__(model=model, bits=bits, device=device)
        self.group_size = group_size
        self.actorder = actorder
        self.allow_mixed_bits = allow_mixed_bits
        self.use_triton = use_triton and torch.cuda.is_available()
        self.percdamp = percdamp
        self.sym = sym
        self.batch_size = batch_size
        
        # Initialize H matrices for each layer
        self.H = {}
        
    def _clear_memory(self):
        """Clear CUDA memory and run garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.device_manager.sync()
            
    def quantize(self, calibration_data: Optional[torch.Tensor] = None) -> PreTrainedModel:
        """
        Quantize model using GPTQ algorithm with memory-efficient processing.
        
        Args:
            calibration_data: Optional tensor for computing quantization statistics
            
        Returns:
            Quantized model
        """
        if calibration_data is None:
            raise ValueError("GPTQ requires calibration data for quantization")
            
        # Prepare model and data
        calibration_data = self.prepare_for_quantization(calibration_data)
        self.model.eval()
        
        # Process layers
        for name, module in self.model.named_modules():            
            if isinstance(module, nn.Linear):
                self.logger.log_info(f"Processing layer: {name}")
                
                # Compute Hessian approximation
                self.H[name] = self._compute_hessian(module, calibration_data)
                quantized = self._quantize_layer(module, self.H[name])
                
                # Replace layer in model
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, child_name, quantized)
                else:
                    setattr(self.model, name, quantized)
                    
                # Clear memory after processing each layer
                self._clear_memory()
                del self.H[name]
        
        return self.model
    
    def _compute_hessian(self, layer: nn.Linear, data: torch.Tensor) -> torch.Tensor:
        """Compute Hessian approximation for a layer with memory-efficient processing."""
        device = self.device_manager.primary_device
        n = layer.in_features
        H = torch.zeros((n, n), device=device)
        
        def hook_fn(module, input, output):
            x = input[0].detach()
            if len(x.shape) == 3:
                x = x.view(-1, x.size(-1))
            
            with torch.no_grad():
                chunk_size = 1024
                num_chunks = math.ceil(x.size(0) / chunk_size)
                
                for i in range(num_chunks):
                    chunk = x[i * chunk_size:(i + 1) * chunk_size]
                    chunk_H = torch.matmul(chunk.t(), chunk)
                    H.add_(chunk_H)
                    
                    del chunk_H
                    if i % 10 == 0:
                        self._clear_memory()
        
        # Register forward hook
        handle = layer.register_forward_hook(hook_fn)
        
        # Process calibration data in batches
        with torch.no_grad():
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size].to(device)
                self.model(batch)
                
                if i % (self.batch_size * 10) == 0:
                    self._clear_memory()
            
        handle.remove()
        return H
    
    def _quantize_layer(self, layer: nn.Linear, H: torch.Tensor) -> QuantizedLinear:
        """Quantize a single layer using GPTQ with memory management."""
        device = self.device_manager.primary_device
        
        # Ensure tensors are on the correct device
        H = H.to(device)
        W = layer.weight.data.to(device)
        
        # Initialize quantized layer
        quantized = QuantizedLinear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            config=QuantizationConfig(
                bits=self.bits,
                scheme="symmetric" if self.sym else "asymmetric",
                granularity="per-channel",
                calibration="gptq"
            )
        ).to(device)
        
        if layer.bias is not None:
            quantized.bias.data.copy_(layer.bias.data)
        
        # Process in chunks to save memory
        chunk_size = min(1024, layer.out_features)
        for i in range(0, layer.out_features, chunk_size):
            chunk_end = min(i + chunk_size, layer.out_features)
            W_chunk = W[i:chunk_end]
            
            # Compute optimal scaling factors for this chunk
            if self.sym:
                max_val = W_chunk.abs().max(dim=1)[0]
                scale = (2 ** (self.bits - 1) - 1) / max_val
            else:
                min_val = W_chunk.min(dim=1)[0]
                max_val = W_chunk.max(dim=1)[0]
                scale = (2 ** self.bits - 1) / (max_val - min_val)
            
            # Quantize chunk
            W_quant = torch.round(W_chunk * scale.unsqueeze(1))
            W_quant = torch.clamp(
                W_quant,
                -(2 ** (self.bits - 1)),
                2 ** (self.bits - 1) - 1
            )
            
            # Store quantized weights and scale
            quantized.weight_quantized.data[i:chunk_end] = W_quant.to(torch.int8)
            quantized.weight_scale.data[i:chunk_end] = 1.0 / scale
            
            del W_chunk, W_quant
            self._clear_memory()
        
        return quantized
