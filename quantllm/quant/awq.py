"""AWQ (Activation-Aware Weight Quantization) implementation."""

import gc
import torch
import torch.nn as nn 
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import PreTrainedModel
from .quantization_engine import BaseQuantizer, QuantizationConfig, QuantizedLinear

class AWQQuantizer(BaseQuantizer):
    """AWQ quantization implementation with memory-efficient processing."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        bits: int = 4,
        group_size: int = 128, 
        zero_point: bool = True,
        scale_dtype: str = "fp32",
        version: str = "v2",
        enable_mnn_kernel: bool = False,
        batch_size: int = 2,
        device: Optional[Union[str, torch.device]] = None
    ):
        super().__init__(model=model, bits=bits, device=device)
        self.group_size = group_size
        self.zero_point = zero_point
        self.scale_dtype = scale_dtype
        self.version = version
        self.enable_mnn_kernel = enable_mnn_kernel
        self.batch_size = batch_size
        
        # Initialize activation statistics dictionaries
        self.act_scales = {}
        self.weight_scales = {}
        
    def _clear_memory(self):
        """Clear GPU memory and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def quantize(
        self,
        calibration_data: Optional[torch.Tensor] = None,
        calibration_steps: int = 100
    ) -> PreTrainedModel:
        """Memory-efficient quantization using AWQ algorithm."""
        if calibration_data is None:
            raise ValueError("AWQ requires calibration data for quantization")
            
        # Prepare calibration data
        calibration_data = self.prepare_calibration_data(calibration_data)
        self.model.eval()
        
        # Process calibration data in batches
        total_steps = min(calibration_steps, len(calibration_data))
        for step in range(0, total_steps, self.batch_size):
            # Clear memory before processing batch
            self._clear_memory()
            
            # Get batch
            end_idx = min(step + self.batch_size, total_steps)
            batch = calibration_data[step:end_idx]
            
            # Collect statistics for this batch
            self._collect_activation_stats(batch)
            
            # Clean up batch
            del batch
            self._clear_memory()
            
        # Process collected statistics
        self._process_activation_stats()
        
        # Quantize the model layer by layer
        for name, module in self.model.named_modules():            
            if isinstance(module, nn.Linear):
                self.logger.log_info(f"Processing layer: {name}")
                
                # Get activation scale for this layer
                act_scale = self.act_scales.get(name)
                quantized = self._quantize_layer(module, act_scale)
                
                # Replace layer in model
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, child_name, quantized)
                else:
                    setattr(self.model, name, quantized)
                
                self._clear_memory()
        
        return self.model
    def _collect_activation_stats(
        self,
        data: torch.Tensor,
        num_steps: int
    ):
        """Collect activation statistics for each layer."""
        
        # Register hooks for all linear layers
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                def hook_fn(name):
                    def fn(module, input, output):
                        if name not in self.act_scales:
                            self.act_scales[name] = []
                        x = input[0].detach()
                        # Handle both 2D and 3D inputs
                        if len(x.shape) == 3:
                            # For 3D input (batch_size, seq_len, hidden_size)
                            scale = torch.max(torch.abs(x.view(-1, x.size(-1))))
                        else:
                            scale = torch.max(torch.abs(x))
                        self.act_scales[name].append(scale.cpu())  # Move to CPU to save memory
                    return fn
                    
                handles.append(
                    module.register_forward_hook(hook_fn(name))
                )
        
        # Run calibration in smaller batches
        with torch.no_grad():
            batch_size = 2  # Small batch size to prevent OOM
            for step in range(num_steps):
                # Clear CUDA cache periodically
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Process a small batch
                start_idx = (step * batch_size) % len(data)
                end_idx = min(start_idx + batch_size, len(data))
                batch = data[start_idx:end_idx]
                
                # Move batch to appropriate device
                device = next(self.model.parameters()).device
                batch = batch.to(device)
                
                self.model(batch)
                
                # Move batch back to CPU to free GPU memory
                batch = batch.cpu()
                
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # Move model to CPU temporarily to free GPU memory
        self.model = self.model.cpu()
        torch.cuda.empty_cache()
        
        # Process collected statistics on CPU
        for name in self.act_scales:
            scales = torch.stack(self.act_scales[name])
            # Use 99.9th percentile for more robust statistics
            self.act_scales[name] = torch.quantile(scales, 0.999)
            
        # Move model back to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
            
        # Process collected statistics
        for name in self.act_scales:
            scales = torch.stack(self.act_scales[name])
            # Use 99.9th percentile for more robust statistics
            self.act_scales[name] = torch.quantile(scales, 0.999)
            
    def _quantize_layer(
        self,
        layer: nn.Linear,
        act_scale: torch.Tensor
    ) -> QuantizedLinear:
        """Quantize a single layer using AWQ."""
        device = next(layer.parameters()).device
        
        # Initialize quantized layer
        quantized = QuantizedLinear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            config=QuantizationConfig(
                bits=self.bits,
                scheme="symmetric",
                granularity="per-channel" if self.group_size > 0 else "per-tensor",
                calibration="minmax",
                channel_wise=True,
                dtype=f"int{self.bits}",
                format="awq"
            )
        )
        
        # Copy bias if exists
        if layer.bias is not None:
            quantized.bias.data.copy_(layer.bias.data)
        
        # Get weight matrix
        W = layer.weight.data.clone()
        
        # Scale weights by activation scale
        W = W / act_scale.view(1, -1)
        
        # Compute quantization scales per group
        if self.group_size > 0:
            n_groups = W.shape[0] // self.group_size
            W_groups = W.view(n_groups, self.group_size, -1)
            
            scales = []
            zero_points = [] if self.zero_point else None
            
            for idx in range(n_groups):
                group = W_groups[idx]
                max_abs = torch.max(torch.abs(group))
                scale = (2 ** (self.bits - 1) - 1) / max_abs
                scales.append(scale)
                
                if self.zero_point:
                    zero_point = -(torch.max(group) + torch.min(group)) / 2 * scale
                    zero_points.append(zero_point)
            
            scales = torch.stack(scales)
            if self.zero_point:
                zero_points = torch.stack(zero_points)
            else:
                zero_points = torch.zeros_like(scales)
        else:
            max_abs = torch.max(torch.abs(W), dim=1)[0]
            scales = (2 ** (self.bits - 1) - 1) / max_abs
            if self.zero_point:
                max_vals = torch.max(W, dim=1)[0]
                min_vals = torch.min(W, dim=1)[0]
                zero_points = -(max_vals + min_vals) / 2 * scales
            else:
                zero_points = torch.zeros_like(scales)
                
        # Quantize weights
        W_quant = torch.round(W * scales.view(-1, 1) - zero_points.view(-1, 1))
        
        # Store quantized weights and parameters
        quantized.weight_quantized.copy_(W_quant.to(torch.int8))
        quantized.weight_scale.copy_(1.0 / scales)
        quantized.weight_zero_point.copy_(zero_points)
        
        # Store additional AWQ-specific information
        if hasattr(quantized, 'act_scale'):
            quantized.act_scale.copy_(act_scale)
        
        return quantized
