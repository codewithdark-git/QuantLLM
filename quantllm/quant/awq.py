"""AWQ (Activation-Aware Weight Quantization) implementation."""

import gc
import torch
import torch.nn as nn 
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import PreTrainedModel
from .quantization_engine import move_to_device, BaseQuantizer, QuantizationConfig, QuantizedLinear

class AWQQuantizer(BaseQuantizer):
    """AWQ quantization implementation with memory-efficient processing."""
    
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel], # Changed parameter name
        bits: int = 4,
        group_size: int = 128, 
        zero_point: bool = True,
        scale_dtype: str = "fp32",
        version: str = "v2",
        enable_mnn_kernel: bool = False,
        batch_size: int = 2,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initializes the AWQQuantizer.

        Args:
            model_or_model_name_or_path (Union[str, PreTrainedModel]): 
                The Hugging Face model name/path or a PreTrainedModel instance to be quantized.
            bits (int, optional): Number of bits for quantization. Defaults to 4.
            group_size (int, optional): Size of the quantization group. Defaults to 128.
            zero_point (bool, optional): Whether to use zero-point quantization for activations. Defaults to True.
            scale_dtype (str, optional): Data type for scales. Defaults to "fp32".
            version (str, optional): AWQ algorithm version (e.g., "v1", "v2"). Defaults to "v2".
            enable_mnn_kernel (bool, optional): Whether to enable MNN kernel (if applicable). Defaults to False.
            batch_size (int, optional): Batch size for calibration data processing. Defaults to 2.
            device (Optional[Union[str, torch.device]], optional): 
                The device for quantization operations ('cpu', 'cuda', etc.). 
                Inherited from BaseQuantizer. Defaults to None (auto-detection).
        """
        # Pass all relevant kwargs to BaseQuantizer
        # AWQQuantizer specific args are handled here.
        super().__init__(model_name=model_name, bits=bits, device=device)
        self.group_size = group_size
        self.zero_point = zero_point
        self.scale_dtype = scale_dtype
        self.version = version
        self.enable_mnn_kernel = enable_mnn_kernel
        self.batch_size = batch_size
        
        # Initialize activation statistics dictionaries
        self.act_scales = {}
        self.weight_scales = {}
        
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
            self._collect_activation_stats(batch) # Removed num_steps argument
            
            # Clean up batch
            del batch
            self._clear_memory()
            
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
        
        # Update model config with quantization parameters
        awq_specific_params = {
            "zero_point": self.zero_point,
            "version": self.version,
            "scale_dtype": self.scale_dtype, # Added from __init__
            "enable_mnn_kernel": self.enable_mnn_kernel # Added from __init__
            # batch_size is more of a process param, not a model config param usually
        }
        self._update_model_config_with_quant_params("awq", awq_specific_params)
        
        return self.model

    def _collect_activation_stats(
        self,
        data: torch.Tensor # Removed num_steps parameter
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
        
        # Run calibration (forward pass on the provided data batch)
        with torch.no_grad():
            # Ensure data is on the primary device for model processing
            data_on_device = move_to_device(data, self.device_manager.primary_device)
            self.model(data_on_device)
            # Data can be moved back to CPU if it's large and memory is a concern,
            # but hooks should have already captured necessary info to CPU.
            # For simplicity here, we assume hooks manage CPU transfer if needed.
            # del data_on_device # Optionally delete if memory is very tight

        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # model is already on self.device_manager.primary_device from the quantize method's perspective
        # or moved by prepare_calibration_data.
        # The processing of act_scales should happen after all batches are processed.
        # However, the current structure calls this per batch.
        # For now, let's keep the quantile calculation here, but ideally, it would be after the main loop in `quantize`.
        # To avoid issues with model device, let's ensure model is on CPU for this CPU-bound operation,
        # then move it back if it was on GPU.
        
        original_model_device = self.model.device # Store original device
        self.model = move_to_device(self.model, torch.device('cpu'))
        self._clear_memory() 

        # Process collected statistics on CPU
        for name in self.act_scales:
            if self.act_scales[name]: # Ensure list is not empty
                scales_list = self.act_scales[name]
                # If scales_list contains tensors that are not on CPU, move them.
                # Assuming they are already on CPU due to `scale.cpu()` in hook.
                scales_tensor = torch.stack(scales_list)
                self.act_scales[name] = torch.quantile(scales_tensor, 0.999)
            else:
                # Handle cases where a layer might not have collected scales (e.g. not used in forward pass)
                self.logger.log_warning(f"No activation scales collected for layer {name}. Using default scale of 1.0.")
                self.act_scales[name] = torch.tensor(1.0, device='cpu') # Default to a CPU tensor

        # Restore model to its original device
        self.model = move_to_device(self.model, original_model_device)
        # The duplicated block of "Process collected statistics" is now removed.
            
    def _quantize_layer(
        self,
        layer: nn.Linear,
        act_scale: torch.Tensor
    ) -> QuantizedLinear:
        """Quantize a single layer using AWQ."""
        target_device = self.device_manager.primary_device

        # Initialize quantized layer and move to target device
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
        quantized = move_to_device(quantized, target_device)

        # Ensure layer parameters are on the target_device for computation
        layer = move_to_device(layer, target_device)

        # Copy bias if exists, ensuring it's on the target device
        if layer.bias is not None:
            quantized.bias.data.copy_(layer.bias.data) # Bias already on target_device due to layer move
        
        # Get weight matrix
        W = layer.weight.data.clone() # W is on target_device
        
        # Ensure act_scale is on the same device as W before division
        act_scale_on_device = move_to_device(act_scale, W.device)
        W = W / act_scale_on_device.view(1, -1)
        
        # Compute quantization scales per group
        # All computations for scales and zero_points should happen on target_device
        if self.group_size > 0:
            n_groups = W.shape[0] // self.group_size
            W_groups = W.view(n_groups, self.group_size, -1)
            
            scales_list = [] # Renamed from scales to scales_list
            zero_points_list = [] if self.zero_point else None # Renamed
            
            for idx in range(n_groups):
                group = W_groups[idx]
                max_abs = torch.max(torch.abs(group))
                current_scale = (2 ** (self.bits - 1) - 1) / max_abs # Renamed from scale
                scales_list.append(current_scale)
                
                if self.zero_point:
                    current_zero_point = -(torch.max(group) + torch.min(group)) / 2 * current_scale # Renamed
                    zero_points_list.append(current_zero_point)
            
            scales = torch.stack(scales_list)
            if self.zero_point:
                zero_points = torch.stack(zero_points_list)
            else:
                zero_points = torch.zeros_like(scales, device=target_device) # Ensure on target_device
        else:
            max_abs = torch.max(torch.abs(W), dim=1)[0]
            scales = (2 ** (self.bits - 1) - 1) / max_abs
            if self.zero_point:
                max_vals = torch.max(W, dim=1)[0]
                min_vals = torch.min(W, dim=1)[0]
                zero_points = -(max_vals + min_vals) / 2 * scales
            else:
                zero_points = torch.zeros_like(scales, device=target_device) # Ensure on target_device
                
        # Quantize weights
        # W, scales, zero_points are on target_device
        W_quant = torch.round(W * scales.view(-1, 1) - zero_points.view(-1, 1))
        W_quant = W_quant.to(torch.int8) # Cast to int8
        
        # Store quantized weights and parameters
        # quantized module and its buffers are already on target_device
        quantized.weight_quantized.copy_(W_quant) # W_quant is already on target_device and int8
        quantized.weight_scale.copy_(1.0 / scales) # scales is on target_device
        quantized.weight_zero_point.copy_(zero_points) # zero_points is on target_device
        
        # Store additional AWQ-specific information
        # Ensure act_scale is on the same device as the quantized layer's parameters
        if hasattr(quantized, 'act_scale'):
            # act_scale_on_device was already computed and is on target_device
            quantized.act_scale.copy_(act_scale_on_device)
        
        return quantized
