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
                act_scale_list_or_tensor = self.act_scales.get(name)

                if act_scale_list_or_tensor is not None:
                    if isinstance(act_scale_list_or_tensor, list):
                        if all(isinstance(t, torch.Tensor) for t in act_scale_list_or_tensor):
                            # Average the list of tensors
                            act_scale = torch.stack(act_scale_list_or_tensor).mean(dim=0)
                        else:
                            # Handle unexpected content in the list
                            self.logger.log_error(f"Activation scales for {name} contain non-tensor elements. Quantization may be incorrect.")
                            # Fallback: attempt to use the list directly if _quantize_layer can handle it, or create a default
                            # For safety, creating a default scale here.
                            act_scale = torch.ones(module.in_features, device=self.device_manager.primary_device)
                    elif isinstance(act_scale_list_or_tensor, torch.Tensor):
                        # If it's already a tensor (e.g., if averaging was done elsewhere or only one batch)
                        act_scale = act_scale_list_or_tensor
                    else:
                        self.logger.log_error(f"Unexpected type for activation scales of {name}: {type(act_scale_list_or_tensor)}. Using default.")
                        act_scale = torch.ones(module.in_features, device=self.device_manager.primary_device)
                else:
                    self.logger.log_warning(f"No activation scales found for {name}. Using default scale of 1.0.")
                    # module.in_features should correspond to the expected dimension of the scale
                    act_scale = torch.ones(module.in_features, device=self.device_manager.primary_device)
                
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

    def _collect_activation_stats(self, data: torch.Tensor):
      """Collect activation statistics for each layer."""
      # Store temporary scales for this batch
      batch_scales = {}
      
      # Register hooks for all linear layers
      handles = []
      for name, module in self.model.named_modules():
          if isinstance(module, nn.Linear):
              def hook_fn(name):
                  def fn(module, input, output):
                      # Initialize the list for this layer if not exists
                      if name not in batch_scales:
                          batch_scales[name] = []
                      
                      x = input[0].detach()
                      # Handle both 2D and 3D inputs
                      if len(x.shape) == 3:
                          # For 3D input (batch_size, seq_len, hidden_size)
                          # Compute scales per hidden channel: (hidden_size,)
                          scale = torch.amax(torch.abs(x), dim=[0, 1])
                      else:
                          # For 2D input (batch_size, hidden_size)
                          # Compute scales per hidden channel: (hidden_size,)
                          scale = torch.amax(torch.abs(x), dim=0)
                      # Store scale tensor (moved to CPU) in our temporary dictionary
                      batch_scales[name].append(scale.cpu())
                  return fn
              
              handles.append(
                  module.register_forward_hook(hook_fn(name))
              )
      
      # Run calibration (forward pass on the provided data batch)
      with torch.no_grad():
          data_on_device = move_to_device(data, self.device_manager.primary_device)
          self.model(data_on_device)
          del data_on_device # Free memory after forward pass
      
      # Remove hooks
      for handle in handles:
          handle.remove()
      
      # Process the collected scales
      for name in batch_scales:
          if batch_scales[name]:  # If we collected any scales for this layer
              # If this is the first batch for this layer
              if name not in self.act_scales:
                  self.act_scales[name] = []
              # Extend the list of scale tensors for this layer
              # batch_scales[name] already contains CPU tensors
              self.act_scales[name].extend(batch_scales[name])
      
      # Clean up
      del batch_scales
      self._clear_memory()
            
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
        quantized = quantized.to(target_device)

        # Ensure layer parameters are on the target_device for computation
        layer = layer.to(target_device)

        # Copy bias if exists, ensuring it's on the target device
        if layer.bias is not None:
            quantized.bias.data.copy_(layer.bias.data) # Bias already on target_device due to layer move
        
        # Get weight matrix
        W = layer.weight.data.clone() # W is on target_device
        
        # Ensure act_scale is on the same device as W before division
        act_scale_on_device = move_to_device(act_scale, W.device)
        
        try:
            W = W / act_scale_on_device.view(1, -1)
        except RuntimeError as e:
            error_message = (
                f"Failed to scale weights with activation scales in _quantize_layer.\n"
                f"  Weight (W) shape: {W.shape}\n"
                f"  Activation scale (act_scale_on_device) shape: {act_scale_on_device.shape}\n"
                f"  Original error: {str(e)}"
            )
            self.logger.log_error(error_message)
            raise RuntimeError(error_message) from e
        
        # Compute quantization scales per group
        # All computations for scales and zero_points should happen on target_device
        if self.group_size > 0:
            if W.shape[0] % self.group_size != 0:
                error_message = (
                    f"Weight dimension {W.shape[0]} is not divisible by group_size {self.group_size} "
                    f"in _quantize_layer for layer being processed."
                )
                self.logger.log_error(error_message)
                raise ValueError(error_message) # ValueError is more appropriate here

            n_groups = W.shape[0] // self.group_size
            try:
                W_groups = W.view(n_groups, self.group_size, -1)
            except RuntimeError as e:
                error_message = (
                    f"Failed to create view for grouped weights in _quantize_layer.\n"
                    f"  Weight (W) shape: {W.shape}\n"
                    f"  Calculated n_groups: {n_groups}\n"
                    f"  Group size: {self.group_size}\n"
                    f"  Original error: {str(e)}"
                )
                self.logger.log_error(error_message)
                raise RuntimeError(error_message) from e
            
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
        del W # Free memory for W as it's no longer needed
        
        # Store quantized weights and parameters
        # quantized module and its buffers are already on target_device
        quantized.weight_quantized.copy_(W_quant) # W_quant is already on target_device and int8
        quantized.weight_scale.copy_(1.0 / scales) # scales is on target_device
        quantized.weight_zero_point.copy_(zero_points) # zero_points is on target_device
        del scales, zero_points # Free memory for scales and zero_points
        
        # Store additional AWQ-specific information
        # Ensure act_scale is on the same device as the quantized layer's parameters
        if hasattr(quantized, 'act_scale'):
            # act_scale_on_device was already computed and is on target_device
            quantized.act_scale.copy_(act_scale_on_device)
        
        return quantized
