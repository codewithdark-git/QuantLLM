"""GGUF (GGML Universal Format) quantization implementation."""

import gc
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import PreTrainedModel
from .quantization_engine import move_to_device, BaseQuantizer, QuantizationConfig, QuantizedLinear

try:
    import ctransformers
    CT_AVAILABLE = True
except ImportError:
    CT_AVAILABLE = False

class GGUFQuantizer(BaseQuantizer):
    """GGUF quantization implementation with CTransformers integration."""
    
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel], # Changed parameter name
        bits: int = 4,
        group_size: int = 32,
        desc_act: bool = False,
        desc_ten: bool = False,
        use_packed: bool = True,
        legacy_format: bool = False,
        batch_size: int = 4,
        device: Optional[Union[str, torch.device]] = None,
        cpu_offload: bool = False
    ):
        """
        Initializes the GGUFQuantizer.

        Args:
            model_or_model_name_or_path (Union[str, PreTrainedModel]): 
                The Hugging Face model name/path or a PreTrainedModel instance to be quantized.
            bits (int, optional): Number of bits for quantization. Defaults to 4.
            group_size (int, optional): Size of the quantization group. Defaults to 32.
            desc_act (bool, optional): Whether to describe activations in GGUF metadata. Defaults to False.
            desc_ten (bool, optional): Whether to describe tensors in GGUF metadata. Defaults to False.
            use_packed (bool, optional): Whether to use packed quantization types (e.g., Q4_K_M). Defaults to True.
            legacy_format (bool, optional): Whether to use a legacy GGUF format version. Defaults to False.
            batch_size (int, optional): Batch size for processing calibration data. Defaults to 4.
            device (Optional[Union[str, torch.device]], optional): 
                The device for quantization operations ('cpu', 'cuda', etc.). 
                Inherited from BaseQuantizer. Defaults to None (auto-detection).
            cpu_offload (bool, optional): 
                If True, quantized layers and some computations are forced to CPU, 
                reducing GPU memory usage. Defaults to False.
        """
        if not CT_AVAILABLE:
            raise ImportError("CTransformers is required for GGUF quantization. Install with: pip install ctransformers")
            
        super().__init__(model_name=model_name, bits=bits, device=device)
        self.group_size = group_size
        self.desc_act = desc_act
        self.desc_ten = desc_ten
        self.use_packed = use_packed
        self.legacy_format = legacy_format
        self.batch_size = batch_size
        self.cpu_offload = cpu_offload
    def quantize(
        self,
        calibration_data: Optional[torch.Tensor] = None
    ) -> PreTrainedModel:
        """Quantize model using GGUF format with memory-efficient processing."""
        # Prepare model and calibration data
        if calibration_data is not None:
            calibration_data = self.prepare_calibration_data(calibration_data)
        self.model.eval()
        
        # Collect statistics if provided
        stats = {}
        if calibration_data is not None:
            stats = self._collect_stats(calibration_data)
        
        # Convert linear layers to quantized versions
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.logger.log_info(f"Processing layer: {name}")
                
                # Create quantized layer
                layer_stats = stats.get(name, None)
                quantized = self._quantize_layer(module, layer_stats)
                
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
        gguf_specific_params = {
            "use_packed": self.use_packed,
            "cpu_offload": self.cpu_offload,
            "desc_act": self.desc_act,
            "desc_ten": self.desc_ten,
            "legacy_format": self.legacy_format
            # group_size is handled by BaseQuantizer if present as self.group_size
        }
        self._update_model_config_with_quant_params("gguf", gguf_specific_params)
        
        return self.model

    def _collect_stats(self, data: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collect statistics for quantization with memory-efficient batch processing."""
        stats = {}
        
        def hook_fn(name):
            def fn(module, input, output):
                if name not in stats:
                    stat_device = torch.device('cpu') if self.cpu_offload else self.device_manager.primary_device
                    output_sample = output[0] # output_sample is on self.device_manager.primary_device
                    stats[name] = {
                        "min_val": torch.tensor(float('inf'), device=stat_device),
                        "max_val": torch.tensor(float('-inf'), device=stat_device),
                        "sum": torch.zeros(output_sample.shape[1:], device=stat_device), # sum over batch dim
                        "sq_sum": torch.zeros(output_sample.shape[1:], device=stat_device), # sum over batch dim
                        "count": torch.tensor(0, dtype=torch.long, device=stat_device)
                    }
                
                stat_device = stats[name]["sum"].device # Get the actual device where stats are stored
                x = output[0].detach() # x is on self.device_manager.primary_device
                
                # Process in chunks to save memory (chunking along batch dimension of x)
                chunk_size = 1024  # Adjust based on available memory for x_batch_dim * x_other_dims
                num_chunks = math.ceil(x.size(0) / chunk_size)
                
                for i in range(num_chunks):
                    chunk_on_primary = x[i * chunk_size:(i + 1) * chunk_size]
                    chunk_for_stats = move_to_device(chunk_on_primary, stat_device)
                    
                    # Update statistics on stat_device
                    stats[name]["min_val"] = torch.min(stats[name]["min_val"], torch.min(chunk_for_stats.view(-1, chunk_for_stats.size(-1)), dim=0)[0])
                    stats[name]["max_val"] = torch.max(stats[name]["max_val"], torch.max(chunk_for_stats.view(-1, chunk_for_stats.size(-1)), dim=0)[0])
                    stats[name]["sum"] += torch.sum(chunk_for_stats, dim=0) # Summing over batch for each feature
                    stats[name]["sq_sum"] += torch.sum(chunk_for_stats ** 2, dim=0) # Summing over batch
                    stats[name]["count"] += chunk_for_stats.size(0)
                    
                    # Clear intermediate tensors
                    del chunk_on_primary, chunk_for_stats
                    if i % 10 == 0:  # Periodic memory cleanup
                        self._clear_memory()
            
            return fn
        
        # Register hooks for all linear layers
        hooks = []
        # self.model is already on self.device_manager.primary_device due to BaseQuantizer._prepare_model
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # module is part of self.model, so it's on self.device_manager.primary_device
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Process calibration data in batches
        # calibration_data is already on self.device_manager.primary_device from self.prepare_calibration_data
        with torch.no_grad():
            for i in range(0, len(data), self.batch_size):
                batch = move_to_device(data[i:i+self.batch_size], self.device_manager.primary_device)
                self.model(batch) # Model and batch are on primary_device
                
                # Periodic memory cleanup
                if i % (self.batch_size * 10) == 0:
                    self._clear_memory()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute final statistics
        for name in stats:
            current_count = stats[name]["count"].item() # Get Python number for conditional
            if current_count > 0:
                stats[name]["mean"] = stats[name]["sum"] / stats[name]["count"]
                var = stats[name]["sq_sum"] / stats[name]["count"] - stats[name]["mean"] ** 2
                # Ensure variance is not negative due to floating point inaccuracies
                stats[name]["std"] = torch.sqrt(torch.max(torch.tensor(0.0, device=var.device), var))
            else:
                stat_device = stats[name]["sum"].device
                # Keep original shape for sum/sq_sum by using zeros_like
                stats[name]["mean"] = torch.zeros_like(stats[name]["sum"], device=stat_device)
                stats[name]["std"] = torch.zeros_like(stats[name]["sum"], device=stat_device)
            
            # Clean up intermediate values
            del stats[name]["sum"]
            del stats[name]["sq_sum"]
            del stats[name]["count"]
        
        return stats
        
    def _quantize_layer(
        self,
        layer: nn.Linear,
        stats: Optional[Dict[str, torch.Tensor]] = None
    ) -> QuantizedLinear:
        """Quantize a single layer to GGUF format with memory-efficient processing."""
        target_device = torch.device('cpu') if self.cpu_offload else self.device_manager.primary_device
        
        layer = move_to_device(layer, target_device)

        # Initialize quantized layer and move to target_device
        quantized = QuantizedLinear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            config=QuantizationConfig(
                bits=self.bits,
                scheme="symmetric",
                granularity="per-channel" if self.group_size > 0 else "per-tensor",
                calibration="minmax",
                channel_wise=self.group_size > 0,
                dtype=f"int{self.bits}",
                format="gguf",
                format_config={
                    "desc_act": self.desc_act,
                    "desc_ten": self.desc_ten,
                    "use_packed": self.use_packed,
                    "legacy_format": self.legacy_format,
                    "group_size": self.group_size
                }
            )
        )
        
        # Process in chunks to save memory
        chunk_size = 1024  # Adjust based on available memory
        
        
        quantized = quantized.to(target_device)

        # Copy bias if exists
        if layer.bias is not None:
            # layer.bias.data is already on target_device because layer was moved
            quantized.bias.data.copy_(layer.bias.data)
            
        # Get weight matrix, clone, and ensure it's on target_device
        W = move_to_device(layer.weight.data.clone(), target_device)
        
        # Apply statistics if available
        if stats is not None:
            # stats values (mean, std) are on stat_device from _collect_stats
            # Move them to W.device (target_device) for the operation
            mean = move_to_device(stats["mean"], W.device)
            std = move_to_device(stats["std"], W.device)
            W = (W - mean) / (std + 1e-6) # W is on target_device
        
        # Compute scales per group if using grouping
        # All intermediate tensors (max_abs, scale_val, group_chunk etc.) should be on target_device
        if self.group_size > 0:
            n_groups = W.shape[0] // self.group_size
            W_groups = W.view(n_groups, self.group_size, -1)
            
            scales_list = [] # Renamed from scales
            zero_points_list = [] # Renamed from zero_points
            
            # Process groups in chunks
            for start_idx in range(0, n_groups, chunk_size):
                end_idx = min(start_idx + chunk_size, n_groups)
                group_chunk = W_groups[start_idx:end_idx] # group_chunk is on target_device
                
                current_chunk_scales = []
                current_chunk_zero_points = []
                
                for group_idx_in_chunk in range(group_chunk.size(0)):
                    group = group_chunk[group_idx_in_chunk] # group is on target_device
                    max_abs = torch.max(torch.abs(group))
                    current_scale_val = (2 ** (self.bits - 1) - 1) / max_abs # current_scale_val on target_device
                    current_chunk_scales.append(current_scale_val)
                    current_chunk_zero_points.append(torch.zeros_like(current_scale_val, device=target_device)) # Ensure on target_device
                
                scales_list.extend(current_chunk_scales)
                zero_points_list.extend(current_chunk_zero_points)
                
                # Clear intermediate tensors
                del group_chunk, current_chunk_scales, current_chunk_zero_points
                self._clear_memory()
            
            scales = torch.stack(scales_list) # scales is on target_device
            zero_points = torch.stack(zero_points_list) # zero_points is on target_device
        else:
            # Global quantization
            max_abs = torch.max(torch.abs(W)) # W is on target_device
            scale_val = (2 ** (self.bits - 1) - 1) / max_abs # scale_val is on target_device
            # target_device is already defined
            scales = torch.full(
                (W.size(0),),
                scale_val,
                device=target_device # Ensure scales is on target_device
            )
            zero_points = torch.zeros_like(scales, device=target_device) # Ensure zero_points is on target_device
            
        # Quantize weights in chunks
        W_quant_list = [] # Renamed from W_quant
        for start_idx in range(0, W.size(0), chunk_size):
            end_idx = min(start_idx + chunk_size, W.size(0))
            # W, scales are on target_device
            chunk = W[start_idx:end_idx] 
            current_chunk_scales = scales[start_idx:end_idx] # Renamed
            
            chunk_quant = torch.round(chunk * current_chunk_scales.view(-1, 1)) # chunk_quant on target_device
            
            # Pack groups if enabled
            if self.use_packed and self.group_size > 0:
                chunk_quant = chunk_quant.view(-1, self.group_size, W.size(1))
                if self.bits == 4:
                    # Pack two 4-bit values into one byte
                    chunk_quant = torch.clamp(chunk_quant, -8, 7)
                    chunk_packed = (chunk_quant[..., ::2] + 8) | ((chunk_quant[..., 1::2] + 8) << 4)
                    chunk_quant = chunk_packed.to(torch.uint8)
                elif self.bits == 2:
                    # Pack four 2-bit values into one byte
                    chunk_quant = torch.clamp(chunk_quant, -2, 1)
                    chunk_packed = (chunk_quant[..., ::4] + 2) | \
                                ((chunk_quant[..., 1::4] + 2) << 2) | \
                                ((chunk_quant[..., 2::4] + 2) << 4) | \
                                ((chunk_quant[..., 3::4] + 2) << 6)
                    chunk_quant = chunk_packed.to(torch.uint8)
                elif self.bits == 8:
                    chunk_quant = chunk_quant.to(torch.int8) # Already on target_device
                
                chunk_quant = chunk_quant.view(chunk.size(0), -1) # Reshape back
            
            W_quant_list.append(chunk_quant)
            
            # Clear intermediate tensors
            del chunk, current_chunk_scales, chunk_quant
            if 'chunk_packed' in locals(): del chunk_packed
            self._clear_memory()
        
        W_quant = torch.cat(W_quant_list, dim=0) # W_quant is on target_device
        
        # scales and zero_points are already on target_device
        # W_quant is already on target_device and correct dtype from packing logic or .to(torch.int8)
        
        # Store quantized weights and parameters
        # quantized and its buffers are on target_device
        quantized.weight_quantized.copy_(W_quant)
        quantized.weight_scale.copy_(1.0 / scales)
        quantized.weight_zero_point.copy_(zero_points)
        
        # Store additional GGUF-specific information
        if stats is not None:
            # stats values (mean, std) are on stat_device from _collect_stats
            # Move them to target_device for storing in quantized module's buffers (if they exist)
            if hasattr(quantized, 'input_mean'):
                quantized.input_mean.copy_(move_to_device(stats["mean"], target_device))
            if hasattr(quantized, 'input_std'):
                quantized.input_std.copy_(move_to_device(stats["std"], target_device))
        
        return quantized
    def convert_to_gguf(self, output_path: str):
        """Convert quantized model to GGUF format using CTransformers."""
        if not CT_AVAILABLE:
            raise ImportError("CTransformers is required for GGUF conversion")
            
        try:
            # Extract configuration
            config = {
                "model_type": self.model.config.model_type,
                "vocab_size": self.model.config.vocab_size,
                "hidden_size": self.model.config.hidden_size,
                "intermediate_size": self.model.config.intermediate_size,
                "num_hidden_layers": self.model.config.num_hidden_layers,
                "num_attention_heads": self.model.config.num_attention_heads,
                "max_position_embeddings": self.model.config.max_position_embeddings,
                "layer_norm_eps": self.model.config.layer_norm_eps,
                "use_cache": True,
                "quantization": {
                    "bits": self.bits,
                    "group_size": self.group_size,
                    "desc_act": self.desc_act,
                    "desc_ten": self.desc_ten,
                    "use_packed": self.use_packed,
                    "legacy_format": self.legacy_format
                }
            }
            
            # Save tensors in chunks to avoid memory issues
            with open(output_path, 'wb') as f:
                # Write config
                ctransformers.save_config(config, f)
                
                # Save tensors in chunks
                for name, module in self.model.named_modules():
                    if isinstance(module, QuantizedLinear):
                        # Process weight quantization tensors
                        for tensor_name in ["weight_quantized", "weight_scale", "weight_zero_point"]:
                            if hasattr(module, tensor_name):
                                tensor = getattr(module, tensor_name)
                                chunk_size = 1024 * 1024  # 1MB chunks
                                
                                for start_idx in range(0, tensor.numel(), chunk_size):
                                    end_idx = min(start_idx + chunk_size, tensor.numel())
                                    chunk = tensor.view(-1)[start_idx:end_idx]
                                    
                                    # If self.cpu_offload is True, module tensors are already on CPU.
                                    # Otherwise, they are on primary_device and need to be moved to CPU.
                                    if not self.cpu_offload:
                                        chunk = move_to_device(chunk, torch.device('cpu'))
                                    
                                    ctransformers.save_tensor(
                                        f"{name}.{tensor_name}",
                                        chunk,
                                        start_idx,
                                        f
                                    )
                                    
                                    del chunk
                                    self._clear_memory()
                        
                        # Process bias if exists
                        if hasattr(module, "bias") and module.bias is not None:
                            bias_data = module.bias.data # Renamed to avoid conflict
                            if not self.cpu_offload:
                                bias_data = move_to_device(bias_data, torch.device('cpu'))
                            ctransformers.save_tensor(f"{name}.bias", bias_data, 0, f)
                            del bias_data
                            
                        self._clear_memory()
            
            print(f"Successfully saved GGUF model to {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert model to GGUF format: {str(e)}")
        finally:
            self._clear_memory()
