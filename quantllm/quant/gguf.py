"""GGUF (GGML Universal Format) quantization implementation."""

import gc
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import PreTrainedModel
from .quantization_engine import BaseQuantizer, QuantizationConfig, QuantizedLinear

try:
    import ctransformers
    CT_AVAILABLE = True
except ImportError:
    CT_AVAILABLE = False

class GGUFQuantizer(BaseQuantizer):
    """GGUF quantization implementation with CTransformers integration."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        bits: int = 4,
        group_size: int = 32,
        desc_act: bool = False,
        desc_ten: bool = False,
        use_packed: bool = True,
        legacy_format: bool = False,
        batch_size: int = 4,
        device: Optional[Union[str, torch.device]] = None
    ):
        if not CT_AVAILABLE:
            raise ImportError("CTransformers is required for GGUF quantization. Install with: pip install ctransformers")
            
        super().__init__(model=model, bits=bits, device=device)
        self.group_size = group_size
        self.desc_act = desc_act
        self.desc_ten = desc_ten
        self.use_packed = use_packed
        self.legacy_format = legacy_format
        self.batch_size = batch_size
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
        
        return self.model
    
    def _collect_stats(self, data: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collect statistics for quantization with memory-efficient batch processing."""
        device = next(self.model.parameters()).device
        stats = {}
        
        def hook_fn(name):
            def fn(module, input, output):
                if name not in stats:
                    # Initialize stats on CPU if offloading is enabled
                    target_device = 'cpu' if self.cpu_offload else device
                    stats[name] = {
                        "min_val": torch.tensor(float('inf'), device=target_device),
                        "max_val": torch.tensor(float('-inf'), device=target_device),
                        "sum": torch.zeros_like(output[0], device=target_device),
                        "sq_sum": torch.zeros_like(output[0], device=target_device),
                        "count": 0
                    }
                
                x = output[0].detach()
                
                # Process in chunks to save memory
                chunk_size = 1024  # Adjust based on available memory
                num_chunks = math.ceil(x.size(0) / chunk_size)
                
                for i in range(num_chunks):
                    chunk = x[i * chunk_size:(i + 1) * chunk_size]
                    if self.cpu_offload:
                        chunk = chunk.cpu()
                    
                    # Update statistics
                    stats[name]["min_val"] = torch.min(stats[name]["min_val"], torch.min(chunk))
                    stats[name]["max_val"] = torch.max(stats[name]["max_val"], torch.max(chunk))
                    stats[name]["sum"] += torch.sum(chunk, dim=0)
                    stats[name]["sq_sum"] += torch.sum(chunk ** 2, dim=0)
                    stats[name]["count"] += chunk.size(0)
                    
                    # Clear intermediate tensors
                    del chunk
                    if i % 10 == 0:  # Periodic memory cleanup
                        self._clear_memory()
            
            return fn
        
        # Register hooks for all linear layers
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Process calibration data in batches
        with torch.no_grad():
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                self.model(batch)
                
                # Periodic memory cleanup
                if i % (self.batch_size * 10) == 0:
                    self._clear_memory()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute final statistics
        for name in stats:
            stats[name]["mean"] = stats[name]["sum"] / stats[name]["count"]
            stats[name]["std"] = torch.sqrt(
                stats[name]["sq_sum"] / stats[name]["count"] 
                - stats[name]["mean"] ** 2
            )
            
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
        
        # Copy bias if exists
        if layer.bias is not None:
            quantized.bias.data.copy_(layer.bias.data)
            
        # Get weight matrix
        W = layer.weight.data.clone()
        if self.cpu_offload:
            W = W.cpu()
        
        # Apply statistics if available
        if stats is not None:
            if self.cpu_offload:
                stats = {k: v.cpu() for k, v in stats.items()}
            W = (W - stats["mean"]) / (stats["std"] + 1e-6)
        
        # Compute scales per group if using grouping
        if self.group_size > 0:
            n_groups = W.shape[0] // self.group_size
            W_groups = W.view(n_groups, self.group_size, -1)
            
            scales = []
            zero_points = []
            
            # Process groups in chunks
            for start_idx in range(0, n_groups, chunk_size):
                end_idx = min(start_idx + chunk_size, n_groups)
                group_chunk = W_groups[start_idx:end_idx]
                
                chunk_scales = []
                chunk_zero_points = []
                
                for group in group_chunk:
                    max_abs = torch.max(torch.abs(group))
                    scale = (2 ** (self.bits - 1) - 1) / max_abs
                    chunk_scales.append(scale)
                    chunk_zero_points.append(torch.zeros_like(scale))
                
                scales.extend(chunk_scales)
                zero_points.extend(chunk_zero_points)
                
                # Clear intermediate tensors
                del group_chunk, chunk_scales, chunk_zero_points
                self._clear_memory()
            
            scales = torch.stack(scales)
            zero_points = torch.stack(zero_points)
        else:
            # Global quantization
            max_abs = torch.max(torch.abs(W))
            scales = torch.full(
                (W.size(0),),
                (2 ** (self.bits - 1) - 1) / max_abs,
                device='cpu' if self.cpu_offload else device
            )
            zero_points = torch.zeros_like(scales)
            
        # Quantize weights in chunks
        W_quant = []
        for start_idx in range(0, W.size(0), chunk_size):
            end_idx = min(start_idx + chunk_size, W.size(0))
            chunk = W[start_idx:end_idx]
            chunk_scales = scales[start_idx:end_idx]
            
            chunk_quant = torch.round(chunk * chunk_scales.view(-1, 1))
            
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
                    chunk_quant = chunk_quant.to(torch.int8)
                
                chunk_quant = chunk_quant.view(chunk.size(0), -1)
            
            W_quant.append(chunk_quant)
            
            # Clear intermediate tensors
            del chunk, chunk_scales, chunk_quant
            self._clear_memory()
        
        W_quant = torch.cat(W_quant, dim=0)
        
        # Move tensors to appropriate device
        target_device = device if not self.cpu_offload else 'cpu'
        W_quant = W_quant.to(target_device)
        scales = scales.to(target_device)
        zero_points = zero_points.to(target_device)
        
        # Store quantized weights and parameters
        quantized.weight_quantized.copy_(W_quant)
        quantized.weight_scale.copy_(1.0 / scales)
        quantized.weight_zero_point.copy_(zero_points)
        
        # Store additional GGUF-specific information
        if stats is not None:
            if hasattr(quantized, 'input_mean'):
                quantized.input_mean.copy_(stats["mean"].to(target_device))
            if hasattr(quantized, 'input_std'):
                quantized.input_std.copy_(stats["std"].to(target_device))
        
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
                                    
                                    if self.cpu_offload:
                                        chunk = chunk.cpu()
                                    
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
                            bias = module.bias.data
                            if self.cpu_offload:
                                bias = bias.cpu()
                            ctransformers.save_tensor(f"{name}.bias", bias, 0, f)
                            del bias
                            
                        self._clear_memory()
            
            print(f"Successfully saved GGUF model to {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert model to GGUF format: {str(e)}")
        finally:
            self._clear_memory()
