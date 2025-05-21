"""GGUF (GGML Universal Format) quantization implementation."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import PreTrainedModel
from .quantization_engine import QuantizationConfig, QuantizedLinear

try:
    import ctransformers
    CT_AVAILABLE = True
except ImportError:
    CT_AVAILABLE = False

class GGUFQuantizer:
    """GGUF quantization implementation with CTransformers integration."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        bits: int = 4,
        group_size: int = 32,
        desc_act: bool = False,
        desc_ten: bool = False,
        use_packed: bool = True,
        legacy_format: bool = False
    ):
        if not CT_AVAILABLE:
            raise ImportError("CTransformers is required for GGUF quantization. Install with: pip install ctransformers")
        
        self.model = model
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.desc_ten = desc_ten
        self.use_packed = use_packed
        self.legacy_format = legacy_format
        
    def quantize(
        self,
        calibration_data: Optional[torch.Tensor] = None
    ) -> PreTrainedModel:
        """
        Quantize model using GGUF format.
        
        Args:
            calibration_data: Optional tensor for computing quantization statistics
            
        Returns:
            Quantized model
        """
        # Prepare model for quantization
        self.model.eval()
        
        # Collect statistics if provided
        stats = {}
        if calibration_data is not None:
            stats = self._collect_stats(calibration_data)
        
        # Convert linear layers to quantized versions
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
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
        
        return self.model
        
    def _collect_stats(self, data: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collect statistics for quantization."""
        stats = {}
        
        def hook_fn(name):
            def fn(module, input, output):
                if name not in stats:
                    stats[name] = {
                        "min_val": torch.tensor(float('inf')),
                        "max_val": torch.tensor(float('-inf')),
                        "sum": torch.zeros_like(output[0]),
                        "sq_sum": torch.zeros_like(output[0]),
                        "count": 0
                    }
                    
                x = output[0].detach()
                stats[name]["min_val"] = torch.min(
                    stats[name]["min_val"],
                    torch.min(x)
                )
                stats[name]["max_val"] = torch.max(
                    stats[name]["max_val"],
                    torch.max(x)
                )
                stats[name]["sum"] += torch.sum(x, dim=0)
                stats[name]["sq_sum"] += torch.sum(x.pow(2), dim=0)
                stats[name]["count"] += x.size(0)
            return fn
            
        # Register hooks
        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(
                    module.register_forward_hook(hook_fn(name))
                )
                
        # Run calibration
        with torch.no_grad():
            self.model(data)
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # Process statistics
        for name in stats:
            count = stats[name]["count"]
            mean = stats[name]["sum"] / count
            var = (stats[name]["sq_sum"] / count) - mean.pow(2)
            std = torch.sqrt(var + 1e-6)
            
            stats[name] = {
                "min": stats[name]["min_val"],
                "max": stats[name]["max_val"],
                "mean": mean,
                "std": std
            }
            
        return stats
        
    def _quantize_layer(
        self,
        layer: nn.Linear,
        stats: Optional[Dict[str, torch.Tensor]] = None
    ) -> QuantizedLinear:
        """Quantize a single layer to GGUF format."""
        device = next(layer.parameters()).device
        
        # Initialize quantized layer
        quantized = QuantizedLinear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            config=QuantizationConfig(
                bits=self.bits,
                scheme="symmetric",
                granularity="per-group" if self.group_size > 0 else "per-tensor",
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
        
        # Copy bias if exists
        if layer.bias is not None:
            quantized.bias.data.copy_(layer.bias.data)
            
        # Get weight matrix
        W = layer.weight.data.clone()
        
        # Apply statistics if available
        if stats is not None:
            W = (W - stats["mean"]) / (stats["std"] + 1e-6)
        
        # Compute scales per group if using grouping
        if self.group_size > 0:
            n_groups = W.shape[0] // self.group_size
            W_groups = W.view(n_groups, self.group_size, -1)
            
            scales = []
            zero_points = []
            
            for idx in range(n_groups):
                group = W_groups[idx]
                max_abs = torch.max(torch.abs(group))
                scale = (2 ** (self.bits - 1) - 1) / max_abs
                scales.append(scale)
                zero_points.append(torch.zeros_like(scale))
                
            scales = torch.stack(scales)
            zero_points = torch.stack(zero_points)
        else:
            # Global quantization
            max_abs = torch.max(torch.abs(W))
            scales = torch.full(
                (W.size(0),),
                (2 ** (self.bits - 1) - 1) / max_abs,
                device=device
            )
            zero_points = torch.zeros_like(scales)
            
        # Quantize weights
        W_quant = torch.round(W * scales.view(-1, 1))
        
        # Pack groups if enabled
        if self.use_packed and self.group_size > 0:
            W_quant = W_quant.view(-1, self.group_size, W.size(1))
            if self.bits == 4:
                # Pack two 4-bit values into one byte
                W_quant = torch.clamp(W_quant, -8, 7)  # -8 to 7 for 4-bit
                packed = (W_quant[..., ::2] + 8) | ((W_quant[..., 1::2] + 8) << 4)
                W_quant = packed.to(torch.uint8).view(W.size(0), -1)
            elif self.bits == 2:
                # Pack four 2-bit values into one byte
                W_quant = torch.clamp(W_quant, -2, 1)  # -2 to 1 for 2-bit
                packed = (W_quant[..., ::4] + 2) | \
                        ((W_quant[..., 1::4] + 2) << 2) | \
                        ((W_quant[..., 2::4] + 2) << 4) | \
                        ((W_quant[..., 3::4] + 2) << 6)
                W_quant = packed.to(torch.uint8).view(W.size(0), -1)
            elif self.bits == 8:
                W_quant = W_quant.to(torch.int8).view(W.size(0), -1)
        
        # Store quantized weights and parameters
        quantized.weight_quantized.copy_(W_quant)
        quantized.weight_scale.copy_(1.0 / scales)
        quantized.weight_zero_point.copy_(zero_points)
        
        # Store additional GGUF-specific information
        if stats is not None:
            if hasattr(quantized, 'input_mean'):
                quantized.input_mean.copy_(stats["mean"])
            if hasattr(quantized, 'input_std'):
                quantized.input_std.copy_(stats["std"])
        
        return quantized
        
    def convert_to_gguf(self, output_path: str):
        """Convert quantized model to GGUF format using CTransformers."""
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
                    "use_packed": self.use_packed
                }
            }
            
            # Create GGUF file
            with open(output_path, 'wb') as f:
                # Write magic number and version
                f.write(b'GGUF')  # Magic
                f.write(torch.tensor(1, dtype=torch.int32).numpy().tobytes())  # Version
                
                # Write model configuration
                for key, value in config.items():
                    # Serialize based on type
                    if isinstance(value, (int, float)):
                        dtype = torch.int32 if isinstance(value, int) else torch.float32
                        f.write(torch.tensor(value, dtype=dtype).numpy().tobytes())
                    elif isinstance(value, str):
                        bytes_val = value.encode('utf-8')
                        f.write(torch.tensor(len(bytes_val), dtype=torch.int32).numpy().tobytes())
                        f.write(bytes_val)
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            dtype = torch.int32 if isinstance(v, int) else torch.float32
                            f.write(torch.tensor(v, dtype=dtype).numpy().tobytes())
                            
                # Write model weights
                for name, module in self.model.named_modules():
                    if isinstance(module, QuantizedLinear):
                        # Write layer name
                        name_bytes = name.encode('utf-8')
                        f.write(torch.tensor(len(name_bytes), dtype=torch.int32).numpy().tobytes())
                        f.write(name_bytes)
                        
                        # Write quantized weights
                        f.write(module.weight_quantized.numpy().tobytes())
                        f.write(module.weight_scale.numpy().tobytes())
                        f.write(module.weight_zero_point.numpy().tobytes())
                        
                        if module.bias is not None:
                            f.write(module.bias.numpy().tobytes())
                        
                        if hasattr(module, 'input_mean'):
                            f.write(module.input_mean.numpy().tobytes())
                            f.write(module.input_std.numpy().tobytes())
                            
        except Exception as e:
            raise RuntimeError(f"Failed to convert model to GGUF format: {str(e)}")
        
        return output_path
