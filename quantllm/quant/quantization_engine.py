"""Efficient model quantization engine."""

from typing import Dict, Any, Optional, Union, List, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel
import numpy as np
from ..trainer.logger import TrainingLogger

class QuantizationConfig:
    """Configuration for quantization parameters."""
    
    def __init__(
        self,
        bits: int = 8,
        scheme: str = "symmetric",
        granularity: str = "per-tensor",
        calibration: str = "minmax",
        channel_wise: bool = False,
        dtype: str = "int8",
        format: Optional[str] = None,
        format_config: Optional[Dict[str, Any]] = None
    ):
        self.bits = bits
        self.scheme = scheme
        self.granularity = granularity
        self.calibration = calibration
        self.channel_wise = channel_wise
        self.dtype = dtype
        self.format = format
        self.format_config = format_config or {}
        self.validate()
        
    def validate(self):
        """Validate configuration parameters."""
        valid_schemes = {"symmetric", "asymmetric", "power_of_2"}
        valid_granularity = {"per-tensor", "per-channel"}
        valid_calibration = {"minmax", "histogram", "entropy"}
        valid_dtypes = {"int8", "uint8", "int4", "uint4", "int2", "uint2"}
        valid_formats = {None, "gguf", "gptq", "awq"}
        
        if self.scheme not in valid_schemes:
            raise ValueError(f"Invalid quantization scheme: {self.scheme}")
        if self.granularity not in valid_granularity:
            raise ValueError(f"Invalid granularity: {self.granularity}")
        if self.calibration not in valid_calibration:
            raise ValueError(f"Invalid calibration method: {self.calibration}")
        if self.dtype not in valid_dtypes:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if not 2 <= self.bits <= 8:
            raise ValueError(f"Bits must be between 2 and 8, got {self.bits}")
        if self.format not in valid_formats:
            raise ValueError(f"Invalid format: {self.format}")

class QuantizedLinear(nn.Module):
    """Memory-efficient quantized linear layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: QuantizationConfig = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()
        
        # Quantized parameters
        self.register_buffer(
            'weight_scale',
            torch.ones(out_features if config.channel_wise else 1)
        )
        self.register_buffer(
            'weight_zero_point',
            torch.zeros(out_features if config.channel_wise else 1)
        )
        
        # Initialize quantized weights
        weight_shape = (out_features, in_features)
        self.register_buffer(
            'weight_quantized',
            torch.zeros(weight_shape, dtype=torch.int8)
        )
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights."""
        # Dequantize weights
        weight_deq = (
            self.weight_quantized.float() * self.weight_scale[:, None]
            + self.weight_zero_point[:, None]
        )
        
        # Compute output
        output = torch.nn.functional.linear(x, weight_deq)
        if hasattr(self, 'bias'):
            output += self.bias
            
        return output

class QuantizationEngine:
    """Engine for model quantization operations."""
    
    def __init__(
        self,
        config: QuantizationConfig,
        logger: Optional[TrainingLogger] = None
    ):
        self.config = config
        self.logger = logger or TrainingLogger()
        
    def quantize_model(
        self,
        model: PreTrainedModel,
        calibration_data: Optional[torch.Tensor] = None
    ) -> PreTrainedModel:
        """
        Quantize a pre-trained model.
        
        Args:
            model: Model to quantize
            calibration_data: Optional calibration data
            
        Returns:
            Quantized model
        """
        try:
            # Clone model for quantization
            model = self._prepare_model(model)
            
            # Collect calibration statistics if needed
            if calibration_data is not None:
                stats = self._collect_stats(model, calibration_data)
            else:
                stats = None
                
            # Quantize layers
            self._quantize_layers(model, stats)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error during quantization: {str(e)}")
            raise
            
    def _prepare_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Prepare model for quantization."""
        # Clone model
        model = model.cpu()
        model.eval()
        
        # Replace layers with quantizable versions
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                setattr(
                    model,
                    name,
                    self._create_quantized_linear(module)
                )
            elif len(list(module.children())) > 0:
                setattr(
                    model,
                    name,
                    self._prepare_model(module)
                )
                
        return model
    
    def _create_quantized_linear(
        self,
        layer: nn.Linear
    ) -> QuantizedLinear:
        """Create quantized version of linear layer."""
        quantized = QuantizedLinear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            config=self.config
        )
        
        # Copy weights and bias
        if self.config.scheme == "symmetric":
            max_val = layer.weight.abs().max()
            scale = (2 ** (self.config.bits - 1) - 1) / max_val
            zero_point = 0
        else:
            min_val = layer.weight.min()
            max_val = layer.weight.max()
            scale = (2 ** self.config.bits - 1) / (max_val - min_val)
            zero_point = -min_val * scale
            
        weight_quantized = torch.clamp(
            torch.round(layer.weight.data * scale - zero_point),
            -(2 ** (self.config.bits - 1)),
            2 ** (self.config.bits - 1) - 1
        ).to(torch.int8)
        
        quantized.weight_quantized.copy_(weight_quantized)
        quantized.weight_scale.fill_(1.0 / scale)
        quantized.weight_zero_point.fill_(zero_point)
        
        if layer.bias is not None:
            quantized.bias.copy_(layer.bias)
            
        return quantized
    
    def _collect_stats(
        self,
        model: PreTrainedModel,
        data: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collect calibration statistics."""
        stats = {}
        
        def hook_fn(name):
            def fn(module, input, output):
                if name not in stats:
                    stats[name] = {
                        "min": float('inf'),
                        "max": float('-inf')
                    }
                stats[name]["min"] = min(
                    stats[name]["min"],
                    output.min().item()
                )
                stats[name]["max"] = max(
                    stats[name]["max"],
                    output.max().item()
                )
            return fn
            
        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, QuantizedLinear)):
                handles.append(
                    module.register_forward_hook(hook_fn(name))
                )
                
        # Run calibration
        with torch.no_grad():
            model(data)
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        return stats
    
    def _quantize_layers(
        self,
        model: PreTrainedModel,
        stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ):
        """Quantize model layers using statistics."""
        if stats is None:
            return
            
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear) and name in stats:
                # Update quantization parameters based on stats
                min_val = stats[name]["min"]
                max_val = stats[name]["max"]
                
                if self.config.scheme == "symmetric":
                    max_abs = max(abs(min_val), abs(max_val))
                    scale = (2 ** (self.config.bits - 1) - 1) / max_abs
                    zero_point = 0
                else:
                    scale = (2 ** self.config.bits - 1) / (max_val - min_val)
                    zero_point = -min_val * scale
                    
                module.weight_scale.fill_(1.0 / scale)
                module.weight_zero_point.fill_(zero_point)
                
    def export_model(
        self,
        model: PreTrainedModel,
        path: str,
        format: str = "onnx"
    ):
        """Export quantized model."""
        try:
            if format == "onnx":
                import onnx
                import onnxruntime
                
                # Export to ONNX
                dummy_input = torch.zeros(
                    1,
                    model.config.max_position_embeddings,
                    model.config.hidden_size
                )
                
                torch.onnx.export(
                    model,
                    dummy_input,
                    path,
                    opset_version=13,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size', 1: 'sequence'},
                        'output': {0: 'batch_size', 1: 'sequence'}
                    }
                )
                
                # Optimize ONNX model
                model = onnx.load(path)
                model = onnxruntime.OnnxModelCompressionOperator(model)
                model.optimize()
                onnx.save(model, path)
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting model: {str(e)}")
            raise
            
    def benchmark(
        self,
        model: PreTrainedModel,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Benchmark quantized model performance."""
        try:
            model = model.cuda() if torch.cuda.is_available() else model
            dummy_input = torch.randn(*input_shape)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                torch.cuda.synchronize()
                
            # Warmup
            for _ in range(10):
                model(dummy_input)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            # Benchmark
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            latencies = []
            
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    start_time.record()
                else:
                    start = torch.cuda.Event(enable_timing=True)
                    
                model(dummy_input)
                
                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    latencies.append(
                        start_time.elapsed_time(end_time)
                    )
                else:
                    end = torch.cuda.Event(enable_timing=True)
                    latencies.append(
                        (end - start) * 1000  # Convert to ms
                    )
                    
            latencies = torch.tensor(latencies)
            
            return {
                "mean_latency": latencies.mean().item(),
                "std_latency": latencies.std().item(),
                "min_latency": latencies.min().item(),
                "max_latency": latencies.max().item(),
                "p90_latency": torch.quantile(latencies, 0.9).item(),
                "p95_latency": torch.quantile(latencies, 0.95).item(),
                "p99_latency": torch.quantile(latencies, 0.99).item()
            }
            
        except Exception as e:
            self.logger.error(f"Error during benchmarking: {str(e)}")
            raise