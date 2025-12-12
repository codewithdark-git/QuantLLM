"""
Universal Export Module for QuantLLM v2.0

Provides unified export functionality to multiple formats:
- GGUF (llama.cpp, Ollama, LM Studio)
- SafeTensors (HuggingFace)
- ONNX (ONNX Runtime, TensorRT)
- MLX (Apple Silicon)
- AWQ (AutoAWQ)
"""

import os
import shutil
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import torch
import torch.nn as nn


class ExportFormat:
    """Supported export formats."""
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    MLX = "mlx"
    AWQ = "awq"
    PYTORCH = "pytorch"


class UniversalExporter:
    """
    Export models to any format with optimal settings.
    
    Example:
        >>> exporter = UniversalExporter(model, tokenizer)
        >>> exporter.export("gguf", "model.gguf", quant="Q4_K_M")
        >>> exporter.export("onnx", "model.onnx")
        >>> exporter.export("mlx", "./mlx_model/")
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._model_path: Optional[str] = None
    
    def export(
        self,
        format: str,
        output_path: str,
        *,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Export model to specified format.
        
        Args:
            format: Target format (gguf, safetensors, onnx, mlx, awq)
            output_path: Output file or directory path
            quantization: Quantization type (format-specific)
            **kwargs: Format-specific options
            
        Returns:
            Path to exported model
        """
        format = format.lower()
        
        exporters = {
            ExportFormat.GGUF: self._export_gguf,
            ExportFormat.SAFETENSORS: self._export_safetensors,
            ExportFormat.ONNX: self._export_onnx,
            ExportFormat.MLX: self._export_mlx,
            ExportFormat.AWQ: self._export_awq,
            ExportFormat.PYTORCH: self._export_pytorch,
        }
        
        if format not in exporters:
            available = list(exporters.keys())
            raise ValueError(f"Unknown format: {format}. Available: {available}")
        
        return exporters[format](output_path, quantization=quantization, **kwargs)
    
    def _export_gguf(
        self,
        output_path: str,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Export to GGUF format."""
        from ..quant import convert_to_gguf
        
        quant_type = quantization or "Q4_K_M"
        
        return convert_to_gguf(
            self.model,
            output_path,
            quant_type=quant_type,
            **kwargs,
        )
    
    def _export_safetensors(
        self,
        output_path: str,
        **kwargs,
    ) -> str:
        """Export to SafeTensors format."""
        os.makedirs(output_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_path, safe_serialization=True)
        
        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)
        
        return output_path
    
    def _export_onnx(
        self,
        output_path: str,
        quantization: Optional[str] = None,
        opset_version: int = 14,
        **kwargs,
    ) -> str:
        """
        Export to ONNX format.
        
        Args:
            output_path: Output ONNX file path
            quantization: ONNX quantization type (dynamic, static, qint8)
            opset_version: ONNX opset version
        """
        try:
            import onnx
            from torch.onnx import export as torch_onnx_export
        except ImportError:
            raise ImportError("ONNX export requires: pip install onnx onnxruntime")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Prepare dummy input
        if self.tokenizer:
            dummy_input = self.tokenizer(
                "Hello world",
                return_tensors="pt",
                padding=True,
            )
            dummy_input = {k: v.to(self.model.device) for k, v in dummy_input.items()}
        else:
            dummy_input = {
                "input_ids": torch.zeros(1, 32, dtype=torch.long, device=self.model.device)
            }
        
        # Export to ONNX
        self.model.eval()
        
        input_names = list(dummy_input.keys())
        output_names = ["logits"]
        
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        }
        
        torch_onnx_export(
            self.model,
            tuple(dummy_input.values()),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={k: v for k, v in dynamic_axes.items() if k in input_names or k in output_names},
            opset_version=opset_version,
            do_constant_folding=True,
        )
        
        # Apply quantization if requested
        if quantization:
            output_path = self._quantize_onnx(output_path, quantization)
        
        return output_path
    
    def _quantize_onnx(
        self,
        model_path: str,
        quant_type: str,
    ) -> str:
        """Apply ONNX quantization."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            raise ImportError("ONNX quantization requires: pip install onnxruntime")
        
        quantized_path = model_path.replace(".onnx", f"_{quant_type}.onnx")
        
        quantize_dynamic(
            model_path,
            quantized_path,
            weight_type=QuantType.QInt8,
        )
        
        return quantized_path
    
    def _export_mlx(
        self,
        output_path: str,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Export to MLX format for Apple Silicon.
        
        Args:
            output_path: Output directory
            quantization: MLX quantization (4bit, 8bit)
        """
        try:
            import mlx.core as mx
            from mlx_lm import convert
        except ImportError:
            raise ImportError(
                "MLX export requires Apple Silicon and: pip install mlx mlx-lm"
            )
        
        os.makedirs(output_path, exist_ok=True)
        
        # First save as HF format
        temp_hf_path = os.path.join(output_path, "_temp_hf")
        self.model.save_pretrained(temp_hf_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(temp_hf_path)
        
        # Convert to MLX
        mlx_path = os.path.join(output_path, "mlx_model")
        
        # Use mlx-lm convert
        convert_args = [temp_hf_path, "--mlx-path", mlx_path]
        if quantization:
            if quantization == "4bit":
                convert_args.extend(["-q", "--q-bits", "4"])
            elif quantization == "8bit":
                convert_args.extend(["-q", "--q-bits", "8"])
        
        # Clean up temp
        shutil.rmtree(temp_hf_path, ignore_errors=True)
        
        return mlx_path
    
    def _export_awq(
        self,
        output_path: str,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Export to AWQ (Activation-aware Weight Quantization) format.
        
        Provides better quality than naive quantization for 4-bit.
        """
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError("AWQ export requires: pip install autoawq")
        
        os.makedirs(output_path, exist_ok=True)
        
        # AWQ quantization config
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
        }
        
        if quantization:
            if "8" in quantization:
                quant_config["w_bit"] = 8
            if "group64" in quantization:
                quant_config["q_group_size"] = 64
        
        # Get model name for AWQ
        model_name = getattr(self.model.config, '_name_or_path', 'model')
        
        # Initialize AWQ model
        awq_model = AutoAWQForCausalLM.from_pretrained(model_name)
        
        # Quantize
        awq_model.quantize(
            self.tokenizer,
            quant_config=quant_config,
        )
        
        # Save
        awq_model.save_quantized(output_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)
        
        return output_path
    
    def _export_pytorch(
        self,
        output_path: str,
        **kwargs,
    ) -> str:
        """Export as PyTorch checkpoint."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config.to_dict() if hasattr(self.model, 'config') else {},
        }, output_path)
        
        return output_path
    
    @staticmethod
    def list_formats() -> Dict[str, str]:
        """List available export formats with descriptions."""
        return {
            ExportFormat.GGUF: "GGUF for llama.cpp, Ollama, LM Studio",
            ExportFormat.SAFETENSORS: "SafeTensors for HuggingFace ecosystem",
            ExportFormat.ONNX: "ONNX for ONNX Runtime and TensorRT",
            ExportFormat.MLX: "MLX for Apple Silicon Macs",
            ExportFormat.AWQ: "AWQ for high-quality 4-bit quantization",
            ExportFormat.PYTORCH: "PyTorch checkpoint (.pt)",
        }


def export_model(
    model: nn.Module,
    format: str,
    output_path: str,
    tokenizer: Any = None,
    **kwargs,
) -> str:
    """
    Export a model to any format.
    
    Convenience function for quick exports.
    
    Args:
        model: Model to export
        format: Target format (gguf, safetensors, onnx, mlx, awq)
        output_path: Output file or directory
        tokenizer: Optional tokenizer
        **kwargs: Format-specific options
        
    Returns:
        Path to exported model
        
    Example:
        >>> from quantllm.core import export_model
        >>> export_model(model, "gguf", "model-q4.gguf", quant="Q4_K_M")
        >>> export_model(model, "onnx", "model.onnx")
    """
    exporter = UniversalExporter(model, tokenizer)
    return exporter.export(format, output_path, **kwargs)
