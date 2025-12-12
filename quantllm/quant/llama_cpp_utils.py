"""
LlamaCpp Utilities - Legacy module for llama.cpp integration.

This module is deprecated in v2.0. Use the new GGUFConverter instead:
    from quantllm.quant import GGUFConverter, convert_to_gguf
"""

from typing import Optional, Dict, Any
import os
import subprocess
import sys


class LlamaCppConverter:
    """
    Legacy LlamaCpp converter class.
    
    DEPRECATED: Use GGUFConverter from quantllm.quant instead:
        from quantllm.quant import convert_to_gguf
        convert_to_gguf(model, "output.gguf", quant_type="Q4_K_M")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        """Initialize the converter."""
        self.model_path = model_path
        self.output_path = output_path
        self._llama_cpp_path: Optional[str] = None
        
    def convert(
        self,
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
        quant_type: str = "Q4_K_M",
    ) -> str:
        """
        Convert model to GGUF format.
        
        This is a legacy method. Use the new converter instead.
        """
        # Redirect to new converter
        from .gguf_converter import convert_to_gguf
        
        path = model_path or self.model_path
        out = output_path or self.output_path
        
        if not path or not out:
            raise ValueError("model_path and output_path are required")
        
        return convert_to_gguf(path, out, quant_type=quant_type)
    
    def is_available(self) -> bool:
        """Check if llama.cpp tools are available."""
        from .gguf_converter import GGUFConverter
        converter = GGUFConverter()
        return converter._quantize_bin is not None


class ProgressTracker:
    """Simple progress tracker for conversion."""
    
    def __init__(self):
        self.current = 0
        self.total = 100
    
    def update(self, progress: int, message: str = ""):
        """Update progress."""
        self.current = progress
        if message:
            print(f"[{progress}%] {message}")
    
    def finish(self):
        """Mark as complete."""
        self.current = 100
