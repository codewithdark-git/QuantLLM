"""
Compilation Utilities for QuantLLM

torch.compile integration for maximum inference speed.
"""

from typing import Optional, Literal, Any
import torch
import torch.nn as nn


# Compilation mode types
CompileMode = Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]


def is_compile_supported() -> bool:
    """Check if torch.compile is supported on this system."""
    if not hasattr(torch, 'compile'):
        return False
    
    # Requires PyTorch 2.0+
    try:
        major, minor = torch.__version__.split('.')[:2]
        return int(major) >= 2
    except (ValueError, AttributeError):
        return False


def compile_model(
    model: nn.Module,
    mode: CompileMode = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = True,
    backend: str = "inductor",
    disable: bool = False,
) -> nn.Module:
    """
    Compile a model for optimized inference.
    
    Uses torch.compile with smart defaults for LLM inference.
    
    Args:
        model: Model to compile
        mode: Compilation mode:
            - "default": Balanced compilation
            - "reduce-overhead": Faster compilation, good for inference
            - "max-autotune": Best performance, slower compilation
            - "max-autotune-no-cudagraphs": Max perf without CUDA graphs
        fullgraph: If True, require full graph capture (may fail on dynamic control flow)
        dynamic: If True, handle dynamic shapes (recommended for LLMs)
        backend: Compilation backend (default: "inductor" for CUDA)
        disable: If True, skip compilation entirely
        
    Returns:
        Compiled model (or original if compilation not supported/disabled)
        
    Example:
        >>> model = TurboModel.from_pretrained("...")
        >>> model = compile_model(model, mode="reduce-overhead")
        >>> # Model is now compiled for faster inference
    """
    if disable:
        return model
    
    if not is_compile_supported():
        print("Warning: torch.compile not supported on this system")
        return model
    
    try:
        # Suppress compilation errors for robustness
        import torch._dynamo as dynamo
        dynamo.config.suppress_errors = True
        dynamo.config.automatic_dynamic_shapes = dynamic
        
        # Compile the model
        compiled = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
        )
        
        return compiled
        
    except Exception as e:
        print(f"Warning: torch.compile failed: {e}")
        return model


def compile_for_inference(model: nn.Module) -> nn.Module:
    """
    Compile model with inference-optimized settings.
    
    Uses settings optimized for inference rather than training:
    - reduce-overhead mode for faster warmup
    - Dynamic shapes for variable sequence lengths
    - Suppressed errors for robustness
    
    Args:
        model: Model to compile
        
    Returns:
        Compiled model
    """
    return compile_model(
        model,
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=True,
    )


def compile_for_training(model: nn.Module) -> nn.Module:
    """
    Compile model with training-optimized settings.
    
    Uses settings optimized for training:
    - default mode for balanced compilation
    - Dynamic shapes for gradient checkpointing compatibility
    
    Args:
        model: Model to compile
        
    Returns:
        Compiled model
    """
    return compile_model(
        model,
        mode="default",
        fullgraph=False,
        dynamic=True,
    )


def compile_for_max_speed(model: nn.Module) -> nn.Module:
    """
    Compile model for maximum inference speed.
    
    Warning: This uses max-autotune which has longer compilation time.
    Best for production deployment where you compile once and run many times.
    
    Args:
        model: Model to compile
        
    Returns:
        Compiled model
    """
    return compile_model(
        model,
        mode="max-autotune",
        fullgraph=True,  # Attempt full graph for best optimization
        dynamic=False,   # Static shapes for CUDA graphs
    )


class CompiledModelWrapper(nn.Module):
    """
    Wrapper that lazily compiles a model on first forward pass.
    
    This avoids compilation overhead until the model is actually used,
    and handles compilation failures gracefully.
    """
    
    def __init__(
        self,
        model: nn.Module,
        mode: CompileMode = "reduce-overhead",
        warmup_runs: int = 1,
    ):
        super().__init__()
        self.model = model
        self.mode = mode
        self.warmup_runs = warmup_runs
        self._compiled = False
        self._compiled_model = None
        self._run_count = 0
    
    def forward(self, *args, **kwargs):
        # Compile after warmup runs
        if not self._compiled and self._run_count >= self.warmup_runs:
            try:
                self._compiled_model = compile_model(self.model, mode=self.mode)
                self._compiled = True
            except Exception:
                self._compiled_model = self.model
                self._compiled = True
        
        self._run_count += 1
        
        if self._compiled:
            return self._compiled_model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)


def get_optimal_compile_settings(
    model_size_gb: float,
    gpu_memory_gb: float,
    inference_only: bool = True,
) -> dict:
    """
    Get optimal compilation settings based on model and hardware.
    
    Args:
        model_size_gb: Model size in GB
        gpu_memory_gb: Available GPU memory in GB
        inference_only: Whether model is for inference only
        
    Returns:
        Dict of compilation settings
    """
    # Memory headroom after model loading
    headroom = gpu_memory_gb - model_size_gb
    
    if headroom < 2:
        # Very tight memory - don't use CUDA graphs
        return {
            "mode": "default",
            "fullgraph": False,
            "dynamic": True,
        }
    elif headroom < 4:
        # Some memory available
        return {
            "mode": "reduce-overhead",
            "fullgraph": False,
            "dynamic": True,
        }
    else:
        # Plenty of memory - can use max autotune
        return {
            "mode": "max-autotune" if inference_only else "reduce-overhead",
            "fullgraph": inference_only,
            "dynamic": not inference_only,
        }
