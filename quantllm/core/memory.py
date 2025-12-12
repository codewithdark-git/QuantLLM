"""
Memory Optimization Utilities for QuantLLM v2.0

Advanced memory management for training and inference of large models
on limited GPU memory.

Features:
- Dynamic layer offloading (CPU<->GPU)
- Automatic gradient checkpointing
- Optimizer state CPU offloading
- Memory-efficient data loading
"""

import gc
from typing import Optional, Dict, Any, List, Union, Callable
from contextlib import contextmanager
import torch
import torch.nn as nn

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryManager:
    """
    Centralized memory management for optimal GPU/CPU utilization.
    
    Features:
        - Automatic memory monitoring
        - Intelligent garbage collection
        - Layer-wise memory tracking
        - OOM prevention strategies
    
    Example:
        >>> mm = MemoryManager()
        >>> with mm.optimize_inference():
        ...     output = model(inputs)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._memory_cache: Dict[str, float] = {}
    
    @staticmethod
    def get_gpu_memory_info(device_id: int = 0) -> Dict[str, float]:
        """Get current GPU memory status."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        try:
            props = torch.cuda.get_device_properties(device_id)
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            total = props.total_memory
            
            return {
                "available": True,
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "free_gb": (total - reserved) / (1024**3),
                "usage_percent": 100 * reserved / total,
            }
        except Exception:
            return {"available": False}
    
    @staticmethod
    def get_system_memory_info() -> Dict[str, float]:
        """Get system RAM status."""
        if not PSUTIL_AVAILABLE:
            return {"available": False}
        
        try:
            mem = psutil.virtual_memory()
            return {
                "available": True,
                "total_gb": mem.total / (1024**3),
                "used_gb": mem.used / (1024**3),
                "free_gb": mem.available / (1024**3),
                "usage_percent": mem.percent,
            }
        except Exception:
            return {"available": False}
    
    @staticmethod
    def clear_cache(full: bool = True) -> None:
        """
        Clear memory caches.
        
        Args:
            full: If True, also run garbage collection
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        if full:
            gc.collect()
    
    @contextmanager
    def optimize_inference(self):
        """
        Context manager for memory-efficient inference.
        
        Enables inference mode and cleans up after.
        """
        try:
            with torch.inference_mode():
                yield
        finally:
            self.clear_cache(full=False)
    
    @contextmanager
    def memory_efficient_forward(self, model: nn.Module):
        """
        Context manager that temporarily enables memory optimizations.
        """
        original_training = model.training
        model.eval()
        
        try:
            with torch.inference_mode():
                yield
        finally:
            if original_training:
                model.train()
            self.clear_cache(full=False)
    
    def estimate_model_memory(
        self,
        model: nn.Module,
        batch_size: int = 1,
        seq_length: int = 512,
        include_gradients: bool = False,
        include_optimizer: bool = False,
    ) -> Dict[str, float]:
        """
        Estimate memory requirements for a model.
        
        Args:
            model: The model to analyze
            batch_size: Batch size for estimation
            seq_length: Sequence length for transformers
            include_gradients: Include gradient memory
            include_optimizer: Include optimizer states (2x for Adam)
            
        Returns:
            Dict with memory estimates in GB
        """
        # Model parameters
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        param_gb = param_bytes / (1024**3)
        
        # Buffers (non-trainable)
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        buffer_gb = buffer_bytes / (1024**3)
        
        # Gradients (same size as parameters)
        grad_gb = param_gb if include_gradients else 0
        
        # Optimizer states (2x for Adam's m and v)
        optimizer_gb = param_gb * 2 if include_optimizer else 0
        
        # Activation memory estimation (rough)
        # For transformers: batch * seq * hidden * layers * 2 (for intermediate + output)
        hidden_size = getattr(model.config, 'hidden_size', 4096) if hasattr(model, 'config') else 4096
        num_layers = getattr(model.config, 'num_hidden_layers', 32) if hasattr(model, 'config') else 32
        
        activation_bytes = batch_size * seq_length * hidden_size * num_layers * 2 * 4  # float32
        activation_gb = activation_bytes / (1024**3)
        
        total_gb = param_gb + buffer_gb + grad_gb + optimizer_gb + activation_gb
        
        return {
            "parameters_gb": param_gb,
            "buffers_gb": buffer_gb,
            "gradients_gb": grad_gb,
            "optimizer_states_gb": optimizer_gb,
            "activations_estimate_gb": activation_gb,
            "total_estimate_gb": total_gb,
        }


class DynamicOffloader:
    """
    Dynamic layer offloading for large models.
    
    Automatically moves layers between CPU and GPU based on
    memory availability and access patterns.
    
    Example:
        >>> offloader = DynamicOffloader(model, max_gpu_layers=20)
        >>> output = offloader.forward(inputs)
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_gpu_layers: Optional[int] = None,
        cpu_offload_threshold: float = 0.85,
    ):
        """
        Initialize dynamic offloader.
        
        Args:
            model: Model with layers to offload
            max_gpu_layers: Maximum layers to keep on GPU
            cpu_offload_threshold: GPU memory usage threshold to trigger offload
        """
        self.model = model
        self.cpu_offload_threshold = cpu_offload_threshold
        self._layer_locations: Dict[str, str] = {}
        self._access_order: List[str] = []
        
        # Get all named modules that can be offloaded
        self.offloadable_layers = self._find_offloadable_layers()
        
        # Determine max GPU layers
        if max_gpu_layers is None:
            gpu_mem = MemoryManager.get_gpu_memory_info()
            if gpu_mem.get("available"):
                # Estimate based on available memory
                free_gb = gpu_mem.get("free_gb", 8)
                avg_layer_size = self._estimate_avg_layer_size()
                max_gpu_layers = int(free_gb / max(avg_layer_size, 0.1))
            else:
                max_gpu_layers = len(self.offloadable_layers)
        
        self.max_gpu_layers = min(max_gpu_layers, len(self.offloadable_layers))
        
        # Initialize: keep first N layers on GPU
        self._initialize_placements()
    
    def _find_offloadable_layers(self) -> List[str]:
        """Find layers that can be offloaded."""
        layers = []
        for name, module in self.model.named_modules():
            # Look for transformer layers
            if any(pattern in name.lower() for pattern in ['layer', 'block', 'decoder']):
                if hasattr(module, 'parameters') and list(module.parameters()):
                    # Only include top-level repeated layers
                    if name.count('.') <= 2:  # Avoid nested modules
                        layers.append(name)
        return layers
    
    def _estimate_avg_layer_size(self) -> float:
        """Estimate average layer size in GB."""
        if not self.offloadable_layers:
            return 0.5  # Default
        
        total_size = 0
        for name in self.offloadable_layers[:5]:  # Sample first 5
            layer = dict(self.model.named_modules())[name]
            size = sum(p.numel() * p.element_size() for p in layer.parameters())
            total_size += size
        
        return (total_size / min(5, len(self.offloadable_layers))) / (1024**3)
    
    def _initialize_placements(self):
        """Initialize layer placements."""
        for i, name in enumerate(self.offloadable_layers):
            if i < self.max_gpu_layers:
                self._layer_locations[name] = "cuda"
            else:
                self._layer_locations[name] = "cpu"
                # Move to CPU
                layer = dict(self.model.named_modules())[name]
                layer.to("cpu")
    
    def move_layer_to_gpu(self, layer_name: str) -> None:
        """Move a layer to GPU, evicting others if needed."""
        if self._layer_locations.get(layer_name) == "cuda":
            return  # Already on GPU
        
        # Check if we need to evict
        gpu_layers = [n for n, loc in self._layer_locations.items() if loc == "cuda"]
        
        if len(gpu_layers) >= self.max_gpu_layers:
            # Evict least recently accessed
            for evict_name in reversed(self._access_order):
                if evict_name in gpu_layers and evict_name != layer_name:
                    self._move_to_cpu(evict_name)
                    break
        
        # Move requested layer to GPU
        layer = dict(self.model.named_modules())[layer_name]
        layer.to("cuda")
        self._layer_locations[layer_name] = "cuda"
        
        # Update access order
        if layer_name in self._access_order:
            self._access_order.remove(layer_name)
        self._access_order.insert(0, layer_name)
    
    def _move_to_cpu(self, layer_name: str) -> None:
        """Move a layer to CPU."""
        layer = dict(self.model.named_modules())[layer_name]
        layer.to("cpu")
        self._layer_locations[layer_name] = "cpu"
        torch.cuda.empty_cache()
    
    def prefetch_next_layer(self, current_layer_idx: int) -> None:
        """Prefetch the next layer to GPU asynchronously."""
        if current_layer_idx + 1 < len(self.offloadable_layers):
            next_layer = self.offloadable_layers[current_layer_idx + 1]
            self.move_layer_to_gpu(next_layer)


class GradientCheckpointManager:
    """
    Intelligent gradient checkpointing management.
    
    Automatically enables gradient checkpointing for memory-constrained
    training scenarios.
    """
    
    @staticmethod
    def enable_for_model(model: nn.Module, use_reentrant: bool = False) -> bool:
        """
        Enable gradient checkpointing for a model.
        
        Args:
            model: The model to enable checkpointing for
            use_reentrant: Whether to use reentrant checkpointing
            
        Returns:
            True if successfully enabled
        """
        if hasattr(model, 'gradient_checkpointing_enable'):
            # HuggingFace models
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
                return True
            except TypeError:
                # Older versions don't support kwargs
                model.gradient_checkpointing_enable()
                return True
        
        # Manual implementation for other models
        return GradientCheckpointManager._enable_manual(model)
    
    @staticmethod
    def _enable_manual(model: nn.Module) -> bool:
        """Manually enable gradient checkpointing."""
        from torch.utils.checkpoint import checkpoint
        
        checkpointed = False
        for name, module in model.named_modules():
            if 'layer' in name.lower() or 'block' in name.lower():
                if hasattr(module, 'forward'):
                    original_forward = module.forward
                    
                    def checkpointed_forward(*args, _orig=original_forward, **kwargs):
                        return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
                    
                    module.forward = checkpointed_forward
                    checkpointed = True
        
        return checkpointed
    
    @staticmethod
    def estimate_memory_savings(model: nn.Module) -> Dict[str, float]:
        """
        Estimate memory savings from gradient checkpointing.
        
        Returns:
            Dict with memory estimates
        """
        num_layers = 0
        hidden_size = 4096
        
        if hasattr(model, 'config'):
            num_layers = getattr(model.config, 'num_hidden_layers', 32)
            hidden_size = getattr(model.config, 'hidden_size', 4096)
        
        # Without checkpointing: activations for all layers
        # With checkpointing: activations for sqrt(layers) on average
        
        normal_activations = num_layers
        checkpointed_activations = num_layers ** 0.5
        
        savings_ratio = 1 - (checkpointed_activations / normal_activations) if normal_activations > 0 else 0
        
        return {
            "layers": num_layers,
            "activation_reduction_ratio": savings_ratio,
            "estimated_savings_percent": savings_ratio * 100,
            "trade_off": "~20-30% slower training for significant memory savings"
        }


class CPUOffloadOptimizer:
    """
    Optimizer with CPU-offloaded states for memory efficiency.
    
    Keeps optimizer states (momentum, variance for Adam) on CPU
    and only moves them to GPU during the update step.
    """
    
    def __init__(
        self,
        params,
        optimizer_class: type = torch.optim.AdamW,
        **optimizer_kwargs
    ):
        """
        Initialize CPU-offloaded optimizer.
        
        Args:
            params: Model parameters to optimize
            optimizer_class: Base optimizer class to use
            **optimizer_kwargs: Arguments for the optimizer
        """
        # Create CPU copies of parameters for optimizer
        self.cpu_params = []
        self.gpu_params = []
        
        for p in params:
            if p.requires_grad:
                cpu_p = p.detach().clone().to("cpu")
                cpu_p.requires_grad = True
                self.cpu_params.append(cpu_p)
                self.gpu_params.append(p)
        
        # Create optimizer on CPU parameters
        self.optimizer = optimizer_class(self.cpu_params, **optimizer_kwargs)
        self.optimizer_class = optimizer_class
    
    def step(self):
        """Perform optimization step with CPU-GPU synchronization."""
        # Copy gradients from GPU to CPU
        for cpu_p, gpu_p in zip(self.cpu_params, self.gpu_params):
            if gpu_p.grad is not None:
                cpu_p.grad = gpu_p.grad.detach().to("cpu")
        
        # Perform optimizer step on CPU
        self.optimizer.step()
        
        # Copy updated parameters back to GPU
        for cpu_p, gpu_p in zip(self.cpu_params, self.gpu_params):
            gpu_p.data.copy_(cpu_p.data.to(gpu_p.device))
    
    def zero_grad(self):
        """Zero gradients on both CPU and GPU parameters."""
        self.optimizer.zero_grad()
        for gpu_p in self.gpu_params:
            if gpu_p.grad is not None:
                gpu_p.grad.zero_()
    
    @property
    def param_groups(self):
        """Get parameter groups from the underlying optimizer."""
        return self.optimizer.param_groups


def setup_memory_efficient_training(
    model: nn.Module,
    gradient_checkpointing: bool = True,
    cpu_offload_optimizer: bool = False,
    learning_rate: float = 2e-4,
) -> Dict[str, Any]:
    """
    Configure a model and optimizer for memory-efficient training.
    
    Args:
        model: Model to configure
        gradient_checkpointing: Enable gradient checkpointing
        cpu_offload_optimizer: Use CPU-offloaded optimizer
        learning_rate: Learning rate for optimizer
        
    Returns:
        Dict with configured components
        
    Example:
        >>> components = setup_memory_efficient_training(model)
        >>> optimizer = components['optimizer']
        >>> # Training loop...
    """
    result = {"model": model}
    
    # Enable gradient checkpointing
    if gradient_checkpointing:
        success = GradientCheckpointManager.enable_for_model(model)
        result["gradient_checkpointing_enabled"] = success
        if success:
            savings = GradientCheckpointManager.estimate_memory_savings(model)
            result["estimated_memory_savings"] = savings
    
    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    if cpu_offload_optimizer:
        optimizer = CPUOffloadOptimizer(
            params,
            optimizer_class=torch.optim.AdamW,
            lr=learning_rate,
        )
        result["optimizer_type"] = "cpu_offload"
    else:
        optimizer = torch.optim.AdamW(params, lr=learning_rate)
        result["optimizer_type"] = "standard"
    
    result["optimizer"] = optimizer
    
    return result
