"""
SmartConfig for QuantLLM.

Automatically configures optimal settings based on hardware
and model requirements. Zero-config experience.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch

from .hardware import HardwareProfiler, HardwareProfile
from .model_analyzer import ModelAnalyzer, ModelInfo


@dataclass
class SmartConfig:
    """
    Auto-configured settings for model loading and quantization.
    
    All parameters are automatically determined based on hardware
    and model characteristics, but can be overridden if needed.
    """
    
    # Quantization settings
    bits: int = 4
    quant_type: str = "Q4_K_M"
    group_size: int = 128
    
    # Memory settings
    max_memory: Dict[str, str] = field(default_factory=dict)
    cpu_offload: bool = False
    gradient_checkpointing: bool = False
    
    # Speed settings
    use_flash_attention: bool = True
    use_fused_kernels: bool = True
    compile_model: bool = False
    
    # Training settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Device settings
    device: torch.device = field(default_factory=lambda: torch.device("cuda:0"))
    dtype: torch.dtype = field(default_factory=lambda: torch.float16)
    
    # Model settings
    max_seq_length: int = 4096
    
    @classmethod
    def detect(
        cls,
        model_name: str,
        *,
        bits: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        training: bool = False,
    ) -> "SmartConfig":
        """
        Auto-detect optimal configuration for a model.
        
        Args:
            model_name: HuggingFace model name or local path
            bits: Override bit-width (auto-detect if None)
            max_seq_length: Override sequence length (auto-detect if None)
            device: Override device (auto-detect if None)
            dtype: Override dtype (auto-detect if None)
            training: Whether config is for training (uses more memory)
            
        Returns:
            SmartConfig with optimal settings
        """
        # Profile hardware
        hw = HardwareProfiler.detect()
        
        # Analyze model
        model_info = ModelAnalyzer.analyze(model_name)
        
        # Determine optimal settings
        config = cls()
        
        # Device
        if device is not None:
            config.device = torch.device(device)
        else:
            config.device = hw.device
        
        # Dtype
        if dtype is not None:
            config.dtype = getattr(torch, dtype)
        elif hw.supports_bf16:
            config.dtype = torch.bfloat16
        elif hw.cuda_available:
            config.dtype = torch.float16
        else:
            config.dtype = torch.float32
        
        # Bits
        if bits is not None:
            config.bits = bits
        else:
            config.bits = cls._choose_bits(hw, model_info, training)
        
        # Quant type
        config.quant_type = cls._choose_quant_type(config.bits, hw)
        
        # Group size (smaller = more accurate, larger = faster)
        config.group_size = cls._choose_group_size(hw, model_info)
        
        # Memory settings
        if hw.gpus:
            available_memory = hw.best_gpu.memory_free_gb if hw.best_gpu else 8.0
            estimated_size = model_info.estimated_size_at_bits(config.bits)
            
            # Need more headroom for training
            headroom_factor = 3.0 if training else 1.5
            
            if estimated_size * headroom_factor > available_memory:
                config.cpu_offload = True
            
            if training and estimated_size > available_memory * 0.5:
                config.gradient_checkpointing = True
        
        # Speed optimizations
        config.use_flash_attention = hw.supports_flash_attention
        config.use_fused_kernels = hw.supports_bf16  # Requires Ampere+
        
        # Only compile for larger GPUs (compilation overhead)
        if hw.gpus and hw.total_gpu_memory_gb >= 16:
            config.compile_model = True
        
        # Batch size
        config.batch_size = cls._optimal_batch_size(hw, model_info, config.bits, training)
        
        # Gradient accumulation to reach effective batch of 32
        target_effective_batch = 32
        config.gradient_accumulation_steps = max(1, target_effective_batch // config.batch_size)
        
        # Sequence length
        if max_seq_length is not None:
            config.max_seq_length = max_seq_length
        else:
            config.max_seq_length = min(model_info.max_position_embeddings, 8192)
        
        return config
    
    @staticmethod
    def _choose_bits(hw: HardwareProfile, model_info: ModelInfo, training: bool) -> int:
        """Intelligently choose bit-width based on constraints."""
        if not hw.gpus:
            return 4  # Default for CPU
        
        available_memory = hw.best_gpu.memory_free_gb if hw.best_gpu else 8.0
        
        # Reserve memory for activations
        reserve_factor = 3.0 if training else 1.5
        
        # Try from highest quality to lowest
        for bits in [8, 6, 5, 4, 3, 2]:
            estimated_size = model_info.estimated_size_at_bits(bits)
            if estimated_size * reserve_factor <= available_memory * 0.9:
                return bits
        
        return 2  # Minimum viable
    
    @staticmethod
    def _choose_quant_type(bits: int, hw: HardwareProfile) -> str:
        """Choose optimal GGUF quantization type."""
        # GGUF type naming convention: Q{bits}_{variant}
        type_map = {
            8: "Q8_0",
            6: "Q6_K",
            5: "Q5_K_M",
            4: "Q4_K_M",
            3: "Q3_K_M",
            2: "Q2_K",
        }
        return type_map.get(bits, "Q4_K_M")
    
    @staticmethod
    def _choose_group_size(hw: HardwareProfile, model_info: ModelInfo) -> int:
        """Choose optimal group size for quantization."""
        # Smaller group = better accuracy, worse speed
        # Larger group = worse accuracy, better speed
        
        if model_info.num_params > 30_000_000_000:  # >30B
            return 64  # Higher accuracy for large models
        elif model_info.num_params > 10_000_000_000:  # >10B
            return 128  # Balanced
        else:
            return 128  # Standard
    
    @staticmethod
    def _optimal_batch_size(
        hw: HardwareProfile,
        model_info: ModelInfo,
        bits: int,
        training: bool
    ) -> int:
        """Determine optimal batch size for available memory."""
        if not hw.gpus:
            return 1  # CPU is memory-limited
        
        available_memory = hw.best_gpu.memory_free_gb if hw.best_gpu else 8.0
        model_size = model_info.estimated_size_at_bits(bits)
        
        # Rough estimation:
        # - Per sample activation memory scales with sequence length
        # - Training needs ~3x memory vs inference (gradients + optimizer states)
        
        per_sample_mb = model_info.max_position_embeddings * model_info.hidden_size * 4 / (1024**2)
        
        if training:
            # Need memory for: model + activations + gradients + optimizer (2x)
            free_for_batch = (available_memory - model_size * 2) * 1024  # MB
        else:
            # Just model + activations
            free_for_batch = (available_memory - model_size) * 1024 * 0.8  # MB
        
        # Estimate max batch
        max_batch = max(1, int(free_for_batch / per_sample_mb))
        
        # Clamp to reasonable values
        return min(max_batch, 32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "bits": self.bits,
            "quant_type": self.quant_type,
            "group_size": self.group_size,
            "max_memory": self.max_memory,
            "cpu_offload": self.cpu_offload,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_flash_attention": self.use_flash_attention,
            "use_fused_kernels": self.use_fused_kernels,
            "compile_model": self.compile_model,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "max_seq_length": self.max_seq_length,
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary of the configuration."""
        print("\n" + "=" * 50)
        print(" QUANTLLM CONFIGURATION ".center(50, "="))
        print("=" * 50)
        
        print(f"\n📦 Quantization:")
        print(f"   Bits:       {self.bits}")
        print(f"   Type:       {self.quant_type}")
        print(f"   Group Size: {self.group_size}")
        
        print(f"\n💾 Memory:")
        print(f"   CPU Offload:      {'✓' if self.cpu_offload else '✗'}")
        print(f"   Grad Checkpoint:  {'✓' if self.gradient_checkpointing else '✗'}")
        
        print(f"\n⚡ Speed:")
        print(f"   Flash Attention:  {'✓' if self.use_flash_attention else '✗'}")
        print(f"   Fused Kernels:    {'✓' if self.use_fused_kernels else '✗'}")
        print(f"   torch.compile:    {'✓' if self.compile_model else '✗'}")
        
        print(f"\n🎯 Training:")
        print(f"   Batch Size:       {self.batch_size}")
        print(f"   Grad Accum:       {self.gradient_accumulation_steps}")
        print(f"   Effective Batch:  {self.batch_size * self.gradient_accumulation_steps}")
        
        print(f"\n🖥️  Hardware:")
        print(f"   Device:           {self.device}")
        print(f"   Dtype:            {self.dtype}")
        print(f"   Max Seq Length:   {self.max_seq_length}")
        
        print("\n" + "=" * 50)
