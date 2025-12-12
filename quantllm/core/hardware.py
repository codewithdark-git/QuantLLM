"""
Hardware Profiler for QuantLLM.

Automatically detects and profiles available hardware to enable
smart default configurations.
"""

import os
import platform
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    memory_total_gb: float
    memory_free_gb: float
    compute_capability: tuple
    supports_bf16: bool
    supports_flash_attention: bool
    cuda_cores: int = 0


@dataclass
class HardwareProfile:
    """Complete hardware profile of the system."""
    # CPU Info
    cpu_count: int
    cpu_name: str
    system_memory_gb: float
    
    # GPU Info
    num_gpus: int
    gpus: List[GPUInfo]
    cuda_available: bool
    cuda_version: Optional[str]
    
    # Capabilities
    supports_bf16: bool
    supports_flash_attention: bool
    best_gpu_index: int
    total_gpu_memory_gb: float
    
    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        """Get the best GPU available."""
        if self.gpus:
            return self.gpus[self.best_gpu_index]
        return None
    
    @property
    def device(self) -> torch.device:
        """Get the best device to use."""
        if self.cuda_available and self.gpus:
            return torch.device(f"cuda:{self.best_gpu_index}")
        return torch.device("cpu")


class HardwareProfiler:
    """
    Automatically detect and profile system hardware.
    
    This enables smart auto-configuration of quantization parameters,
    batch sizes, and optimization strategies.
    
    Example:
        >>> profiler = HardwareProfiler()
        >>> profile = profiler.detect()
        >>> print(f"Best GPU: {profile.best_gpu.name} with {profile.best_gpu.memory_total_gb:.1f}GB")
    """
    
    _cached_profile: Optional[HardwareProfile] = None
    
    @classmethod
    def detect(cls, force_refresh: bool = False) -> HardwareProfile:
        """
        Detect and profile all available hardware.
        
        Args:
            force_refresh: If True, re-detect even if cached
            
        Returns:
            HardwareProfile containing all detected hardware info
        """
        if cls._cached_profile is not None and not force_refresh:
            return cls._cached_profile
        
        # Detect CPU info
        cpu_count = os.cpu_count() or 1
        cpu_name = cls._get_cpu_name()
        system_memory_gb = cls._get_system_memory()
        
        # Detect GPU info
        cuda_available = torch.cuda.is_available()
        cuda_version = None
        gpus = []
        num_gpus = 0
        
        if cuda_available:
            cuda_version = torch.version.cuda
            num_gpus = torch.cuda.device_count()
            
            for i in range(num_gpus):
                gpu_info = cls._get_gpu_info(i)
                if gpu_info:
                    gpus.append(gpu_info)
        
        # Determine best GPU (most memory)
        best_gpu_index = 0
        if gpus:
            best_gpu_index = max(range(len(gpus)), key=lambda i: gpus[i].memory_total_gb)
        
        # Aggregate capabilities
        supports_bf16 = any(gpu.supports_bf16 for gpu in gpus) if gpus else False
        supports_flash_attention = any(gpu.supports_flash_attention for gpu in gpus) if gpus else False
        total_gpu_memory = sum(gpu.memory_total_gb for gpu in gpus)
        
        profile = HardwareProfile(
            cpu_count=cpu_count,
            cpu_name=cpu_name,
            system_memory_gb=system_memory_gb,
            num_gpus=num_gpus,
            gpus=gpus,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            supports_bf16=supports_bf16,
            supports_flash_attention=supports_flash_attention,
            best_gpu_index=best_gpu_index,
            total_gpu_memory_gb=total_gpu_memory,
        )
        
        cls._cached_profile = profile
        return profile
    
    @staticmethod
    def _get_cpu_name() -> str:
        """Get the CPU model name."""
        try:
            if platform.system() == "Windows":
                import subprocess
                output = subprocess.check_output(
                    ["wmic", "cpu", "get", "name"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                lines = [l.strip() for l in output.split('\n') if l.strip() and l.strip() != 'Name']
                return lines[0] if lines else "Unknown CPU"
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif platform.system() == "Darwin":  # macOS
                import subprocess
                output = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                return output
        except Exception:
            pass
        return platform.processor() or "Unknown CPU"
    
    @staticmethod
    def _get_system_memory() -> float:
        """Get total system RAM in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback for systems without psutil
            return 16.0  # Assume 16GB as default
    
    @staticmethod
    def _get_gpu_info(index: int) -> Optional[GPUInfo]:
        """Get detailed info about a specific GPU."""
        try:
            props = torch.cuda.get_device_properties(index)
            
            # Get memory info
            memory_total = props.total_memory / (1024**3)
            memory_free = memory_total  # Approximation when not in use
            
            try:
                # Try to get actual free memory
                torch.cuda.set_device(index)
                memory_free = (props.total_memory - torch.cuda.memory_allocated(index)) / (1024**3)
            except Exception:
                pass
            
            # Compute capability
            compute_cap = (props.major, props.minor)
            
            # BF16 support: Ampere (8.0) and above
            supports_bf16 = compute_cap >= (8, 0)
            
            # Flash Attention support: Ampere (8.0) and above with SM80+
            supports_flash_attention = compute_cap >= (8, 0)
            
            return GPUInfo(
                index=index,
                name=props.name,
                memory_total_gb=memory_total,
                memory_free_gb=memory_free,
                compute_capability=compute_cap,
                supports_bf16=supports_bf16,
                supports_flash_attention=supports_flash_attention,
                cuda_cores=props.multi_processor_count * 128,  # Approximate
            )
        except Exception:
            return None
    
    @classmethod
    def print_summary(cls) -> None:
        """Print a formatted summary of hardware capabilities."""
        profile = cls.detect()
        
        print("\n" + "=" * 60)
        print(" HARDWARE PROFILE ".center(60, "="))
        print("=" * 60)
        
        print(f"\n💻 CPU: {profile.cpu_name}")
        print(f"   Cores: {profile.cpu_count}")
        print(f"   System RAM: {profile.system_memory_gb:.1f} GB")
        
        if profile.cuda_available:
            print(f"\n🎮 CUDA: {profile.cuda_version}")
            print(f"   GPUs: {profile.num_gpus}")
            
            for gpu in profile.gpus:
                marker = "★" if gpu.index == profile.best_gpu_index else " "
                print(f"\n   {marker} GPU {gpu.index}: {gpu.name}")
                print(f"     Memory: {gpu.memory_total_gb:.1f} GB")
                print(f"     Compute: SM{gpu.compute_capability[0]}{gpu.compute_capability[1]}")
                print(f"     BF16: {'✓' if gpu.supports_bf16 else '✗'}")
                print(f"     Flash Attn: {'✓' if gpu.supports_flash_attention else '✗'}")
        else:
            print("\n⚠️  No CUDA GPUs detected - will use CPU")
        
        print("\n" + "=" * 60)
    
    @classmethod
    def get_optimal_dtype(cls) -> torch.dtype:
        """Get the optimal dtype for this hardware."""
        profile = cls.detect()
        if profile.supports_bf16:
            return torch.bfloat16
        elif profile.cuda_available:
            return torch.float16
        else:
            return torch.float32
    
    @classmethod
    def estimate_max_model_size_gb(cls, bits: int = 4) -> float:
        """
        Estimate maximum model size that can fit in GPU memory.
        
        Args:
            bits: Quantization bit-width
            
        Returns:
            Maximum model size in GB
        """
        profile = cls.detect()
        if not profile.gpus:
            return profile.system_memory_gb * 0.5  # Use half of system RAM
        
        best_gpu = profile.best_gpu
        if best_gpu is None:
            return 8.0  # Safe default
        
        # Reserve ~20% for activations and overhead
        available = best_gpu.memory_free_gb * 0.8
        
        # Model size = params * bytes_per_param
        # For 4-bit: 0.5 bytes per param
        bytes_per_param = bits / 8
        
        # Approximate: 1GB of 4-bit weights ≈ 2B params
        return available / bytes_per_param
