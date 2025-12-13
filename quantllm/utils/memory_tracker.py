"""Memory tracking utilities for profiling and debugging."""

import gc
import time
from .progress import logger, print_warning
from typing import Dict, Any, Optional
import torch


class MemoryTracker:
    """
    Track memory usage across different stages of computation.
    Useful for debugging memory leaks and optimizing memory usage.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize memory tracker.
        
        Args:
            device: Target device to track (defaults to cuda:0 if available)
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.memory_logs: Dict[str, Dict[str, Any]] = {}
        self.start_time: Optional[float] = None
        self._is_tracking = False
    
    def start_tracking(self) -> None:
        """Start memory tracking session."""
        self._is_tracking = True
        self.start_time = time.time()
        self.memory_logs = {}
        
        # Force garbage collection before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        
        self.log_memory("initial_state")
    
    def stop_tracking(self) -> None:
        """Stop memory tracking session."""
        self.log_memory("final_state")
        self._is_tracking = False
    
    def log_memory(self, tag: str) -> Dict[str, Any]:
        """
        Log current memory state with a descriptive tag.
        
        Args:
            tag: Descriptive name for this memory checkpoint
            
        Returns:
            Dictionary containing memory statistics
        """
        timestamp = time.time() - (self.start_time or time.time())
        
        memory_info = {
            "timestamp_s": timestamp,
            "cpu_memory": self._get_cpu_memory(),
            "gpu_memory": self._get_gpu_memory() if torch.cuda.is_available() else {}
        }
        
        self.memory_logs[tag] = memory_info
        return memory_info
    
    def _get_cpu_memory(self) -> Dict[str, float]:
        """Get current CPU memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_gb": memory_info.rss / (1024**3),
                "vms_gb": memory_info.vms / (1024**3),
            }
        except ImportError:
            return {}
    
    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {}
        
        try:
            device_idx = self.device.index if self.device.index is not None else 0
            return {
                "gpu_allocated_current_gb": torch.cuda.memory_allocated(device_idx) / (1024**3),
                "gpu_allocated_peak_gb": torch.cuda.max_memory_allocated(device_idx) / (1024**3),
                "gpu_reserved_current_gb": torch.cuda.memory_reserved(device_idx) / (1024**3),
                "gpu_reserved_peak_gb": torch.cuda.max_memory_reserved(device_idx) / (1024**3),
            }
        except Exception:
            return {}
    
    def get_memory_delta(self, tag1: str, tag2: str) -> Dict[str, float]:
        """
        Calculate memory difference between two checkpoints.
        
        Args:
            tag1: First checkpoint tag
            tag2: Second checkpoint tag
            
        Returns:
            Dictionary with memory differences
        """
        if tag1 not in self.memory_logs or tag2 not in self.memory_logs:
            return {}
        
        log1 = self.memory_logs[tag1]
        log2 = self.memory_logs[tag2]
        
        delta = {}
        
        # GPU memory delta
        if "gpu_memory" in log1 and "gpu_memory" in log2:
            for key in log1["gpu_memory"]:
                if key in log2["gpu_memory"]:
                    delta[f"delta_{key}"] = log2["gpu_memory"][key] - log1["gpu_memory"][key]
        
        # Time delta
        delta["elapsed_s"] = log2["timestamp_s"] - log1["timestamp_s"]
        
        return delta
    
    def print_report(self) -> None:
        """Print a formatted memory report."""
        print("\n" + "=" * 60)
        print(" MEMORY TRACKING REPORT ".center(60, "="))
        print("=" * 60)
        
        for tag, data in self.memory_logs.items():
            print(f"\n📍 {tag} (t={data['timestamp_s']:.2f}s)")
            print("-" * 40)
            
            if data.get("gpu_memory"):
                gpu = data["gpu_memory"]
                print(f"  GPU Allocated: {gpu.get('gpu_allocated_current_gb', 0):.3f} GB")
                print(f"  GPU Peak:      {gpu.get('gpu_allocated_peak_gb', 0):.3f} GB")
                print(f"  GPU Reserved:  {gpu.get('gpu_reserved_current_gb', 0):.3f} GB")
            
            if data.get("cpu_memory"):
                cpu = data["cpu_memory"]
                print(f"  CPU RSS:       {cpu.get('rss_gb', 0):.3f} GB")
        
        print("\n" + "=" * 60)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the memory tracking session."""
        if not self.memory_logs:
            return {}
        
        summary = {
            "num_checkpoints": len(self.memory_logs),
            "tags": list(self.memory_logs.keys()),
        }
        
        # Get peak GPU memory across all checkpoints
        if torch.cuda.is_available():
            peak_gpu = 0
            for data in self.memory_logs.values():
                if "gpu_memory" in data:
                    peak_gpu = max(peak_gpu, data["gpu_memory"].get("gpu_allocated_peak_gb", 0))
            summary["peak_gpu_gb"] = peak_gpu
        
        return summary


def track_memory(func):
    """Decorator to track memory usage of a function."""
    def wrapper(*args, **kwargs):
        tracker = MemoryTracker()
        tracker.start_tracking()
        tracker.log_memory("before_function")
        
        try:
            result = func(*args, **kwargs)
            tracker.log_memory("after_function")
            return result
        finally:
            tracker.stop_tracking()
            tracker.print_report()
    
    return wrapper
