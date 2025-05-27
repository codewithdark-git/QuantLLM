"""Memory tracking utilities for QuantLLM."""

import gc
import psutil
import torch
from typing import Optional, Dict, List, Union
from pathlib import Path
import json
import time
from .logger import logger

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False

class MemoryTracker:
    """Enhanced memory tracking for CPU and GPU."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """Initialize memory tracker."""
        self.log_dir = Path(log_dir) if log_dir else Path("memory_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.memory_logs: List[Dict] = []
        self.peak_memory: Dict[str, float] = {
            'cpu': 0.0,
            'gpu': 0.0
        }
        
        if PYNVML_AVAILABLE:
            self.gpu_handles = []
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    self.gpu_handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            except Exception as e:
                logger.log_warning(f"Failed to initialize GPU handles: {e}")
    
    def get_cpu_memory(self) -> float:
        """Get current CPU memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def get_gpu_memory(self, device_index: int = 0) -> Optional[Dict[str, float]]:
        """Get current GPU memory usage in GB."""
        if not PYNVML_AVAILABLE or not self.gpu_handles:
            return None
            
        try:
            handle = self.gpu_handles[device_index]
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                'total': info.total / (1024 ** 3),
                'used': info.used / (1024 ** 3),
                'free': info.free / (1024 ** 3)
            }
        except Exception as e:
            logger.log_warning(f"Failed to get GPU memory info: {e}")
        return None

    def get_torch_memory(self, device: Optional[Union[str, torch.device]] = None) -> Optional[float]:
        """Get PyTorch allocated memory in GB."""
        if device is None and torch.cuda.is_available():
            device = torch.cuda.current_device()
            
        try:
            if isinstance(device, str):
                device = torch.device(device)
                
            if device.type == 'cuda':
                return torch.cuda.memory_allocated(device) / (1024 ** 3)
            return None
        except Exception as e:
            logger.log_warning(f"Failed to get PyTorch memory info: {e}")
            return None
    
    def log_memory(self, operation: str, extra_info: Optional[Dict] = None):
        """Log current memory usage."""
        timestamp = time.time()
        cpu_mem = self.get_cpu_memory()
        self.peak_memory['cpu'] = max(self.peak_memory['cpu'], cpu_mem)
        
        memory_info = {
            'timestamp': timestamp,
            'operation': operation,
            'cpu_memory_gb': cpu_mem,
            'peak_cpu_memory_gb': self.peak_memory['cpu']
        }
        
        if PYNVML_AVAILABLE:
            for i, handle in enumerate(self.gpu_handles):
                gpu_mem = self.get_gpu_memory(i)
                if gpu_mem:
                    memory_info[f'gpu{i}_memory_gb'] = gpu_mem
                    self.peak_memory['gpu'] = max(
                        self.peak_memory['gpu'],
                        gpu_mem['used']
                    )
                    memory_info[f'peak_gpu{i}_memory_gb'] = self.peak_memory['gpu']
                    
                    # Get GPU utilization
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory_info[f'gpu{i}_utilization'] = {
                            'gpu': util.gpu,
                            'memory': util.memory
                        }
                    except Exception as e:
                        logger.log_warning(f"Failed to get GPU utilization: {e}")
        
        # Add PyTorch specific memory info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch_mem = self.get_torch_memory(f'cuda:{i}')
                if torch_mem is not None:
                    memory_info[f'torch_gpu{i}_allocated_gb'] = torch_mem
        
        if extra_info:
            memory_info.update(extra_info)
        
        self.memory_logs.append(memory_info)
        logger.log_memory(
            operation,
            memory_info['cpu_memory_gb'],
            'cpu'
        )
        
        # Save to file
        self._save_logs()
    
    def _save_logs(self):
        """Save memory logs to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"memory_log_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.memory_logs, f, indent=2)
    
    def clear_memory(self):
        """Clear memory and caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
    
    def get_summary(self) -> Dict:
        """Get memory usage summary."""
        if not self.memory_logs:
            return {}
            
        summary = {
            'peak_cpu_memory_gb': self.peak_memory['cpu'],
            'peak_gpu_memory_gb': self.peak_memory['gpu'],
            'num_operations': len(self.memory_logs),
            'total_duration': self.memory_logs[-1]['timestamp'] - self.memory_logs[0]['timestamp']
        }
        
        # Calculate average memory usage
        cpu_memories = [log['cpu_memory_gb'] for log in self.memory_logs]
        summary['avg_cpu_memory_gb'] = sum(cpu_memories) / len(cpu_memories)
        
        if PYNVML_AVAILABLE and self.gpu_handles:
            gpu_memories = []
            for i in range(len(self.gpu_handles)):
                gpu_key = f'gpu{i}_memory_gb'
                gpu_mems = [
                    log[gpu_key]['used']
                    for log in self.memory_logs
                    if gpu_key in log
                ]
                if gpu_mems:
                    summary[f'avg_gpu{i}_memory_gb'] = sum(gpu_mems) / len(gpu_mems)
                    gpu_memories.extend(gpu_mems)
            
            if gpu_memories:
                summary['avg_gpu_memory_gb'] = sum(gpu_memories) / len(gpu_memories)
        
        return summary
    
    def __del__(self):
        """Cleanup NVML on deletion."""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.log_warning(f"Failed to shutdown NVML: {e}")

# Global memory tracker instance
memory_tracker = MemoryTracker()
