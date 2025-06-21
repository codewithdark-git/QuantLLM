"""Enhanced high-level API for GGUF model quantization with improved UX and performance."""

from typing import Optional, Dict, Any, Union, Tuple, Callable, List
import torch
import time
import threading
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from ..quant.gguf import GGUFQuantizer, SUPPORTED_GGUF_BITS, SUPPORTED_GGUF_TYPES
from ..quant.llama_cpp_utils import LlamaCppConverter
from ..utils.logger import logger
import psutil
import math
import os
import json
from pathlib import Path
import tempfile
import shutil
import glob

class SystemResourceMonitor:
    """Monitor system resources during quantization."""
    
    def __init__(self):
        self.gpu_info = self._get_gpu_info()
        self.cpu_info = self._get_cpu_info()
        self.memory_info = self._get_memory_info()
        
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information."""
        gpu_info = {"available": False, "devices": []}
        
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)
                allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
                free_mem = total_mem - allocated_mem
                
                gpu_info["devices"].append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": total_mem,
                    "allocated_memory_gb": allocated_mem,
                    "free_memory_gb": free_mem,
                    "compute_capability": f"{props.major}.{props.minor}"
                })
        
        return gpu_info
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        return {
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get system memory information."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percentage": mem.percent
        }
    
    def get_optimal_config(self, model_size_gb: float) -> Dict[str, Any]:
        """Get optimal configuration based on available resources."""
        config = {
            "device": "cpu",
            "device_map": "cpu",
            "load_in_4bit": False,
            "cpu_offload": True,
            "max_memory": None,
            "optimization_level": "balanced"
        }
        
        if self.gpu_info["available"]:
            max_gpu_mem = max(gpu["free_memory_gb"] for gpu in self.gpu_info["devices"])
            total_gpu_mem = sum(gpu["free_memory_gb"] for gpu in self.gpu_info["devices"])
            
            if model_size_gb <= max_gpu_mem * 0.8:
                config.update({
                    "device": "cuda",
                    "device_map": "auto",
                    "cpu_offload": False,
                    "optimization_level": "quality"
                })
            elif model_size_gb <= total_gpu_mem * 0.8:
                config.update({
                    "device": "cuda",
                    "device_map": "auto",
                    "cpu_offload": False,
                    "max_memory": {
                        i: f"{int(gpu['free_memory_gb'] * 0.8)}GB"
                        for i, gpu in enumerate(self.gpu_info["devices"])
                    },
                    "optimization_level": "balanced"
                })
            elif model_size_gb <= (total_gpu_mem + self.memory_info["available_gb"]) * 0.6:
                config.update({
                    "device": "cuda",
                    "device_map": "auto",
                    "cpu_offload": True,
                    "max_memory": {
                        **{
                            i: f"{int(gpu['free_memory_gb'] * 0.7)}GB"
                            for i, gpu in enumerate(self.gpu_info["devices"])
                        },
                        "cpu": f"{int(self.memory_info['available_gb'] * 0.5)}GB"
                    }
                })
            elif model_size_gb > total_gpu_mem * 0.5:
                config["cpu_offload"] = True
                config["max_memory"] = {
                    **{i: f"{int(gpu['free_memory_gb'] * 0.5)}GB" for i, gpu in enumerate(self.gpu_info["devices"])},
                    "cpu": f"{int(self.memory_info['available_gb'] * 0.7)}GB"
                }
        
        if self.cpu_info["cores_physical"] >= 8:
            config["optimization_level"] = "quality"
        elif self.cpu_info["cores_physical"] >= 4:
            config["optimization_level"] = "balanced"
        else:
            config["optimization_level"] = "fast"
        
        return config

def estimate_model_size(model_name: Union[str, PreTrainedModel]) -> Dict[str, float]:
    """Enhanced model size estimation with detailed breakdown."""
    try:
        if isinstance(model_name, PreTrainedModel):
            params = sum(p.numel() for p in model_name.parameters())
            model_size_fp16 = (params * 2) / (1024**3)
            
            embedding_params = sum(p.numel() for n, p in model_name.named_parameters() if 'embed' in n.lower())
            attention_params = sum(p.numel() for n, p in model_name.named_parameters() if any(x in n.lower() for x in ['attn', 'attention']))
            
            return {
                "total_params": params,
                "size_fp16_gb": model_size_fp16,
                "size_fp32_gb": model_size_fp16 * 2,
                "embedding_params": embedding_params,
                "attention_params": attention_params,
                "other_params": params - embedding_params - attention_params
            }
        else:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            return _estimate_from_config(config, model_name)
            
    except Exception as e:
        logger.log_warning(f"Error estimating model size: {e}. Using fallback estimation.")
        return _fallback_size_estimation(model_name)

def _estimate_from_config(config, model_name: str) -> Dict[str, float]:
    """Estimate model size from configuration."""
    if hasattr(config, 'num_parameters'):
        params = config.num_parameters
    elif hasattr(config, 'n_params'):
        params = config.n_params
    elif hasattr(config, 'hidden_size') and hasattr(config, 'num_hidden_layers'):
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        vocab_size = getattr(config, 'vocab_size', 32000)
        
        if hasattr(config, 'intermediate_size'):
            ffn_size = config.intermediate_size
        else:
            ffn_size = hidden_size * 4
        
        attention_params = 4 * num_layers * hidden_size * hidden_size
        ffn_params = 2 * num_layers * hidden_size * ffn_size
        embedding_params = vocab_size * hidden_size * 2
        norm_params = 2 * num_layers * hidden_size
        
        params = attention_params + ffn_params + embedding_params + norm_params
    else:
        params = _fallback_param_estimation(model_name)
    
    size_fp16 = (params * 2) / (1024**3)
    
    return {
        "total_params": params,
        "size_fp16_gb": size_fp16,
        "size_fp32_gb": size_fp16 * 2,
        "embedding_params": embedding_params if 'embedding_params' in locals() else 0,
        "attention_params": attention_params if 'attention_params' in locals() else 0,
        "other_params": params - (embedding_params + attention_params) if 'embedding_params' in locals() else params
    }

def _fallback_size_estimation(model_name: str) -> Dict[str, float]:
    """Fallback size estimation based on model name patterns."""
    model_name_lower = str(model_name).lower()
    
    size_patterns = {
        "7b": 13.0, "8b": 15.0, "3b": 6.0, "1b": 2.0,
        "13b": 24.0, "15b": 28.0, "20b": 38.0,
        "30b": 58.0, "34b": 65.0, "40b": 75.0,
        "65b": 120.0, "70b": 130.0, "72b": 135.0,
        "175b": 350.0, "180b": 360.0
    }
    
    for pattern, size in size_patterns.items():
        if pattern in model_name_lower:
            params = int(pattern.replace('b', '')) * 1e9
            return {
                "total_params": params,
                "size_fp16_gb": size,
                "size_fp32_gb": size * 2,
                "embedding_params": 0,
                "attention_params": 0,
                "other_params": params
            }
    
    default_size = 7.0
    default_params = 7e9
    return {
        "total_params": default_params,
        "size_fp16_gb": default_size,
        "size_fp32_gb": default_size * 2,
        "embedding_params": 0,
        "attention_params": 0,
        "other_params": default_params
    }

def _fallback_param_estimation(model_name: str) -> int:
    """Fallback parameter estimation based on model name patterns."""
    model_name_lower = str(model_name).lower()
    
    param_patterns = {
        "7b": 7e9, "8b": 8e9, "3b": 3e9, "1b": 1e9,
        "13b": 13e9, "15b": 15e9, "20b": 20e9,
        "30b": 30e9, "34b": 34e9, "40b": 40e9,
        "65b": 65e9, "70b": 70e9, "72b": 72e9,
        "175b": 175e9, "180b": 180e9
    }
    
    for pattern, params in param_patterns.items():
        if pattern in model_name_lower:
            return int(params)
    
    return int(7e9)

class ProgressTracker:
    """Track quantization progress with detailed metrics."""
    
    def __init__(self):
        self.start_time = None
        self.phase_times = {}
        self.current_phase = None
        self.total_steps = 0
        self.completed_steps = 0
        
    def start(self, total_steps: int = 100):
        """Start progress tracking."""
        self.start_time = time.time()
        self.total_steps = total_steps
        self.completed_steps = 0
        logger.log_info("🚀 Starting quantization process...")
        
    def start_phase(self, phase_name: str):
        """Start a new phase."""
        if self.current_phase:
            self.end_phase()
        
        self.current_phase = phase_name
        self.phase_times[phase_name] = {"start": time.time(), "end": None}
        logger.log_info(f"📋 Phase: {phase_name}")
        
    def end_phase(self):
        """End current phase."""
        if self.current_phase and self.current_phase in self.phase_times:
            self.phase_times[self.current_phase]["end"] = time.time()
            duration = self.phase_times[self.current_phase]["end"] - self.phase_times[self.current_phase]["start"]
            logger.log_info(f"✓ Completed {self.current_phase} in {duration:.2f}s")
        
    def update(self, steps: int = 1, message: str = None):
        """Update progress."""
        self.completed_steps += steps
        if message:
            logger.log_info(f"  {message}")
        
        progress = (self.completed_steps / self.total_steps) * 100
        if progress % 10 < (steps / self.total_steps) * 100:
            logger.log_info(f"📊 Progress: {progress:.1f}%")
    
    def finish(self):
        """Finish tracking."""
        if self.current_phase:
            self.end_phase()
            
        total_time = time.time() - self.start_time if self.start_time else 0
        logger.log_info(f"🎉 Quantization completed in {total_time:.2f}s")
        
        if self.phase_times:
            logger.log_info("\n⏱️  Phase Breakdown:")
            for phase, times in self.phase_times.items():
                if times["end"]:
                    duration = times["end"] - times["start"]
                    logger.log_info(f"  • {phase}: {duration:.2f}s")

def get_gpu_memory():
    """Get available GPU memory in GB."""
    monitor = SystemResourceMonitor()
    if monitor.gpu_info["available"]:
        return [gpu["free_memory_gb"] for gpu in monitor.gpu_info["devices"]]
    return []

def get_system_memory():
    """Get available system memory in GB."""
    monitor = SystemResourceMonitor()
    return monitor.memory_info["available_gb"]

class QuantLLM:
    """Enhanced high-level API for GGUF model quantization."""
    
    def __init__(self):
        """Initialize QuantLLM with system monitoring."""
        self.system_monitor = SystemResourceMonitor()
        self.progress_tracker = None
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        return {
            "gpu": self.system_monitor.gpu_info,
            "cpu": self.system_monitor.cpu_info,
            "memory": self.system_monitor.memory_info
        }
    
    def get_optimal_config(self, model_size_gb: float) -> Dict[str, Any]:
        """Get optimal configuration based on system resources."""
        return self.system_monitor.get_optimal_config(model_size_gb)
    
    def estimate_model_size(self, model_name: Union[str, PreTrainedModel]) -> Dict[str, float]:
        """Estimate model size and get detailed breakdown."""
        return estimate_model_size(model_name)
    
    def get_recommended_bits(
        self,
        model_size_gb: float,
        target_size_gb: Optional[float] = None,
        priority: str = "balanced"
    ) -> Tuple[int, str]:
        """Get recommended quantization bits and type."""
        return self.get_recommended_quant_type(
            model_size_gb=model_size_gb,
            target_size_gb=target_size_gb,
            priority=priority
        )
    
    def start_progress_tracking(self, total_steps: int = 100):
        """Initialize progress tracking."""
        self.progress_tracker = ProgressTracker()
        self.progress_tracker.start(total_steps)
    
    def update_progress(self, step: int, message: Optional[str] = None):
        """Update progress tracking."""
        if self.progress_tracker:
            self.progress_tracker.update(step, message)
    
    def end_progress_tracking(self):
        """End progress tracking."""
        if self.progress_tracker:
            self.progress_tracker.finish()
            self.progress_tracker = None
    
    def cleanup(self):
        """Clean up resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        if self.progress_tracker:
            self.end_progress_tracking()
    
    @staticmethod
    def list_quant_types(bits: Optional[int] = None) -> Dict[str, str]:
        """
        List available quantization types and their descriptions.
        
        Args:
            bits: Optional bit width to filter quantization types
            
        Returns:
            Dictionary mapping quantization types to their descriptions
        """
        quant_types = {}
        
        if bits is not None:
            if bits not in SUPPORTED_GGUF_BITS:
                raise ValueError(f"Unsupported bits: {bits}. Supported values: {SUPPORTED_GGUF_BITS}")
            types = SUPPORTED_GGUF_TYPES[bits]
            for qtype, config in types.items():
                quant_types[qtype] = config["description"]
        else:
            for bits_val, types in SUPPORTED_GGUF_TYPES.items():
                for qtype, config in types.items():
                    quant_types[f"{qtype} ({bits_val}-bit)"] = config["description"]
        
        return quant_types
    
    @staticmethod
    def get_recommended_quant_type(
        model_size_gb: float,
        target_size_gb: Optional[float] = None,
        priority: str = "balanced"
    ) -> Tuple[int, str]:
        """
        Get recommended quantization type based on model size and requirements.
        
        Args:
            model_size_gb: Original model size in gigabytes
            target_size_gb: Target model size in gigabytes (optional)
            priority: Optimization priority ("speed", "quality", or "balanced")
            
        Returns:
            Tuple of (bits, quant_type)
        """
        if priority not in ["speed", "quality", "balanced"]:
            raise ValueError("Priority must be 'speed', 'quality', or 'balanced'")
        
        if target_size_gb:
            required_ratio = model_size_gb / target_size_gb
            
            if required_ratio <= 2:
                bits, qtype = (8, "Q8_0") if priority == "quality" else (6, "Q6_K")
            elif required_ratio <= 4:
                if priority == "quality":
                    bits, qtype = (5, "Q5_1")
                elif priority == "speed":
                    bits, qtype = (4, "Q4_K_S")
                else:
                    bits, qtype = (4, "Q4_K_M")
            elif required_ratio <= 8:
                if priority == "quality":
                    bits, qtype = (4, "Q4_1")
                elif priority == "speed":
                    bits, qtype = (2, "IQ2_XXS")
                else:
                    bits, qtype = (2, "IQ2_XS")
            else:
                bits, qtype = (2, "Q2_K")
        else:
            if model_size_gb <= 2:
                bits, qtype = (5, "Q5_1") if priority == "quality" else (4, "Q4_K_M")
            elif model_size_gb <= 7:
                bits, qtype = (5, "Q5_1") if priority == "quality" else (4, "Q4_K_M")
            elif model_size_gb <= 13:
                bits, qtype = (4, "Q4_K_M") if priority != "speed" else (4, "Q4_K_S")
            else:
                bits, qtype = (2, "IQ2_XS")
        
        return bits, qtype
    
    @staticmethod
    def analyze_model(model_name: Union[str, PreTrainedModel]) -> Dict[str, Any]:
        """
        Analyze a model and provide optimization recommendations.
        
        Args:
            model_name: Model identifier or instance
            
        Returns:
            Dictionary with model analysis and recommendations
        """
        logger.log_info("🔍 Analyzing model...")
        
        size_info = estimate_model_size(model_name)
        
        monitor = SystemResourceMonitor()
        optimal_config = monitor.get_optimal_config(size_info["size_fp16_gb"])
        
        recommended_bits, recommended_type = QuantLLM.get_recommended_quant_type(
            size_info["size_fp16_gb"],
            priority=optimal_config["optimization_level"]
        )
        
        compression_ratios = {
            8: 2.0, 6: 2.7, 5: 3.2, 4: 4.0, 
            3: 5.3, 2: 8.0
        }
        
        quantized_sizes = {}
        for bits in SUPPORTED_GGUF_BITS:
            ratio = compression_ratios.get(bits, 4.0)
            quantized_sizes[f"{bits}bit"] = size_info["size_fp16_gb"] / ratio
        
        analysis = {
            "model_info": {
                "name": str(model_name),
                "total_parameters": size_info["total_params"],
                "original_size_gb": size_info["size_fp16_gb"],
                "estimated_breakdown": {
                    "embedding_params": size_info.get("embedding_params", 0),
                    "attention_params": size_info.get("attention_params", 0),
                    "other_params": size_info.get("other_params", size_info["total_params"])
                }
            },
            "system_resources": {
                "gpu_available": monitor.gpu_info["available"],
                "gpu_devices": len(monitor.gpu_info.get("devices", [])),
                "total_gpu_memory_gb": sum(gpu["total_memory_gb"] for gpu in monitor.gpu_info.get("devices", [])),
                "free_gpu_memory_gb": sum(gpu["free_memory_gb"] for gpu in monitor.gpu_info.get("devices", [])),
                "cpu_cores": monitor.cpu_info["cores_physical"],
                "system_memory_gb": monitor.memory_info["total_gb"],
                "available_memory_gb": monitor.memory_info["available_gb"]
            },
            "recommendations": {
                "optimal_device": optimal_config["device"],
                "device_map": optimal_config["device_map"],
                "cpu_offload": optimal_config["cpu_offload"],
                "optimization_level": optimal_config["optimization_level"],
                "recommended_bits": recommended_bits,
                "recommended_type": recommended_type,
                "load_in_4bit": optimal_config.get("load_in_4bit", False)
            },
            "quantized_sizes": quantized_sizes,
            "memory_requirements": {
                "minimum_gpu_memory_gb": size_info["size_fp16_gb"] * 1.2,
                "minimum_system_memory_gb": size_info["size_fp16_gb"] * 1.5,
                "recommended_gpu_memory_gb": size_info["size_fp16_gb"] * 1.5,
                "recommended_system_memory_gb": size_info["size_fp16_gb"] * 2.0
            }
        }
        
        return analysis
    
    @staticmethod
    def quantize_from_pretrained(
        model_name: Union[str, PreTrainedModel],
        bits: int = 4,
        group_size: int = 128,
        quant_type: Optional[str] = None,
        use_packed: bool = True,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: torch.dtype = torch.float16,
        bnb_4bit_use_double_quant: bool = True,
        use_gradient_checkpointing: bool = True,
        device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]] = "auto",
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        offload_folder: Optional[str] = None,
        offload_state_dict: bool = False,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        auto_device: bool = True,
        optimize_for: str = "balanced",
        cpu_offload: bool = False,
        verbose: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> PreTrainedModel:
        try:
            progress = ProgressTracker()
            if verbose:
                progress.start(100)
                progress.start_phase("Initialization")
            
            logger.log_info(f"Starting quantization preparation with {bits} bits")
            
            if bits not in SUPPORTED_GGUF_BITS:
                raise ValueError(f"Unsupported bits: {bits}. Supported values: {SUPPORTED_GGUF_BITS}")
            if quant_type and quant_type not in SUPPORTED_GGUF_TYPES.get(bits, {}):
                raise ValueError(f"Unsupported quant_type: {quant_type} for {bits} bits")
                
            if verbose:
                progress.update(10, "Analyzing model and system resources...")
            
            size_info = estimate_model_size(model_name)
            model_size_gb = size_info["size_fp16_gb"]
            
            monitor = SystemResourceMonitor()
            gpu_mem = [gpu["free_memory_gb"] for gpu in monitor.gpu_info.get("devices", [])]
            system_mem = monitor.memory_info["available_gb"]
            
            if verbose:
                logger.log_info(f"📊 Model Analysis:")
                logger.log_info(f"  • Parameters: {size_info['total_params']:,}")
                logger.log_info(f"  • Size (FP16): {model_size_gb:.2f} GB")
                logger.log_info(f"  • Available GPU memory: {gpu_mem}")
                logger.log_info(f"  • Available system memory: {system_mem:.2f} GB")
                
                progress.update(10, "Resource analysis completed")
                progress.end_phase()
                progress.start_phase("Configuration")
            
            if auto_device:
                optimal_config = monitor.get_optimal_config(model_size_gb)
                
                if device is None:
                    device = optimal_config["device"]
                if device_map == "auto":
                    device_map = optimal_config["device_map"]
                if max_memory is None:
                    max_memory = optimal_config.get("max_memory")
                if not cpu_offload:
                    cpu_offload = optimal_config["cpu_offload"]
                    
                if verbose:
                    logger.log_info(f"🔧 Auto-configuration:")
                    logger.log_info(f"  • Device: {device}")
                    logger.log_info(f"  • Device map: {device_map}")
                    logger.log_info(f"  • CPU offload: {cpu_offload}")
                    logger.log_info(f"  • Optimization level: {optimal_config['optimization_level']}")
            
            bnb_config = None
            if bits in [4, 8]:
                compute_dtype = bnb_4bit_compute_dtype or torch.float16
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=bits == 4,
                    load_in_8bit=bits == 8,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    llm_int8_enable_fp32_cpu_offload=cpu_offload
                )
            
            if not quant_type:
                _, quant_type = QuantLLM.get_recommended_quant_type(
                    model_size_gb=model_size_gb,
                    priority=optimize_for
                )
                if verbose:
                    logger.log_info(f"📋 Selected quantization: {quant_type} ({bits}-bit)")
            
            if verbose:
                progress.update(20, "Configuration completed")
                progress.end_phase()
                progress.start_phase("Model Loading")
            
            if progress_callback:
                progress_callback(30, "Creating quantizer...")
            
            quantizer = GGUFQuantizer(
                model_name=model_name,
                bits=bits,
                group_size=group_size,
                quant_type=quant_type,
                use_packed=use_packed,
                device=device,
                quantization_config=bnb_config,
                use_gradient_checkpointing=use_gradient_checkpointing,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                torch_dtype=torch_dtype,
                cpu_offload=cpu_offload
            )
            
            if verbose:
                progress.update(40, "Quantizer created")
                progress.end_phase()
                progress.finish()
            
            if progress_callback:
                progress_callback(100, "Model preparation completed successfully!")
            
            quantizer.model._quantizer = quantizer
            quantizer.model.config.quantization_config = {
                "bits": bits,
                "quant_type": quant_type,
                "group_size": group_size
            }
            
            return quantizer.model
            
        except Exception as e:
            logger.log_error(f"Quantization preparation failed: {str(e)}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    @staticmethod
    def save_quantized_model(
        model: PreTrainedModel,
        output_path: str,
        save_format: str = "safetensors",
        save_tokenizer: bool = True,
        quant_config: Optional[Dict[str, Any]] = None,
        safe_serialization: bool = True,
        verbose: bool = False,
        progress_callback: Optional[Callable] = None,
        replace_original: bool = True
    ):
        """
        Save a quantized model in either GGUF or safetensors format.
        
        Args:
            model: The quantized model to save
            output_path: Path to save the model
            save_format: Format to save in ("gguf" or "safetensors")
            save_tokenizer: Whether to save the tokenizer
            quant_config: Optional quantization configuration
            safe_serialization: Whether to use safe serialization
            verbose: Whether to show detailed progress
            progress_callback: Optional callback for progress updates
            replace_original: Whether to replace original model files with quantized ones
        """
        try:
            progress = ProgressTracker()
            if verbose:
                progress.start(100)
                progress.start_phase("Initialization")
            
            if progress_callback:
                progress_callback(0, "Starting model export...")
            
            original_path = None
            if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
                from huggingface_hub import HfFolder
                cache_dir = os.getenv('HF_HOME', HfFolder.default_cache_path)
                model_id = model.config._name_or_path
                if '/' in model_id:
                    org, model_name = model_id.split('/')
                    potential_paths = glob.glob(os.path.join(cache_dir, 'models--' + org + '--' + model_name, '*', 'snapshots', '*'))
                    if potential_paths:
                        original_path = potential_paths[0]
                        if verbose:
                            logger.log_info(f"Found original model in cache: {original_path}")

            if output_path == "auto" and original_path:
                output_path = original_path
            else:
                output_path = os.path.abspath(output_path)
                
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if verbose:
                logger.log_info(f"Saving model to: {output_path}")
                
            with tempfile.TemporaryDirectory(prefix="quant_save_") as temp_dir:
                if save_format.lower() == "gguf":
                    if verbose:
                        progress.start_phase("GGUF Conversion")
                    
                    from ..quant.llama_cpp_utils import LlamaCppConverter
                    converter = LlamaCppConverter(verbose=verbose)
                    
                    if progress_callback:
                        progress_callback(30, "Converting to GGUF format...")
                    
                    if not output_path.lower().endswith('.gguf'):
                        output_path = f"{output_path}.gguf"
                    
                    gguf_path = converter.convert_to_gguf(
                        model=model,
                        output_dir=os.path.dirname(output_path),
                        bits=quant_config.get('bits', 4) if quant_config else 4,
                        group_size=quant_config.get('group_size', 128) if quant_config else 128,
                        save_tokenizer=save_tokenizer,
                        custom_name=os.path.basename(output_path),
                        quant_type=quant_config.get('quant_type', 'q4_k_m') if quant_config else None
                    )
                    
                    if verbose:
                        file_size = os.path.getsize(gguf_path) / (1024**3)
                        logger.log_info(f"\n✅ GGUF model saved ({file_size:.2f} GB): {gguf_path}")
                    
                else:
                    if verbose:
                        progress.start_phase("Safetensors Export")
                    
                    if progress_callback:
                        progress_callback(30, "Saving in safetensors format...")
                    
                    model.save_pretrained(
                        temp_dir,
                        safe_serialization=safe_serialization,
                        max_shard_size="2GB"
                    )
                    
                    if save_tokenizer and hasattr(model, 'tokenizer'):
                        model.tokenizer.save_pretrained(temp_dir)
                    
                    if replace_original and original_path:
                        target_dir = original_path
                        if verbose:
                            logger.log_info(f"Replacing original files in: {target_dir}")
                        
                        for file in os.listdir(target_dir):
                            if file.endswith(('.bin', '.safetensors', '.pt', '.gguf')):
                                os.remove(os.path.join(target_dir, file))
                        
                        for file in os.listdir(temp_dir):
                            src = os.path.join(temp_dir, file)
                            dst = os.path.join(target_dir, file)
                            if os.path.exists(dst):
                                os.remove(dst)
                            shutil.copy2(src, dst)
                            
                        if verbose:
                            logger.log_info("✅ Original model files replaced with quantized versions")
                    
                    else:
                        target_dir = output_path if os.path.isdir(output_path) else os.path.dirname(output_path)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        for file in os.listdir(temp_dir):
                            src = os.path.join(temp_dir, file)
                            dst = os.path.join(target_dir, file)
                            if os.path.exists(dst):
                                os.remove(dst)
                            shutil.copy2(src, dst)
                    
                    if verbose:
                        total_size = sum(
                            os.path.getsize(os.path.join(target_dir, f)) / (1024**3)
                            for f in os.listdir(target_dir)
                            if f.endswith('.safetensors')
                        )
                        logger.log_info(f"\n✅ Model saved in safetensors format ({total_size:.2f} GB)")
                        logger.log_info(f"📁 Output directory: {target_dir}")
            
            if verbose:
                progress.end_phase()
                progress.finish()
            
            if progress_callback:
                progress_callback(100, "Model export completed successfully!")
            
        except Exception as e:
            logger.log_error(f"Failed to save model: {str(e)}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
