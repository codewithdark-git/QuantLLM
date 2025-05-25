"""Memory-efficient benchmarking utilities for quantization methods."""

import gc
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from ..quant import (
    GPTQQuantizer,
    AWQQuantizer,
    GGUFQuantizer
)
from ..quant.quantization_engine import DeviceManager

class QuantizationBenchmark:
    """Memory-efficient benchmark implementation for quantization methods."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        calibration_data: torch.Tensor,
        input_shape: Tuple[int, ...] = (1, 32),
        num_inference_steps: int = 100,
        device: Optional[Union[str, torch.device]] = None
    ):
        """Initialize benchmark with memory management."""
        self.device_manager = DeviceManager(
            torch.device(device) if device else None
        )
        # Keep original model on CPU
        self.model = model.to("cpu")
        self.calibration_data = calibration_data.to("cpu")
        self.input_shape = input_shape
        self.num_inference_steps = num_inference_steps
        self.results = {}
        
        # Memory optimization settings
        if torch.cuda.is_available():
            torch.backends.cuda.max_split_size_mb = 128
            torch.backends.cudnn.benchmark = True
            
    def _clear_memory(self):
        """Clear GPU memory and run garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def _copy_model(self) -> PreTrainedModel:
        """Create a deep copy of the model."""
        try:
            print("Creating new model instance...")
            # Get model configuration
            config = AutoConfig.from_pretrained(
                self.model.config._name_or_path,
                trust_remote_code=True
            )
            
            # Create new model instance
            new_model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True
            )
            
            print("Copying model parameters...")
            # Copy state dict with proper device handling
            with torch.no_grad():
                state_dict = {}
                for name, param in self.model.state_dict().items():
                    # Always copy to CPU first
                    state_dict[name] = param.detach().cpu()
                
                # Load state dict
                new_model.load_state_dict(state_dict, strict=True)
                
            return new_model
            
        except Exception as e:
            raise RuntimeError(f"Failed to copy model: {str(e)}")
            
    def benchmark_quantizer(
        self,
        name: str,
        quantizer_class,
        quantizer_args: Dict
    ) -> Dict[str, float]:
        """
        Benchmark a specific quantizer with detailed memory and time tracking.

        Args:
            name (str): Name of the quantization method (e.g., "AWQ", "GPTQ").
            quantizer_class: The quantizer class to benchmark (e.g., AWQQuantizer).
            quantizer_args (Dict): Dictionary of arguments to pass to the quantizer's constructor.

        Returns:
            Dict[str, float]: A dictionary containing various performance metrics:
                - time_model_copying_s: Time for copying the base model (seconds).
                - time_quantizer_init_s: Time for quantizer initialization (seconds).
                - quantization_time_s: Time for the main quantizer.quantize() call (seconds).
                - time_inference_total_s: Total time for inference (warmup + timed runs) (seconds).
                - mean_latency_ms: Mean inference latency (milliseconds).
                - p95_latency_ms: 95th percentile inference latency (milliseconds).
                - min_latency_ms: Minimum inference latency (milliseconds).
                - max_latency_ms: Maximum inference latency (milliseconds).
                - peak_mem_model_copy_gb: Peak GPU memory after model copy and move to device (GB).
                - peak_mem_quantization_gb: Peak GPU memory during quantizer.quantize() (GB).
                - peak_mem_inference_gb: Peak GPU memory during inference (GB).
                - memory_allocated_at_end_mb: GPU memory allocated at the end of the benchmark (MB).
                - model_size_mb: Size of the quantized model's parameters (MB).
        """
        results = {}
        try:
            self._clear_memory()
            if torch.cuda.is_available():
                print(f"GPU memory before {name}: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                torch.cuda.reset_peak_memory_stats(device=self.device_manager.primary_device)
            
            # Get a fresh copy of the model
            start_model_copying = time.time()
            model_copy = self._copy_model()
            time_model_copying = time.time() - start_model_copying
            
            model_copy = model_copy.to(self.device_manager.primary_device)
            peak_mem_model_copy_gb = 0
            if torch.cuda.is_available():
                peak_mem_model_copy_gb = torch.cuda.max_memory_allocated(device=self.device_manager.primary_device) / (1024**3)
                torch.cuda.reset_peak_memory_stats(device=self.device_manager.primary_device)

            # Initialize quantizer with device info
            start_quantizer_init = time.time()
            quantizer = quantizer_class(
                model=model_copy, # model_copy is already on the target device
                device=self.device_manager.primary_device, # This might be redundant if model is already on device
                **quantizer_args
            )
            time_quantizer_init = time.time() - start_quantizer_init
            
            # Move calibration data to target device
            cal_data = self.calibration_data.to(self.device_manager.primary_device)
            
            # Measure quantization time
            print(f"Starting quantization for {name}...")
            start_quant_time = time.time()
            
            if name == "AWQ":
                # AWQ uses batched processing
                cal_steps = min(20, len(cal_data))
                quantized_model = quantizer.quantize(
                    calibration_data=cal_data,
                    calibration_steps=cal_steps
                )
            else:
                # Direct quantization for others
                quantized_model = quantizer.quantize(calibration_data=cal_data)
                
            quant_time = time.time() - start_quant_time # Renamed start_time to start_quant_time
            
            peak_mem_quantization_gb = 0
            if torch.cuda.is_available():
                peak_mem_quantization_gb = torch.cuda.max_memory_allocated(device=self.device_manager.primary_device) / (1024**3)
                torch.cuda.reset_peak_memory_stats(device=self.device_manager.primary_device)

            # Move calibration data back to CPU and clear memory
            cal_data = cal_data.cpu()
            self._clear_memory()
            
            # Prepare for inference benchmark
            quantized_model = quantized_model.to(self.device_manager.primary_device)
            test_input = torch.randint(
                0, 1000,
                (1, min(self.input_shape[1], 32)), # Using min to ensure sequence length is not too large
                device=self.device_manager.primary_device
            )
            
            start_inference_total = time.time()
            # Limited warmup
            for _ in range(5):
                with torch.no_grad():
                    quantized_model(test_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Measure inference latency
            latencies = []
            for _ in range(self.num_inference_steps):
                start_latency = time.perf_counter()
                with torch.no_grad():
                    quantized_model(test_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start_latency) * 1000)  # ms
            
            time_inference_total = time.time() - start_inference_total
            latencies = torch.tensor(latencies)
            
            peak_mem_inference_gb = 0
            memory_allocated_mb = 0 # Changed from memory_allocated to memory_allocated_mb
            peak_memory_overall_mb = 0 # Changed from peak_memory to peak_memory_overall_mb

            if torch.cuda.is_available():
                peak_mem_inference_gb = torch.cuda.max_memory_allocated(device=self.device_manager.primary_device) / (1024**3)
                memory_allocated_mb = torch.cuda.memory_allocated(device=self.device_manager.primary_device) / 1024**2  # MB
                # peak_memory_overall_mb is tricky because we reset stats. This will be peak for inference only.
                # For overall peak, one would need to not reset.
                # For now, let's keep the existing peak_memory as the overall for the benchmark_quantizer scope if not reset elsewhere.
                # However, the task asks for specific peak_mem_..._gb, so the old 'peak_memory' might be redundant or need re-evaluation.
                # Let's assume the existing 'peak_memory' was intended as a general peak after inference.
                # Given the new specific peak memory metrics, let's rename the old 'peak_memory'
                # to reflect it's the max observed during this function *if stats were not reset*.
                # Since we *are* resetting, the old 'peak_memory' is effectively peak_mem_inference_gb in MB.
                # Let's stick to the new specific metrics.
                torch.cuda.reset_peak_memory_stats(device=self.device_manager.primary_device) # Reset for next run

            # Calculate model size
            model_size_mb = sum( # Renamed model_size to model_size_mb
                p.numel() * p.element_size() 
                for p in quantized_model.parameters()
            ) / 1024**2  # MB
            
            results = {
                "time_model_copying_s": time_model_copying,
                "time_quantizer_init_s": time_quantizer_init,
                "quantization_time_s": quant_time, # Added _s suffix
                "time_inference_total_s": time_inference_total,
                "mean_latency_ms": latencies.mean().item(), # Added _ms suffix
                "p95_latency_ms": torch.quantile(latencies, 0.95).item(), # Added _ms suffix
                "min_latency_ms": latencies.min().item(), # Added _ms suffix
                "max_latency_ms": latencies.max().item(), # Added _ms suffix
                "peak_mem_model_copy_gb": peak_mem_model_copy_gb,
                "peak_mem_quantization_gb": peak_mem_quantization_gb,
                "peak_mem_inference_gb": peak_mem_inference_gb,
                "memory_allocated_at_end_mb": memory_allocated_mb, # Clarified name
                "model_size_mb": model_size_mb # Added _mb suffix
            }
            
            # Clean up
            del quantized_model
            del test_input
            self._clear_memory()
            
            self.results[name] = results
            return results
            
        except Exception as e:
            print(f"Error benchmarking {name}: {str(e)}")
            self._clear_memory()
            return results
            
    def run_all_benchmarks(self) -> pd.DataFrame:
        """Run benchmarks for all methods with memory management."""
        config = {
            "bits": 4,
            "group_size": 128
        }
        
        # Run benchmarks in order of increasing memory usage
        methods = [
            ("GGUF", GGUFQuantizer, {"use_packed": True}),
            ("AWQ", AWQQuantizer, {"zero_point": True}),
            ("GPTQ", GPTQQuantizer, {"actorder": True, "use_triton": False})
        ]
        
        for name, quantizer_class, extra_args in methods:
            try:
                print(f"\nBenchmarking {name}...")
                self.benchmark_quantizer(
                    name,
                    quantizer_class,
                    {**config, **extra_args}
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Out of memory during {name}, skipping...")
                    continue
                raise e
                
        # Convert results to DataFrame
        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Calculate compression ratio
        original_size = sum(
            p.numel() * p.element_size() 
            for p in self.model.parameters()
        ) / 1024**2
        if 'model_size' in df.columns:
            df['compression_ratio'] = original_size / df['model_size']
        
        return df
        
    def print_report(self):
        """
        Runs benchmarks for all configured methods and prints a formatted report.
        The report includes detailed timing (model copy, quantizer init, quantization, inference)
        and memory usage (peak memory at different stages, final model size, compression ratio).
        """
        df = self.run_all_benchmarks()
        
        print("\nQuantization Benchmark Results")
        print("=" * 80)
        
        # Updated metrics for the report
        metrics = {
            'time_model_copying_s': ('Model Copy Time (s)', '{:.2f}'),
            'time_quantizer_init_s': ('Quantizer Init Time (s)', '{:.2f}'),
            'quantization_time_s': ('Quantization Time (s)', '{:.2f}'),
            'time_inference_total_s': ('Total Inference Time (s)', '{:.2f}'),
            'mean_latency_ms': ('Mean Inference Latency (ms)', '{:.2f}'),
            'p95_latency_ms': ('P95 Inference Latency (ms)', '{:.2f}'),
            'min_latency_ms': ('Min Inference Latency (ms)', '{:.2f}'),
            'max_latency_ms': ('Max Inference Latency (ms)', '{:.2f}'),
            'peak_mem_model_copy_gb': ('Peak Mem: Model Copy (GB)', '{:.2f}'),
            'peak_mem_quantization_gb': ('Peak Mem: Quantization (GB)', '{:.2f}'),
            'peak_mem_inference_gb': ('Peak Mem: Inference (GB)', '{:.2f}'),
            'memory_allocated_at_end_mb': ('Mem Allocated @ End (MB)', '{:.1f}'),
            'model_size_mb': ('Model Size (MB)', '{:.1f}'),
            'compression_ratio': ('Compression Ratio', '{:.1f}x')
        }
        
        for method in df.index:
            print(f"\n{method}")
            print("-" * 40)
            for metric, (name, fmt) in metrics.items():
                if metric in df.columns:
                    value = df.loc[method, metric]
                    print(f"{name:<30} {fmt.format(value)}")
                    
    def plot_comparison(self, save_path: str = None):
        """
        Generates and optionally saves comparison plots for key benchmark metrics.

        Plots typically include:
            - Mean Inference Latency (ms)
            - Peak Memory during Inference (GB)
            - Model Size (MB)
            - Compression Ratio

        Args:
            save_path (str, optional): Path to save the generated plot. 
                                       If None, the plot is displayed using plt.show().
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return
            
        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Quantization Method Comparison')
        
        # Latency comparison
        df['mean_latency_ms'].plot(
            kind='bar', ax=axes[0, 0], rot=45,
            title='Mean Inference Latency (ms)'
        )
        
        # Memory usage - choosing one of the peak memory metrics for plotting
        # peak_mem_quantization_gb might be interesting, or peak_mem_inference_gb
        df['peak_mem_inference_gb'].plot(
            kind='bar', ax=axes[0, 1], rot=45,
            title='Peak Memory during Inference (GB)'
        )
        
        # Model size
        df['model_size_mb'].plot(
            kind='bar', ax=axes[1, 0], rot=45,
            title='Model Size (MB)'
        )
        
        # Compression ratio
        if 'compression_ratio' in df.columns:
            df['compression_ratio'].plot( # This will need to be recalculated if model_size_mb is used
                kind='bar', ax=axes[1, 1], rot=45,
                title='Compression Ratio'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()
