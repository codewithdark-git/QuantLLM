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
        self.num_warmup_steps = kwargs.get("num_warmup_steps", 10) # Default to 10 warmup steps
        self.results = {}
        self.pynvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml_available = True
            self.nvml_handle = None # Will be set per device
            print("pynvml initialized successfully. GPU utilization monitoring might be available.")
        except Exception as e: # Catch broader exceptions for NVML init
            print(f"pynvml could not be initialized: {e}. GPU utilization monitoring will be unavailable.")
            self.pynvml_available = False
        
        # Memory optimization settings
        if torch.cuda.is_available():
            torch.backends.cuda.max_split_size_mb = 128 # Consider making this configurable
            torch.backends.cudnn.benchmark = True
            
    def _get_gpu_utilization(self) -> Optional[float]:
        """Gets current GPU utilization if pynvml is available."""
        if self.pynvml_available and self.nvml_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                return util.gpu
            except Exception as e: # Catch pynvml errors
                # print(f"Could not get GPU utilization: {e}") # Can be noisy
                return None
        return None

    def _get_gpu_memory_info_gb(self) -> Optional[Tuple[float, float]]:
        """Gets current GPU memory (used, total) in GB if pynvml is available."""
        if self.pynvml_available and self.nvml_handle:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                return mem_info.used / (1024**3), mem_info.total / (1024**3)
            except Exception: # Catch pynvml errors
                return None
        return None

    def _clear_memory(self):
        """Clear GPU memory and run garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # torch.cuda.synchronize() # Synchronization is better around specific operations
            
    def _copy_model(self) -> PreTrainedModel:
        """Create a deep copy of the model, ensuring it's on CPU initially."""
        try:
            print("Creating new model instance...")
            print("Creating new model instance from config...")
            # Get model configuration
            config = AutoConfig.from_pretrained(
                self.model.config._name_or_path, # Use the original model's name or path
                trust_remote_code=True # Add trust_remote_code=True if needed for custom models
            )
            
            # Create new model instance on CPU
            new_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to("cpu")
            
            print("Copying model parameters (state_dict) to CPU...")
            # Copy state dict from the original self.model (which is on CPU)
            with torch.no_grad():
                state_dict_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
                new_model.load_state_dict(state_dict_cpu, assign=True, strict=True)
                del state_dict_cpu # Free memory
                
            return new_model
            
        except Exception as e:
            print(f"Detailed error in _copy_model: {type(e).__name__}: {e}")
            raise RuntimeError(f"Failed to copy model: {str(e)}")
            
    def benchmark_quantizer(
        self,
        name: str,
        quantizer_class,
        quantizer_args: Dict,
        # Add original_model_size_gb for memory efficiency calculation
        original_model_size_gb: float
    ) -> Dict[str, Union[float, str]]:
        """
        Benchmark a specific quantizer with detailed memory, time, and GPU utilization tracking.
        Output format aims to match the specifications provided in the issue.
        """
        current_results: Dict[str, Union[float, str]] = {} # Use Union for mixed types
        
        # Initialize NVML handle for the primary device if available
        if self.pynvml_available and self.device_manager.primary_device.type == 'cuda':
            try:
                nvml_device_index = self.device_manager.primary_device.index if self.device_manager.primary_device.index is not None else 0
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_device_index)
            except Exception as e:
                print(f"Failed to get NVML handle for device {self.device_manager.primary_device}: {e}")
                self.nvml_handle = None # Ensure it's None if handle acquisition fails
        else:
            self.nvml_handle = None


        try:
            self._clear_memory()
            if torch.cuda.is_available() and self.device_manager.primary_device.type == 'cuda':
                print(f"Initial GPU memory allocated on {self.device_manager.primary_device}: {torch.cuda.memory_allocated(self.device_manager.primary_device) / 1024**3:.3f} GB")
                torch.cuda.reset_peak_memory_stats(self.device_manager.primary_device)
            
            # 1. Model Copying
            time_start_model_copying = time.perf_counter()
            model_copy = self._copy_model() # This returns model on CPU
            current_results["time_model_copying_s"] = time.perf_counter() - time_start_model_copying
            
            model_copy = model_copy.to(self.device_manager.primary_device)
            if torch.cuda.is_available() and self.device_manager.primary_device.type == 'cuda':
                torch.cuda.synchronize(self.device_manager.primary_device) # Wait for move to complete
                current_results["peak_mem_model_copy_gb"] = torch.cuda.max_memory_allocated(self.device_manager.primary_device) / (1024**3)
                torch.cuda.reset_peak_memory_stats(self.device_manager.primary_device)
            else:
                current_results["peak_mem_model_copy_gb"] = 0.0

            # 2. Quantizer Initialization
            time_start_quantizer_init = time.perf_counter()
            quantizer = quantizer_class(
                model_name=model_copy, # Pass the model instance
                device=self.device_manager.primary_device,
                **quantizer_args
            )
            current_results["time_quantizer_init_s"] = time.perf_counter() - time_start_quantizer_init
            
            # Ensure calibration data is on the correct device for quantization
            # Quantizers might move data internally, but good to be explicit if needed by quantizer's API
            cal_data_device = self.calibration_data.to(self.device_manager.primary_device)
            
            # 3. Quantization
            print(f"Starting quantization for {name} on {self.device_manager.primary_device}...")
            time_start_quantization = time.perf_counter()
            
            # AWQ specific handling from original code
            if name == "AWQ" and hasattr(quantizer, 'quantize') and 'calibration_steps' in quantizer.quantize.__code__.co_varnames:
                cal_steps = min(quantizer_args.get("calibration_steps", 20), len(cal_data_device))
                quantized_model = quantizer.quantize(calibration_data=cal_data_device, calibration_steps=cal_steps)
            else: # General case
                quantized_model = quantizer.quantize(calibration_data=cal_data_device)
            
            if torch.cuda.is_available() and self.device_manager.primary_device.type == 'cuda':
                torch.cuda.synchronize(self.device_manager.primary_device)
            current_results["quantization_time_s"] = time.perf_counter() - time_start_quantization
            
            if torch.cuda.is_available() and self.device_manager.primary_device.type == 'cuda':
                current_results["peak_mem_quantization_gb"] = torch.cuda.max_memory_allocated(self.device_manager.primary_device) / (1024**3)
                torch.cuda.reset_peak_memory_stats(self.device_manager.primary_device)
            else:
                current_results["peak_mem_quantization_gb"] = 0.0

            del cal_data_device
            self._clear_memory()
            
            # Ensure quantized model is on the primary device for inference
            quantized_model = quantized_model.to(self.device_manager.primary_device)
            
            # 4. Inference Benchmarking (Warm-up and Timed Runs)
            # Use a fixed input shape for fair comparison, ensure batch_size=1 for latency
            # The self.input_shape is (batch_size, seq_len). For latency, usually batch_size=1.
            # Let's use (1, seq_len_from_self.input_shape)
            inference_batch_size = 1 
            sequence_length = self.input_shape[1]
            test_input = torch.randint(
                0, quantized_model.config.vocab_size, # Use model's vocab size
                (inference_batch_size, sequence_length), 
                device=self.device_manager.primary_device
            )
            
            print(f"Starting warm-up ({self.num_warmup_steps} steps)...")
            for _ in range(self.num_warmup_steps):
                with torch.no_grad():
                    _ = quantized_model(test_input)
                if torch.cuda.is_available() and self.device_manager.primary_device.type == 'cuda':
                    torch.cuda.synchronize(self.device_manager.primary_device)
            
            print(f"Starting timed inference ({self.num_inference_steps} steps)...")
            latencies_ms = []
            gpu_utilizations = [] # Store utilization per step if available
            
            time_start_timed_inference = time.perf_counter()
            for _ in range(self.num_inference_steps):
                step_start_time = time.perf_counter()
                with torch.no_grad():
                    _ = quantized_model(test_input)
                if torch.cuda.is_available() and self.device_manager.primary_device.type == 'cuda':
                    torch.cuda.synchronize(self.device_manager.primary_device)
                latencies_ms.append((time.perf_counter() - step_start_time) * 1000)
                
                gpu_util = self._get_gpu_utilization()
                if gpu_util is not None:
                    gpu_utilizations.append(gpu_util)

            total_timed_inference_s = time.perf_counter() - time_start_timed_inference
            current_results["time_inference_total_s"] = total_timed_inference_s # This is only timed part
            
            latencies_tensor = torch.tensor(latencies_ms)
            current_results["mean_latency_ms"] = latencies_tensor.mean().item()
            current_results["p90_latency_ms"] = torch.quantile(latencies_tensor, 0.90).item()
            current_results["p95_latency_ms"] = torch.quantile(latencies_tensor, 0.95).item()
            current_results["p99_latency_ms"] = torch.quantile(latencies_tensor, 0.99).item()
            current_results["min_latency_ms"] = latencies_tensor.min().item()
            current_results["max_latency_ms"] = latencies_tensor.max().item()
            
            current_results["throughput_tokens_per_s"] = (self.num_inference_steps * sequence_length) / total_timed_inference_s
            current_results["throughput_inf_per_s"] = self.num_inference_steps / total_timed_inference_s


            if torch.cuda.is_available() and self.device_manager.primary_device.type == 'cuda':
                current_results["peak_mem_inference_gb"] = torch.cuda.max_memory_allocated(self.device_manager.primary_device) / (1024**3)
                current_results["final_mem_allocated_gb"] = torch.cuda.memory_allocated(self.device_manager.primary_device) / (1024**3)
                # Overall peak memory during the benchmark_quantizer call (if not resetting stats often)
                # This might be more complex to track accurately if resets happen.
                # For now, peak_mem_quantization_gb and peak_mem_inference_gb are key.
            else:
                current_results["peak_mem_inference_gb"] = 0.0
                current_results["final_mem_allocated_gb"] = 0.0

            if gpu_utilizations:
                current_results["mean_gpu_utilization_percent"] = np.mean(gpu_utilizations)
                current_results["peak_gpu_utilization_percent"] = np.max(gpu_utilizations)
            else:
                current_results["mean_gpu_utilization_percent"] = "N/A" # Mark as N/A if not available
                current_results["peak_gpu_utilization_percent"] = "N/A"
            
            # 5. Model Size and Efficiency
            quantized_model_size_params_gb = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024**3)
            current_results["model_param_size_gb"] = quantized_model_size_params_gb
            
            # Compression Ratio (params vs params)
            current_results["compression_ratio_params"] = original_model_size_gb / quantized_model_size_params_gb if quantized_model_size_params_gb > 0 else 0
            
            # Memory Efficiency (peak inference vs original model size)
            # This definition of memory efficiency might need refinement based on exact requirements.
            # E.g., (original_peak_inference_mem - quantized_peak_inference_mem) / original_peak_inference_mem
            # For now, comparing quantized peak inference memory to original model parameter size.
            if original_model_size_gb > 0 :
                # How much smaller is peak inference memory compared to original model's parameter footprint
                # A simple one: (original_param_size - peak_inference_mem_quantized) / original_param_size
                 peak_inf_mem_gb = current_results.get("peak_mem_inference_gb", 0.0)
                 if isinstance(peak_inf_mem_gb, float): # Ensure it's a float
                    current_results["memory_efficiency_percent"] = ((original_model_size_gb - peak_inf_mem_gb) / original_model_size_gb) * 100
                 else:
                    current_results["memory_efficiency_percent"] = "N/A"
            else:
                current_results["memory_efficiency_percent"] = "N/A"

            self.results[name] = current_results
            return current_results
            
        except Exception as e:
            print(f"Error benchmarking {name}: {type(e).__name__} - {str(e)}")
            import traceback
            traceback.print_exc()
            # Store partial results or error message if needed
            self.results[name] = {"error": str(e), **current_results} 
            return {"error": str(e), **current_results} 
        finally:
            if self.pynvml_available and self.nvml_handle:
                # No explicit shutdown needed for handle per call, but global pynvml.nvmlShutdown() at end of program.
                pass
            # Clean up model and other large objects
            if 'quantized_model' in locals(): del quantized_model
            if 'model_copy' in locals(): del model_copy
            if 'quantizer' in locals(): del quantizer
            if 'test_input' in locals(): del test_input
            self._clear_memory()
            if torch.cuda.is_available() and self.device_manager.primary_device.type == 'cuda':
                 print(f"GPU memory after {name} benchmark and cleanup: {torch.cuda.memory_allocated(self.device_manager.primary_device) / 1024**3:.3f} GB")
                 torch.cuda.reset_peak_memory_stats(self.device_manager.primary_device)


    def run_all_benchmarks(self) -> pd.DataFrame:
        """Run benchmarks for all methods with memory management."""
        # Calculate original model size once
        original_model_size_gb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3)
        self.results["Original (FP16/FP32)"] = { # Add a baseline entry for comparison
            "model_param_size_gb": original_model_size_gb,
            "mean_latency_ms": "N/A (Benchmark separately if needed)", # Placeholder
            "throughput_inf_per_s": "N/A",
            "peak_mem_inference_gb": original_model_size_gb, # Approx initial loaded size
             # Add other relevant fields as "N/A" or sensible defaults
            "time_model_copying_s": 0, "time_quantizer_init_s": 0, "quantization_time_s": 0,
            "time_inference_total_s": "N/A", "p90_latency_ms": "N/A", "p95_latency_ms": "N/A",
            "p99_latency_ms": "N/A", "min_latency_ms": "N/A", "max_latency_ms": "N/A",
            "throughput_tokens_per_s": "N/A", "peak_mem_model_copy_gb": original_model_size_gb,
            "peak_mem_quantization_gb": "N/A", "final_mem_allocated_gb": original_model_size_gb,
            "mean_gpu_utilization_percent": "N/A", "peak_gpu_utilization_percent": "N/A",
            "compression_ratio_params": 1.0, "memory_efficiency_percent": 0.0,
        }

        config = {
            "bits": 4, # Common setting
            "group_size": 128 # Common setting
        }
        
        # Define methods to benchmark
        # Ensure quantizer_args match what each quantizer expects
        # For GGUF, model_name is handled by its __init__ if a model instance is passed
        methods_to_benchmark = [
            # Example: ("GGUF_Q4K", GGUFQuantizer, {"bits": 4, "group_size": -1, "use_packed": True}), # GGUF needs model_name typically
            # For GGUF, if passing a model instance, its __init__ takes `model_name` which can be the model object.
            # The `device` arg in GGUFQuantizer is for its internal device manager, not the model's initial device.
            ("GGUF_Q4_GS128", GGUFQuantizer, {"bits": 4, "group_size": 128, "use_packed": True}),
            ("AWQ_B4_GS128", AWQQuantizer, {"bits": 4, "group_size": 128, "zero_point": True}),
            ("GPTQ_B4_GS128", GPTQQuantizer, {"bits": 4, "group_size": 128, "actorder": False, "use_triton": False}), # Triton often needs setup
        ]
        
        for name, quantizer_class, quantizer_specific_args in methods_to_benchmark:
            try:
                print(f"\nBenchmarking {name}...")
                # Merge common config with specific args. Specific args override common ones.
                current_quantizer_args = {**config, **quantizer_specific_args}
                
                self.benchmark_quantizer(
                    name,
                    quantizer_class,
                    current_quantizer_args,
                    original_model_size_gb=original_model_size_gb
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Out of memory during {name} benchmark, skipping...")
                    self.results[name] = {"error": "OOM", "quantization_time_s": "OOM"}
                    continue
                elif "not implemented" in str(e).lower():
                     print(f"Feature not implemented for {name}, skipping... Error: {e}")
                     self.results[name] = {"error": "Not Implemented", "quantization_time_s": "Not Implemented"}
                     continue
                print(f"Runtime error during {name} benchmark: {e}. Skipping.")
                self.results[name] = {"error": str(e), "quantization_time_s": "Error"}
                # raise e # Optionally re-raise if you want to halt on any error
            except Exception as e:
                print(f"Unexpected error during {name} benchmark: {type(e).__name__} - {e}. Skipping.")
                self.results[name] = {"error": str(e), "quantization_time_s": "Error"}


        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Ensure numeric conversion for relevant columns, coercing errors for "N/A" or "OOM"
        numeric_cols = ['model_param_size_gb', 'mean_latency_ms', 'throughput_inf_per_s', 
                        'peak_mem_inference_gb', 'compression_ratio_params', 'memory_efficiency_percent',
                        'quantization_time_s', 'p90_latency_ms', 'p95_latency_ms', 'p99_latency_ms',
                        'peak_mem_model_copy_gb', 'peak_mem_quantization_gb', 'final_mem_allocated_gb',
                        'throughput_tokens_per_s']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.sort_values(by="mean_latency_ms", ascending=True) # Sort by a relevant metric
        
    def print_report(self):
        """
        Runs benchmarks and prints a formatted report with enhanced metrics.
        """
        # Ensure results are populated by running benchmarks if not already done
        if not self.results or len(self.results) <=1 : # <=1 because of "Original" entry
            print("No benchmark results found, running benchmarks first...")
            self.run_all_benchmarks()
        
        df = pd.DataFrame.from_dict(self.results, orient='index')
        # Re-apply numeric conversion in case print_report is called standalone
        numeric_cols = ['model_param_size_gb', 'mean_latency_ms', 'throughput_inf_per_s', 
                        'peak_mem_inference_gb', 'compression_ratio_params', 'memory_efficiency_percent',
                        'quantization_time_s', 'p90_latency_ms', 'p95_latency_ms', 'p99_latency_ms',
                        'peak_mem_model_copy_gb', 'peak_mem_quantization_gb', 'final_mem_allocated_gb',
                        'throughput_tokens_per_s', 'time_model_copying_s', 'time_quantizer_init_s',
                        'min_latency_ms', 'max_latency_ms', 'mean_gpu_utilization_percent', 'peak_gpu_utilization_percent']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print("\n===== Quantization Benchmark Report =====")
        
        # Define metrics to display: {column_name: (Display Name, Format String, LargerIsBetter)}
        # Adjusted to match the new/renamed metrics in benchmark_quantizer
        metrics_to_display = {
            "quantization_time_s": ("Quantization Time (s)", "{:.2f}", False),
            "model_param_size_gb": ("Model Size (Params, GB)", "{:.3f}", False),
            "compression_ratio_params": ("Compression Ratio (Params)", "{:.2f}x", True),
            "mean_latency_ms": ("Mean Latency (ms)", "{:.2f}", False),
            "p90_latency_ms": ("P90 Latency (ms)", "{:.2f}", False),
            "p95_latency_ms": ("P95 Latency (ms)", "{:.2f}", False),
            "p99_latency_ms": ("P99 Latency (ms)", "{:.2f}", False),
            "throughput_inf_per_s": ("Throughput (inf/s)", "{:.2f}", True),
            "throughput_tokens_per_s": ("Throughput (tokens/s)", "{:.2f}", True),
            "peak_mem_quantization_gb": ("Peak Mem: Quant (GB)", "{:.3f}", False),
            "peak_mem_inference_gb": ("Peak Mem: Inference (GB)", "{:.3f}", False),
            "final_mem_allocated_gb": ("Final Mem Allocated (GB)", "{:.3f}", False),
            "memory_efficiency_percent": ("Memory Efficiency (%)", "{:.2f}%", True),
            "mean_gpu_utilization_percent": ("Mean GPU Util (%)", "{:.1f}%", True),
            "peak_gpu_utilization_percent": ("Peak GPU Util (%)", "{:.1f}%", True),
            "time_model_copying_s": ("Model Copy Time (s)", "{:.2f}", False),
            "time_quantizer_init_s": ("Quantizer Init Time (s)", "{:.2f}", False),
            "error": ("Errors", "{}", False)
        }

        # Print sorted by a chosen metric, e.g., Mean Latency
        # Handle cases where sorting metric might be NaN due to errors
        if "mean_latency_ms" in df.columns and not df["mean_latency_ms"].isnull().all():
             df_sorted = df.sort_values(by="mean_latency_ms", ascending=True, na_position='last')
        elif "quantization_time_s" in df.columns and not df["quantization_time_s"].isnull().all():
             df_sorted = df.sort_values(by="quantization_time_s", ascending=True, na_position='last')
        else:
            df_sorted = df

        for method_name in df_sorted.index:
            print(f"\n--- Method: {method_name} ---")
            if df_sorted.loc[method_name].get("error") not in [None, "N/A", np.nan]:
                print(f"  Error: {df_sorted.loc[method_name]['error']}")
                # Optionally print only a few key metrics if there was an error
                # for metric_key in ["quantization_time_s", "model_param_size_gb"]:
                #    if metric_key in df_sorted.columns: ...
                continue # Skip detailed metrics if major error like OOM

            for metric_key, (display_name, fmt_str, _) in metrics_to_display.items():
                if metric_key in df_sorted.columns:
                    value = df_sorted.loc[method_name, metric_key]
                    if pd.isna(value) or value == "N/A":
                        val_str = "N/A"
                    elif isinstance(value, str): # Already formatted string for error
                        val_str = value
                    else:
                        try:
                            val_str = fmt_str.format(value)
                        except (ValueError, TypeError):
                            val_str = str(value) # Fallback if format fails
                    print(f"  {display_name:<30}: {val_str}")
        
        print("=" * 40)
        if self.pynvml_available:
            pynvml.nvmlShutdown() # Clean up NVML
            print("pynvml shutdown.")

    def plot_comparison(self, save_path: str = None):
        """
        Generates and optionally saves comparison plots for key benchmark metrics.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns # For better aesthetics
            sns.set_theme(style="whitegrid")
        except ImportError:
            print("matplotlib and seaborn are required for plotting. Please install them.")
            return
            
        if not self.results or len(self.results) <=1:
             print("No benchmark data to plot. Run benchmarks first.")
             return

        df = pd.DataFrame.from_dict(self.results, orient='index')
        # Remove 'Original' row for plotting if it only contains N/A for key metrics
        # Or handle it by ensuring it has numeric data / selectively plot
        df_plot = df.drop(index="Original (FP16/FP32)", errors='ignore')

        # Convert relevant columns to numeric, coercing errors
        plot_metrics_cols = ['mean_latency_ms', 'throughput_inf_per_s', 
                             'peak_mem_inference_gb', 'model_param_size_gb', 
                             'compression_ratio_params', 'memory_efficiency_percent',
                             'mean_gpu_utilization_percent']
        for col in plot_metrics_cols:
            if col in df_plot.columns:
                df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
        df_plot.dropna(subset=plot_metrics_cols, how='all', inplace=True) # Drop rows where all plot metrics are NaN

        if df_plot.empty:
            print("No valid data available for plotting after filtering.")
            return

        num_metrics = len(plot_metrics_cols)
        # Adjust layout: aim for 2 columns of plots
        ncols = 2
        nrows = (num_metrics + ncols - 1) // ncols 
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
        axes_flat = axes.flatten() # Flatten for easy iteration

        for i, metric_key in enumerate(plot_metrics_cols):
            if metric_key in df_plot.columns and not df_plot[metric_key].isnull().all():
                ax = axes_flat[i]
                # Get display name from print_report's map, default to key
                display_name = metric_key 
                # Example of how to get display name if print_report's map was accessible here
                # display_name = metrics_to_display.get(metric_key, (metric_key, "", True))[0]

                sns.barplot(x=df_plot.index, y=metric_key, ax=ax, palette="viridis")
                ax.set_title(display_name.replace("_", " ").title(), fontsize=14)
                ax.set_xlabel("Quantization Method", fontsize=12)
                ax.set_ylabel(display_name.split(" (")[0], fontsize=12) # Unit from display name
                ax.tick_params(axis='x', rotation=45, ha='right')
                ax.grid(True, linestyle='--', alpha=0.7)
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        plt.tight_layout(pad=3.0)
        fig.suptitle('Quantization Method Comparison', fontsize=18, y=1.02) # Adjust title position
        
        if save_path:
            print(f"Saving plot to {save_path}")
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close(fig) # Close the figure to free memory
            # No explicit cleanup for self.model or self.calibration_data here, they are persistent.
        if self.pynvml_available and self.nvml_handle:
                # nvmlShutdown is typically called once when the application exits, not per benchmark.
                # For now, do not shut down NVML here to allow multiple calls to benchmark_quantizer or run_all_benchmarks.
                # Consider adding a __del__ or close() method to QuantizationBenchmark for global NVML shutdown.
                pass
                 
    def __del__(self):
        # Destructor to ensure NVML is shut down when the object is deleted or program exits.
        if self.pynvml_available:
            try:
                # Check if nvmlInit was called by checking a flag or if any handles were created.
                # This simple check assumes if pynvml_available is true, init was attempted.
                print("Shutting down pynvml...")
                pynvml.nvmlShutdown()
            except Exception as e:
                print(f"Error during pynvml shutdown: {e}")
                pass # Avoid errors during shutdown

    def run_all_benchmarks(self) -> pd.DataFrame: # Keep existing signature
        """Run benchmarks for all methods with memory management."""
        # Calculate original model size once
        # Ensure self.model is on CPU for this calculation to avoid CUDA context if not needed yet
        original_model_cpu = self.model.to('cpu')
        original_model_size_gb = sum(p.numel() * p.element_size() for p in original_model_cpu.parameters()) / (1024**3)
        del original_model_cpu # free the cpu copy if it was made just for this
        self._clear_memory()

        # Baseline entry for the original model (mostly N/A as it's not quantized)
        self.results["Original (FP16/FP32 Estimate)"] = {
            "quantization_time_s": 0.0,
            "model_param_size_gb": original_model_size_gb,
            "compression_ratio_params": 1.0,
            "mean_latency_ms": "N/A", # Needs separate benchmarking run
            "p90_latency_ms": "N/A",
            "p95_latency_ms": "N/A",
            "p99_latency_ms": "N/A",
            "throughput_inf_per_s": "N/A",
            "throughput_tokens_per_s": "N/A",
            "peak_mem_quantization_gb": 0.0, # No quantization
             # Estimate peak inference as model size; actual can be higher due to activations
            "peak_mem_inference_gb": original_model_size_gb, 
            "final_mem_allocated_gb": original_model_size_gb,
            "memory_efficiency_percent": 0.0,
            "mean_gpu_utilization_percent": "N/A",
            "peak_gpu_utilization_percent": "N/A",
            "time_model_copying_s": 0.0,
            "time_quantizer_init_s": 0.0,
            "error": "N/A"
        }
        
        common_quant_config = {"bits": 4, "group_size": 128}
        
        methods_to_benchmark = [
            # Name for report, Quantizer Class, Specific args for this quantizer
            ("GGUF_Q4_GS128", GGUFQuantizer, {"use_packed": True}), # GGUF specific
            ("AWQ_B4_GS128", AWQQuantizer, {"zero_point": True}),   # AWQ specific
            ("GPTQ_B4_GS128", GPTQQuantizer, {"actorder": False, "use_triton": False}), # GPTQ specific
        ]
        
        for name, quantizer_class, specific_args in methods_to_benchmark:
            print(f"\n--- Starting Benchmark for: {name} ---")
            # Combine common config with specific args. Specific args take precedence.
            current_quantizer_args = {**common_quant_config, **specific_args}
            
            try:
                self.benchmark_quantizer(
                    name,
                    quantizer_class,
                    current_quantizer_args,
                    original_model_size_gb=original_model_size_gb
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Out of memory during {name} benchmark. Storing partial results.")
                    if name not in self.results: self.results[name] = {} # Ensure entry exists
                    self.results[name].update({"error": "OOM", "quantization_time_s": "OOM"})
                elif "not implemented" in str(e).lower() or "cublaslt CUSPARSE_STATUS_NOT_INITIALIZED" in str(e): # Triton/kernel issues
                     print(f"Feature not implemented or kernel error for {name}. Error: {e}. Storing partial results.")
                     if name not in self.results: self.results[name] = {}
                     self.results[name].update({"error": "Not Implemented/Kernel Error", "quantization_time_s": "Error"})
                else:
                    print(f"Runtime error during {name} benchmark: {type(e).__name__} - {e}. Storing partial results.")
                    if name not in self.results: self.results[name] = {}
                    self.results[name].update({"error": str(e), "quantization_time_s": "Error"})
            except Exception as e: # Catch any other unexpected errors
                print(f"Unexpected error during {name} benchmark: {type(e).__name__} - {e}. Storing partial results.")
                if name not in self.results: self.results[name] = {}
                self.results[name].update({"error": str(e), "quantization_time_s": "Error"})
            finally:
                self._clear_memory() # Ensure cleanup after each benchmark attempt
                print(f"--- Finished Benchmark for: {name} ---")
                
        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Convert columns to numeric where applicable, coercing errors for "N/A", "OOM", etc.
        numeric_cols = [
            "quantization_time_s", "model_param_size_gb", "compression_ratio_params",
            "mean_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "min_latency_ms", "max_latency_ms", "throughput_inf_per_s", "throughput_tokens_per_s",
            "peak_mem_quantization_gb", "peak_mem_inference_gb", "final_mem_allocated_gb",
            "memory_efficiency_percent", "mean_gpu_utilization_percent", "peak_gpu_utilization_percent",
            "time_model_copying_s", "time_quantizer_init_s", "peak_mem_model_copy_gb"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # 'coerce' will turn non-numeric into NaN

        # Sort by a relevant metric, e.g., mean_latency_ms, handling NaNs
        if 'mean_latency_ms' in df.columns:
            df = df.sort_values(by="mean_latency_ms", ascending=True, na_position='last')
        
        return df
        
    def print_report(self):
        """
        Runs benchmarks for all configured methods and prints a formatted report.
        """
        # Run benchmarks if results are empty (excluding the 'Original' placeholder)
        if not self.results or len(self.results) <= 1:
            print("Benchmark results are empty. Running benchmarks first...")
            self.run_all_benchmarks() # This will populate self.results and return a DataFrame
        
        # Use the DataFrame from self.results, which should be populated by run_all_benchmarks
        df_report = pd.DataFrame.from_dict(self.results, orient='index')

        # Define metrics to display: {column_name: (Display Name, Format String)}
        # Order matters for display. Use a list of tuples or OrderedDict if Python < 3.7
        metrics_to_display = [
            ("quantization_time_s", "Quantization Time (s)", "{:.2f}"),
            ("model_param_size_gb", "Model Size (Params, GB)", "{:.3f}"),
            ("compression_ratio_params", "Compression Ratio (Params)", "{:.2f}x"),
            ("mean_latency_ms", "Mean Latency (ms)", "{:.2f}"),
            ("p90_latency_ms", "P90 Latency (ms)", "{:.2f}"),
            ("p95_latency_ms", "P95 Latency (ms)", "{:.2f}"),
            ("p99_latency_ms", "P99 Latency (ms)", "{:.2f}"),
            ("min_latency_ms", "Min Latency (ms)", "{:.2f}"),
            ("max_latency_ms", "Max Latency (ms)", "{:.2f}"),
            ("throughput_inf_per_s", "Throughput (inf/s)", "{:.2f}"),
            ("throughput_tokens_per_s", "Throughput (tokens/s)", "{:.2f}"),
            ("peak_mem_model_copy_gb", "Peak Mem: Model Copy (GB)", "{:.3f}"),
            ("peak_mem_quantization_gb", "Peak Mem: Quantization (GB)", "{:.3f}"),
            ("peak_mem_inference_gb", "Peak Mem: Inference (GB)", "{:.3f}"),
            ("final_mem_allocated_gb", "Final Mem Allocated (GB)", "{:.3f}"),
            ("memory_efficiency_percent", "Memory Efficiency (%)", "{:.2f}%"),
            ("mean_gpu_utilization_percent", "Mean GPU Util (%)", "{:.1f}%"),
            ("peak_gpu_utilization_percent", "Peak GPU Util (%)", "{:.1f}%"),
            ("time_model_copying_s", "Model Copy Time (s)", "{:.2f}"),
            ("time_quantizer_init_s", "Quantizer Init Time (s)", "{:.2f}"),
            ("error", "Error Status", "{}") # For displaying OOM or other errors
        ]
        
        # Convert relevant columns to numeric for consistent NaN handling
        numeric_metric_keys = [m[0] for m in metrics_to_display if m[0] != "error"]
        for col in numeric_metric_keys:
            if col in df_report.columns:
                df_report[col] = pd.to_numeric(df_report[col], errors='coerce')


        print("\n===== Quantization Benchmark Report =====")
        # Sort for display if desired, e.g., by 'mean_latency_ms'
        if 'mean_latency_ms' in df_report.columns:
            df_sorted_report = df_report.sort_values(by='mean_latency_ms', ascending=True, na_position='last')
        else:
            df_sorted_report = df_report


        for method_name in df_sorted_report.index:
            print(f"\n--- Method: {method_name} ---")
            has_error = pd.notna(df_sorted_report.loc[method_name, "error"]) and df_sorted_report.loc[method_name, "error"] != "N/A"

            for metric_key, display_name, fmt_str in metrics_to_display:
                if metric_key in df_sorted_report.columns:
                    value = df_sorted_report.loc[method_name, metric_key]
                    
                    if pd.isna(value) or value == "N/A":
                        val_str = "N/A"
                    elif metric_key == "error" and not has_error : # Don't print "Error Status: N/A"
                        continue
                    elif isinstance(value, str): # Already a string (e.g. "OOM" for error)
                        val_str = value
                    else: # Attempt to format numeric values
                        try:
                            val_str = fmt_str.format(value)
                        except (ValueError, TypeError): # Fallback if formatting fails for some reason
                            val_str = str(value)
                    print(f"  {display_name:<30}: {val_str}")
            if not has_error and "error" not in df_sorted_report.columns and "error" not in [m[0] for m in metrics_to_display if method_name == "Original (FP16/FP32 Estimate)"]:
                 pass # If no error column or no error, don't print error line

        print("\n" + "=" * 40 + "\n")
        # NVML shutdown is handled in __del__ or a dedicated close() method if added.

    def plot_comparison(self, save_path: str = None):
        """
        Generates and optionally saves comparison plots for key benchmark metrics.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="whitegrid")
        except ImportError:
            print("matplotlib and seaborn are required for plotting. Install with: pip install matplotlib seaborn")
            return
            
        if not self.results or len(self.results) <= 1: # Check if results are empty or only has baseline
             print("No benchmark data to plot. Run benchmarks first via run_all_benchmarks() or print_report().")
             return

        # Use the DataFrame from self.results for plotting
        df_plot_source = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Exclude 'Original' row for most plots as it often has 'N/A' or estimated values
        df_plot = df_plot_source.drop(index="Original (FP16/FP32 Estimate)", errors='ignore')

        # Define metrics to plot. Key: DataFrame column name, Value: Plot Title
        metrics_to_plot = {
            'mean_latency_ms': 'Mean Inference Latency (ms) [Lower is Better]',
            'throughput_inf_per_s': 'Throughput (Inferences/sec) [Higher is Better]',
            'peak_mem_inference_gb': 'Peak GPU Memory during Inference (GB) [Lower is Better]',
            'model_param_size_gb': 'Model Parameter Size (GB) [Lower is Better]',
            'compression_ratio_params': 'Compression Ratio (vs Original Params) [Higher is Better]',
            'memory_efficiency_percent': 'Memory Efficiency (%) [Higher is Better]',
            'mean_gpu_utilization_percent': 'Mean GPU Utilization (%) [Contextual]',
            'quantization_time_s': 'Quantization Time (s) [Lower is Better]'
        }
        
        valid_plot_metrics = {k: v for k, v in metrics_to_plot.items() if k in df_plot.columns}
        
        # Convert relevant columns to numeric, coercing errors, then drop rows if all plot metrics are NaN
        for col in valid_plot_metrics.keys():
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
        df_plot.dropna(subset=list(valid_plot_metrics.keys()), how='all', inplace=True)

        if df_plot.empty:
            print("No valid numeric data available for plotting after filtering methods with errors/NAs.")
            return

        num_actual_plots = len(valid_plot_metrics)
        if num_actual_plots == 0:
            print("No metrics available to plot.")
            return

        ncols = 2  # Fixed number of columns for the plot grid
        nrows = (num_actual_plots + ncols - 1) // ncols 
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        plot_idx = 0
        for metric_key, plot_title in valid_plot_metrics.items():
            if not df_plot[metric_key].isnull().all(): # Check if there's any data for this metric
                ax = axes_flat[plot_idx]
                sns.barplot(x=df_plot.index, y=metric_key, ax=ax, palette="viridis", dodge=False)
                ax.set_title(plot_title, fontsize=14)
                ax.set_xlabel("Quantization Method", fontsize=12)
                ax.set_ylabel(metric_key.split("_")[0].capitalize(), fontsize=12) # Basic Y-axis label
                ax.tick_params(axis='x', rotation=45, ha='right', labelsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                plot_idx += 1
        
        # Hide any unused subplots
        for i in range(plot_idx, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        fig.suptitle('Quantization Method Comparison', fontsize=18, y=1.03 if nrows > 1 else 1.05)
        plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=3.0) # Adjust padding
        
        if save_path:
            print(f"Saving comparison plot to {save_path}")
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close(fig) # Close the figure to free memory resources
