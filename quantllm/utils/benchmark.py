"""Memory-efficient benchmarking utilities for GGUF quantization methods only."""

import gc
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from ..quant import GGUFQuantizer
from ..quant.quantization_engine import DeviceManager
from ..utils.logger import logger

class QuantizationBenchmark:
    """Memory-efficient benchmark implementation for GGUF quantization only."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        calibration_data: torch.Tensor,
        input_shape: Tuple[int, ...] = (1, 32),
        num_inference_steps: int = 100,
        device: Optional[Union[str, torch.device]] = None,
        num_warmup_steps: int = 10
    ):
        self.device_manager = DeviceManager(torch.device(device) if device else None)
        self.model = model.to("cpu")
        self.calibration_data = calibration_data.to("cpu")
        self.input_shape = input_shape
        self.num_inference_steps = num_inference_steps
        self.num_warmup_steps = num_warmup_steps
        self.results = {}
        self.pynvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml_available = True
            self.nvml_handle = None
            logger.log_info("NVML initialized successfully. GPU monitoring available.")
        except Exception as e:
            logger.log_warning(f"NVML initialization failed: {e}. GPU monitoring unavailable.")
        if torch.cuda.is_available():
            torch.backends.cuda.max_split_size_mb = 128
            torch.backends.cudnn.benchmark = True
    
    def _get_gpu_utilization(self) -> Optional[float]:
        if self.pynvml_available and self.nvml_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                return util.gpu
            except Exception:
                return None
        return None

    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _copy_model(self) -> PreTrainedModel:
        try:
            logger.log_info("Creating new model instance...")
            config = AutoConfig.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)
            new_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to("cpu")
            logger.log_info("Copying model parameters to CPU...")
            with torch.no_grad():
                state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
                new_model.load_state_dict(state_dict, strict=True)
                del state_dict
            return new_model
        except Exception as e:
            logger.log_error(f"Model copy failed: {str(e)}")
            raise

    def _get_model_stats(self, model: PreTrainedModel) -> Dict[str, float]:
        """Get detailed model statistics."""
        stats = {}
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate memory usage
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        stats.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": (param_size + buffer_size) / (1024 * 1024),
            "param_size_mb": param_size / (1024 * 1024),
            "buffer_size_mb": buffer_size / (1024 * 1024)
        })
        
        return stats

    def benchmark_quantizer(
        self,
        name: str,
        quantizer_args: Dict,
        original_model_size_gb: float
    ) -> Dict[str, Union[float, str]]:
        current_results: Dict[str, Union[float, str]] = {}
        
        try:
            self._clear_memory()
            logger.log_info(f"\n{'='*20} Benchmarking {name} {'='*20}")
            
            # Get original model stats
            original_stats = self._get_model_stats(self.model)
            logger.log_info("\nOriginal Model Statistics:")
            logger.log_info(f"Total Parameters: {original_stats['total_parameters']:,}")
            logger.log_info(f"Model Size: {original_stats['model_size_mb']:.2f} MB")
            
            # Copy and quantize model
            time_start = time.perf_counter()
            model_copy = self._copy_model()
            model_copy = model_copy.to(self.device_manager.primary_device)
            
            # Initialize quantizer
            quantizer = GGUFQuantizer(
                model_name=model_copy,
                device=self.device_manager.primary_device,
                **quantizer_args
            )
            
            # Quantize model
            logger.log_info(f"\nQuantizing model with {quantizer_args.get('bits', 4)} bits...")
            cal_data_device = self.calibration_data.to(self.device_manager.primary_device)
            quantized_model = quantizer.quantize(calibration_data=cal_data_device)
            quantization_time = time.perf_counter() - time_start
            
            # Get quantized model stats
            quantized_stats = self._get_model_stats(quantized_model)
            compression_ratio = original_stats['model_size_mb'] / quantized_stats['model_size_mb']
            
            logger.log_info("\nQuantized Model Statistics:")
            logger.log_info(f"Total Parameters: {quantized_stats['total_parameters']:,}")
            logger.log_info(f"Model Size: {quantized_stats['model_size_mb']:.2f} MB")
            logger.log_info(f"Compression Ratio: {compression_ratio:.2f}x")
            logger.log_info(f"Quantization Time: {quantization_time:.2f} seconds")
            
            # Run inference benchmarks
            logger.log_info("\nRunning inference benchmarks...")
            quantized_model = quantized_model.to(self.device_manager.primary_device)
            test_input = torch.randint(
                0, quantized_model.config.vocab_size,
                (1, self.input_shape[1]),
                device=self.device_manager.primary_device
            )
            
            # Warmup
            logger.log_info(f"Running {self.num_warmup_steps} warmup steps...")
            for _ in range(self.num_warmup_steps):
                with torch.no_grad():
                    _ = quantized_model(test_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Benchmark inference
            logger.log_info(f"Running {self.num_inference_steps} inference steps...")
            latencies_ms = []
            gpu_utilizations = []
            
            for _ in range(self.num_inference_steps):
                step_start_time = time.perf_counter()
                with torch.no_grad():
                    _ = quantized_model(test_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - step_start_time) * 1000)
                
                gpu_util = self._get_gpu_utilization()
                if gpu_util is not None:
                    gpu_utilizations.append(gpu_util)
            
            # Calculate metrics
            latencies_tensor = torch.tensor(latencies_ms)
            current_results.update({
                "total_parameters": quantized_stats['total_parameters'],
                "model_size_mb": quantized_stats['model_size_mb'],
                "compression_ratio": compression_ratio,
                "quantization_time_s": quantization_time,
                "mean_latency_ms": latencies_tensor.mean().item(),
                "p90_latency_ms": torch.quantile(latencies_tensor, 0.90).item(),
                "p95_latency_ms": torch.quantile(latencies_tensor, 0.95).item(),
                "p99_latency_ms": torch.quantile(latencies_tensor, 0.99).item(),
                "throughput_tokens_per_s": (self.num_inference_steps * self.input_shape[1]) / quantization_time
            })
            
            if gpu_utilizations:
                current_results.update({
                    "mean_gpu_utilization_percent": np.mean(gpu_utilizations),
                    "peak_gpu_utilization_percent": np.max(gpu_utilizations)
                })
            
            logger.log_info("\nBenchmark Results:")
            logger.log_info(f"Mean Latency: {current_results['mean_latency_ms']:.2f} ms")
            logger.log_info(f"P90 Latency: {current_results['p90_latency_ms']:.2f} ms")
            logger.log_info(f"Throughput: {current_results['throughput_tokens_per_s']:.2f} tokens/s")
            if gpu_utilizations:
                logger.log_info(f"Mean GPU Utilization: {current_results['mean_gpu_utilization_percent']:.1f}%")
            
            logger.log_info(f"\n{'='*60}\n")
            
            self.results[name] = current_results
            return current_results
            
        except Exception as e:
            logger.log_error(f"Benchmark failed for {name}: {str(e)}")
            return {"error": str(e), **current_results}
        finally:
            self._clear_memory()

    def run_all_benchmarks(self) -> pd.DataFrame:
        original_model_size_gb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3)
        self.results["Original (FP16/FP32)"] = {
            "model_param_size_gb": original_model_size_gb,
            "compression_ratio_params": 1.0,
            "memory_efficiency_percent": 0.0,
            "mean_latency_ms": "N/A",
            "throughput_inf_per_s": "N/A",
            "peak_mem_inference_gb": original_model_size_gb
        }
        gguf_configs = [
            ("GGUF_Q2_K", {"bits": 2, "group_size": 32, "quant_type": "Q2_K"}),
            ("GGUF_Q4_K_M", {"bits": 4, "group_size": 32, "quant_type": "Q4_K_M"}),
            ("GGUF_Q5_K_M", {"bits": 5, "group_size": 32, "quant_type": "Q5_K_M"}),
            ("GGUF_Q6_K", {"bits": 6, "group_size": 32, "quant_type": "Q6_K"}),
            ("GGUF_Q8_0", {"bits": 8, "group_size": -1, "quant_type": "Q8_0"})
        ]
        for name, config in gguf_configs:
            try:
                logger.log_info(f"\nBenchmarking {name}...")
                self.benchmark_quantizer(
                    name=name,
                    quantizer_args=config,
                    original_model_size_gb=original_model_size_gb
                )
            except Exception as e:
                logger.log_error(f"Failed to benchmark {name}: {str(e)}")
                self.results[name] = {"error": str(e)}
        df = pd.DataFrame.from_dict(self.results, orient='index')
        numeric_cols = [
            'model_param_size_gb', 'mean_latency_ms', 'throughput_inf_per_s',
            'peak_mem_inference_gb', 'compression_ratio_params', 'memory_efficiency_percent',
            'quantization_time_s', 'p90_latency_ms', 'p95_latency_ms', 'p99_latency_ms',
            'peak_mem_model_copy_gb', 'peak_mem_quantization_gb', 'final_mem_allocated_gb',
            'throughput_tokens_per_s'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.sort_values(by='mean_latency_ms', ascending=True, na_position='last')

    def print_report(self):
        if not self.results:
            self.run_all_benchmarks()
        
        print("\n===== GGUF Quantization Benchmark Summary =====")
        
        for method, results in self.results.items():
            print(f"\n{method}:")
            print("-" * 40)
            
            if "error" in results:
                print(f"Error: {results['error']}")
                continue
                
            metrics = [
                ("Total Parameters", results.get("total_parameters", "N/A"), "{:,}"),
                ("Model Size", results.get("model_size_mb", "N/A"), "{:.2f} MB"),
                ("Compression Ratio", results.get("compression_ratio", "N/A"), "{:.2f}x"),
                ("Mean Latency", results.get("mean_latency_ms", "N/A"), "{:.2f} ms"),
                ("P90 Latency", results.get("p90_latency_ms", "N/A"), "{:.2f} ms"),
                ("Throughput", results.get("throughput_tokens_per_s", "N/A"), "{:.2f} tokens/s"),
                ("GPU Utilization", results.get("mean_gpu_utilization_percent", "N/A"), "{:.1f}%")
            ]
            
            for metric, value, fmt in metrics:
                if value != "N/A":
                    try:
                        print(f"{metric:.<30} {fmt.format(value)}")
                    except (ValueError, TypeError):
                        print(f"{metric:.<30} {value}")
                else:
                    print(f"{metric:.<30} N/A")

    def plot_comparison(self, save_path: Optional[str] = None):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="whitegrid")
        except ImportError:
            logger.log_error("matplotlib and seaborn are required for plotting")
            return
        if not self.results:
            self.run_all_benchmarks()
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df = df.drop(index="Original (FP16/FP32)", errors='ignore')
        metrics_to_plot = {
            'mean_latency_ms': 'Mean Latency (ms)',
            'throughput_inf_per_s': 'Throughput (inf/s)',
            'peak_mem_inference_gb': 'Peak Memory (GB)',
            'compression_ratio_params': 'Compression Ratio',
            'memory_efficiency_percent': 'Memory Efficiency (%)'
        }
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        axes = axes.flatten()
        for i, (metric, title) in enumerate(metrics_to_plot.items()):
            if metric in df.columns:
                sns.barplot(x=df.index, y=df[metric], ax=axes[i])
                axes[i].set_title(title)
                axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def __del__(self):
        if self.pynvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
                logger.log_info("NVML shutdown complete")
            except Exception as e:
                logger.log_warning(f"NVML shutdown failed: {e}")
