"""Memory-efficient benchmarking utilities for quantization methods."""

import gc
import time
import torch
import pandas as pd
from typing import Dict, List, Tuple
from transformers import PreTrainedModel
from quantllm.quant import (
    GPTQQuantizer,
    AWQQuantizer,
    GGUFQuantizer
)

class QuantizationBenchmark:
    """Memory-efficient benchmark implementation for quantization methods."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        calibration_data: torch.Tensor,
        input_shape: Tuple[int, ...] = (1, 32),
        num_inference_steps: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize benchmark with memory management."""
        self.model = model.cpu()  # Keep model on CPU initially
        self.calibration_data = calibration_data.cpu()  # Keep data on CPU
        self.input_shape = input_shape
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.results = {}
        
        # Set memory-efficient options
        if torch.cuda.is_available():
            torch.backends.cuda.max_split_size_mb = 128
            torch.backends.cudnn.benchmark = True
            
    def _clear_memory(self):
        """Clear GPU memory and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def benchmark_quantizer(
        self,
        name: str,
        quantizer_class,
        quantizer_args: Dict
    ) -> Dict[str, float]:
        """Benchmark a specific quantizer with memory management."""
        results = {}
        try:
            self._clear_memory()
            
            # Configure quantizer for memory efficiency
            mem_efficient_args = dict(quantizer_args)
            if name == "AWQ":
                mem_efficient_args.update({
                    "group_size": min(mem_efficient_args.get("group_size", 128), 64),
                })
            elif name == "GPTQ":
                mem_efficient_args.update({
                    "percdamp": 0.01,
                    "block_size": 128,
                })
            
            # Initialize quantizer with model on CPU
            model_clone = self.model.clone()
            quantizer = quantizer_class(model=model_clone, **mem_efficient_args)
            
            # Move to device for quantization
            if self.device == "cuda":
                quantizer.model = quantizer.model.cuda()
                cal_data = self.calibration_data.cuda()
            else:
                cal_data = self.calibration_data
                
            # Measure quantization time
            start_time = time.time()
            
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
                
            quant_time = time.time() - start_time
            
            # Move calibration data back to CPU and clear GPU
            if self.device == "cuda":
                cal_data = cal_data.cpu()
                self._clear_memory()
            
            # Prepare for inference benchmark
            quantized_model = quantized_model.to(self.device)
            test_input = torch.randint(
                0, 1000,
                (1, min(self.input_shape[1], 32)),
                device=self.device
            )
            
            # Limited warmup
            for _ in range(5):
                with torch.no_grad():
                    quantized_model(test_input)
                if self.device == "cuda":
                    torch.cuda.synchronize()
            
            # Measure inference latency
            latencies = []
            for _ in range(self.num_inference_steps):
                start = time.perf_counter()
                with torch.no_grad():
                    quantized_model(test_input)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)  # ms
                
            latencies = torch.tensor(latencies)
            
            # Calculate memory usage
            if self.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            else:
                memory_allocated = peak_memory = 0
                
            # Calculate model size
            model_size = sum(
                p.numel() * p.element_size() 
                for p in quantized_model.parameters()
            ) / 1024**2  # MB
            
            results = {
                "quantization_time": quant_time,
                "mean_latency": latencies.mean().item(),
                "p95_latency": torch.quantile(latencies, 0.95).item(),
                "min_latency": latencies.min().item(),
                "max_latency": latencies.max().item(),
                "memory_allocated": memory_allocated,
                "peak_memory": peak_memory,
                "model_size": model_size
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
        """Print formatted benchmark report."""
        df = self.run_all_benchmarks()
        
        print("\nQuantization Benchmark Results")
        print("=" * 80)
        
        metrics = {
            'quantization_time': ('Quantization Time (s)', '{:.2f}'),
            'mean_latency': ('Mean Inference Latency (ms)', '{:.2f}'),
            'p95_latency': ('P95 Inference Latency (ms)', '{:.2f}'),
            'memory_allocated': ('Memory Used (MB)', '{:.1f}'),
            'model_size': ('Model Size (MB)', '{:.1f}'),
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
        """Generate comparison plots with memory usage."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return
            
        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Quantization Method Comparison')
        
        # Latency comparison
        df['mean_latency'].plot(
            kind='bar', ax=axes[0, 0], rot=45,
            title='Mean Inference Latency (ms)'
        )
        
        # Memory usage
        df['memory_allocated'].plot(
            kind='bar', ax=axes[0, 1], rot=45,
            title='Memory Usage (MB)'
        )
        
        # Model size
        df['model_size'].plot(
            kind='bar', ax=axes[1, 0], rot=45,
            title='Model Size (MB)'
        )
        
        # Compression ratio
        if 'compression_ratio' in df.columns:
            df['compression_ratio'].plot(
                kind='bar', ax=axes[1, 1], rot=45,
                title='Compression Ratio'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()
