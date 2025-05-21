"""Benchmarking utilities for quantization methods."""

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
    """Benchmark different quantization methods."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        calibration_data: torch.Tensor,
        input_shape: Tuple[int, ...] = (1, 32),
        num_inference_steps: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.calibration_data = calibration_data
        self.input_shape = input_shape
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.results = {}
        
    def benchmark_quantizer(
        self,
        name: str,
        quantizer_class,
        quantizer_args: Dict
    ) -> Dict[str, float]:
        """Benchmark a specific quantizer."""
        try:
            # Initialize quantizer
            quantizer = quantizer_class(model=self.model.clone(), **quantizer_args)
            
            # Measure quantization time
            start_time = time.time()
            quantized_model = quantizer.quantize(calibration_data=self.calibration_data)
            quant_time = time.time() - start_time
            
            # Move to appropriate device
            quantized_model = quantized_model.to(self.device)
            
            # Generate test input
            test_input = torch.randint(
                0, 1000,
                self.input_shape,
                device=self.device
            )
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    quantized_model(test_input)
            torch.cuda.synchronize() if self.device == "cuda" else None
            
            # Measure inference latency
            latencies = []
            for _ in range(self.num_inference_steps):
                start = time.perf_counter()
                with torch.no_grad():
                    quantized_model(test_input)
                torch.cuda.synchronize() if self.device == "cuda" else None
                latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms
                
            latencies = torch.tensor(latencies)
            
            # Calculate memory usage
            if self.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                memory_allocated = 0
                peak_memory = 0
                
            # Calculate model size
            model_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)  # MB
            
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
            
            self.results[name] = results
            return results
            
        except Exception as e:
            print(f"Error benchmarking {name}: {str(e)}")
            return {}
            
    def run_all_benchmarks(self) -> pd.DataFrame:
        """Run benchmarks for all quantization methods."""
        # Common config
        config = {
            "bits": 4,
            "group_size": 128
        }
        
        # GPTQ
        self.benchmark_quantizer(
            "GPTQ",
            GPTQQuantizer,
            {**config, "actorder": True, "use_triton": False}
        )
        
        # AWQ
        self.benchmark_quantizer(
            "AWQ",
            AWQQuantizer,
            {**config, "zero_point": True}
        )
        
        # GGUF
        self.benchmark_quantizer(
            "GGUF",
            GGUFQuantizer,
            {**config, "use_packed": True}
        )
        
        # Convert results to DataFrame
        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Add compression ratio
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        df['compression_ratio'] = original_size / df['model_size']
        
        return df
        
    def print_report(self):
        """Print a formatted benchmark report."""
        df = self.run_all_benchmarks()
        
        print("\nQuantization Benchmark Results")
        print("=" * 80)
        
        # Format metrics
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
                value = df.loc[method, metric]
                print(f"{name:<30} {fmt.format(value)}")
                
    def plot_comparison(self, save_path: str = None):
        """Generate comparison plots."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting")
            return
            
        df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Quantization Method Comparison')
        
        # Latency comparison
        axes[0, 0].bar(df.index, df['mean_latency'])
        axes[0, 0].set_title('Mean Inference Latency (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage
        axes[0, 1].bar(df.index, df['memory_allocated'])
        axes[0, 1].set_title('Memory Usage (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Model size
        axes[1, 0].bar(df.index, df['model_size'])
        axes[1, 0].set_title('Model Size (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Quantization time
        axes[1, 1].bar(df.index, df['quantization_time'])
        axes[1, 1].set_title('Quantization Time (s)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
