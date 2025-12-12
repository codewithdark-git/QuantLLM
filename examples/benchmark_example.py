"""Example demonstrating benchmarking functionality of QuantLLM."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantllm.utils.benchmark import QuantizationBenchmark
from quantllm.utils.logger import logger


def main():
    # Load model and tokenizer
    model_name = "facebook/opt-125m"
    logger.log_info(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare calibration data
    calibration_text = ["This is an example text for benchmarking."] * 10
    calibration_data = tokenizer(calibration_text, return_tensors="pt", padding=True)["input_ids"]
    
    try:
        # Initialize benchmark
        device = "cuda" if torch.cuda.is_available() else "cpu"
        benchmark = QuantizationBenchmark(
            model=model,
            calibration_data=calibration_data,
            input_shape=(1, 32),
            num_inference_steps=100,
            device=device,
            num_warmup_steps=10
        )
        
        # Run benchmarks for different configurations
        logger.log_info("Running benchmarks...")
        results_df = benchmark.run_all_benchmarks()
        
        # Print detailed report
        logger.log_info("\nBenchmark Report:")
        benchmark.print_report()
        
        # Generate and save visualization
        logger.log_info("\nGenerating benchmark visualization...")
        benchmark.plot_comparison(save_path="benchmark_results.png")
        logger.log_success("Benchmark visualization saved to benchmark_results.png")
        
        # Save detailed results to CSV
        results_df.to_csv("benchmark_results.csv")
        logger.log_success("Detailed benchmark results saved to benchmark_results.csv")
        
    except Exception as e:
        logger.log_error(f"Benchmarking failed: {str(e)}")

if __name__ == "__main__":
    main() 