import torch
import gc
import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Dict, Optional, Any

# Adjust import paths based on your project structure
# Assuming quantllm is in the Python path or installed
from quantllm.quant.gguf import GGUFQuantizer
from quantllm.utils.benchmark import QuantizationBenchmark

DEFAULT_MODEL_LIST = ["facebook/opt-125m", "facebook/opt-350m"]
DEFAULT_GGUF_CONFIGS = [
    {"name": "GGUF_B4_GS32_Packed", "bits": 4, "group_size": 32, "use_packed": True, "desc_act": False},
    {"name": "GGUF_B8_GS128_Packed", "bits": 8, "group_size": 128, "use_packed": True, "desc_act": False},
    {"name": "GGUF_B4_PerTensor_Packed", "bits": 4, "group_size": -1, "use_packed": True, "desc_act": False},
    {"name": "GGUF_B4_GS32_Packed_CPUOffload", "bits": 4, "group_size": 32, "use_packed": True, "desc_act": False, "cpu_offload": True},
]

def _get_dummy_calibration_data(batch_size=1, seq_len=128, vocab_size=50257, num_samples=32) -> torch.Tensor:
    """Generates random tensor for calibration data on CPU."""
    return torch.randint(0, vocab_size, (num_samples, seq_len), device='cpu')

def _load_model_and_tokenizer(model_name: str, trust_remote_code: bool = True) -> tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """Loads a Hugging Face model and tokenizer to CPU."""
    try:
        print(f"Loading model: {model_name}...")
        # Load model on CPU to manage memory before explicit placement by benchmark utility
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code).cpu()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Successfully loaded {model_name}.")
        return model.eval(), tokenizer # Ensure model is in eval mode
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

def run_gguf_benchmarks(
    model_names: Optional[List[str]] = None,
    gguf_configs: Optional[List[Dict[str, Any]]] = None,
    device: Optional[str] = None,
    num_inference_steps: int = 50,
    num_warmup_steps: int = 10,
    seq_len_for_calib: int = 128,
    num_calib_samples: int = 32
):
    """
    Runs GGUF quantization benchmarks using QuantizationBenchmark.

    Args:
        model_names (Optional[List[str]]): List of Hugging Face model names.
                                           Defaults to DEFAULT_MODEL_LIST.
        gguf_configs (Optional[List[Dict[str, Any]]]): List of GGUF configurations to test.
                                                      Each dict must include a "name" key for reporting.
                                                      Defaults to DEFAULT_GGUF_CONFIGS.
        device (Optional[str]): Device to run benchmarks on ('cuda', 'cpu'). Auto-detects if None.
        num_inference_steps (int): Number of timed inference steps.
        num_warmup_steps (int): Number of warm-up inference steps.
        seq_len_for_calib (int): Sequence length for dummy calibration data.
        num_calib_samples (int): Number of samples for dummy calibration data.
    """
    model_names = model_names if model_names else DEFAULT_MODEL_LIST
    gguf_configs = gguf_configs if gguf_configs else DEFAULT_GGUF_CONFIGS

    all_results_summary = [] # To store DataFrames from each benchmark run

    for model_name in model_names:
        print(f"\n{'='*20} Starting GGUF Benchmarks for Model: {model_name} {'='*20}")
        
        original_model, tokenizer = _load_model_and_tokenizer(model_name)
        if original_model is None or tokenizer is None:
            print(f"Skipping benchmarks for {model_name} due to loading error.")
            continue

        calibration_data = _get_dummy_calibration_data(
            vocab_size=original_model.config.vocab_size,
            seq_len=seq_len_for_calib,
            num_samples=num_calib_samples
        )

        # Initialize benchmark utility for the current model
        # The model passed to QuantizationBenchmark is the original, unquantized model.
        # QuantizationBenchmark's _copy_model method will be used internally for each quantizer run.
        benchmark_utility = QuantizationBenchmark(
            model=original_model, # Original model, kept on CPU by benchmark init
            calibration_data=calibration_data, # Kept on CPU by benchmark init
            input_shape=(1, seq_len_for_calib), # (batch_size, seq_len) for inference tests
            num_inference_steps=num_inference_steps,
            # num_warmup_steps is not an __init__ arg for QuantizationBenchmark anymore
            device=device # Benchmark utility will handle device placement
        )
        
        # Calculate original model size (parameters) in GB for efficiency metrics
        # Ensure model is on CPU for this calculation if not already guaranteed
        temp_model_cpu = original_model.to('cpu')
        original_model_size_gb = sum(
            p.numel() * p.element_size() for p in temp_model_cpu.parameters()
        ) / (1024**3)
        del temp_model_cpu
        gc.collect()


        print(f"Original model '{model_name}' parameter size: {original_model_size_gb:.3f} GB")

        for gguf_config_params in gguf_configs:
            config_name = gguf_config_params.get("name", f"GGUF_Custom_{gguf_config_params.get('bits','N')}b_GS{gguf_config_params.get('group_size','N')}")
            full_benchmark_name = f"{model_name}_{config_name}"
            
            print(f"\n--- Benchmarking GGUF Configuration: {config_name} for {model_name} ---")
            
            # Remove 'name' from args passed to quantizer, it's for reporting only
            quantizer_actual_args = {k: v for k, v in gguf_config_params.items() if k != "name"}

            try:
                # benchmark_quantizer handles copying the model, quantizing, and running inference tests
                benchmark_utility.benchmark_quantizer(
                    name=full_benchmark_name, # This name will be a key in benchmark_utility.results
                    quantizer_class=GGUFQuantizer,
                    quantizer_args=quantizer_actual_args,
                    original_model_size_gb=original_model_size_gb,
                    num_warmup_steps=num_warmup_steps # Pass num_warmup_steps here
                )
            except Exception as e:
                print(f"Error during benchmark for {full_benchmark_name}: {e}")
                # Store error in results if benchmark_quantizer didn't handle it internally
                if full_benchmark_name not in benchmark_utility.results:
                     benchmark_utility.results[full_benchmark_name] = {"error": str(e)}
                else: # If benchmark_quantizer stored partials, add/update error
                     benchmark_utility.results[full_benchmark_name]["error"] = str(e)


        # Print report for the current model after all its GGUF configs have been benchmarked
        print(f"\n--- Benchmark Report for Model: {model_name} ---")
        # benchmark_utility.print_report() will use benchmark_utility.results
        # which now contains all runs for *this specific model instance*
        benchmark_utility.print_report() 
        
        # Store the results DataFrame for this model
        # run_all_benchmarks inside print_report returns a DataFrame.
        # Here, we want the df from the current benchmark_utility instance.
        current_model_df = pd.DataFrame.from_dict(benchmark_utility.results, orient='index')
        current_model_df['model_name'] = model_name # Add model name for combined report
        all_results_summary.append(current_model_df)

        # Clean up for the current model
        del original_model, tokenizer, calibration_data, benchmark_utility
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"\n{'='*20} Finished GGUF Benchmarks for Model: {model_name} {'='*20}")

    if all_results_summary:
        final_summary_df = pd.concat(all_results_summary)
        print("\n\n===== Overall GGUF Benchmark Summary =====")
        # Re-format or select columns for the final summary if needed
        # For now, just print the concatenated DataFrame
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(final_summary_df)
    else:
        print("No benchmark results were collected.")


def main():
    parser = argparse.ArgumentParser(description="Run GGUF Quantization Benchmarks.")
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=DEFAULT_MODEL_LIST,
        help="List of Hugging Face model names to benchmark."
    )
    # Configs are defined in code for now, could be loaded from JSON/YAML in future
    parser.add_argument(
        "--device",
        type=str,
        default=None, # Auto-detect
        help="Device to run benchmarks on (e.g., 'cuda', 'cuda:0', 'cpu')."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps for latency/throughput measurement."
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps before timed inference."
    )
    parser.add_argument(
        "--seq_len_calib",
        type=int,
        default=128,
        help="Sequence length for dummy calibration data."
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=32,
        help="Number of samples for dummy calibration data."
    )

    args = parser.parse_args()

    print("Starting GGUF Benchmark Suite...")
    print(f"Models to benchmark: {args.model_names}")
    print(f"GGUF Configurations defined in code: {[c['name'] for c in DEFAULT_GGUF_CONFIGS]}")
    print(f"Device: {'Auto-detect' if args.device is None else args.device}")
    print(f"Inference Steps: {args.num_inference_steps}, Warm-up Steps: {args.num_warmup_steps}")

    run_gguf_benchmarks(
        model_names=args.model_names,
        gguf_configs=DEFAULT_GGUF_CONFIGS, # Using the hardcoded default configs
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        num_warmup_steps=args.num_warmup_steps,
        seq_len_for_calib=args.seq_len_calib,
        num_calib_samples=args.num_calib_samples
    )

    print("\nGGUF Benchmark Suite Finished.")

if __name__ == "__main__":
    main()
