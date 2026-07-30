"""
QuantLLM v2.2 -- Benchmark

Run a live benchmark on a loaded model and display the results.
Uses BenchmarkRunner from the package to collect real metrics.
"""

from quantllm import turbo
from quantllm.benchmark import BenchmarkRunner, format_report_table

# Load model
print("Loading TinyLlama...")
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(f"  bits: {model.config.bits}, quant: {model.config.quant_type}")

# Create runner and benchmark
print("\nBenchmarking (warmup=3, decode_runs=5, each 128 tokens)...")
runner = BenchmarkRunner(verbose=True)
result = runner.benchmark_model(
    model.model,
    model.tokenizer,
    model.config,
    prompt="The meaning of life is",
    max_new_tokens=128,
)

# Show results
print(f"\n  Model:          {result.model}")
print(f"  Backend:        {result.backend}")
print(f"  Bits:           {result.bits}")
print(f"  Prefill:        {result.prefill_tokens_per_sec:.1f} tok/s")
print(f"  Decode:         {result.decode_tokens_per_sec:.1f} tok/s")
print(f"  Latency:        {result.total_latency_ms:.1f} ms")
print(f"  Peak VRAM:      {result.peak_vram_gb:.2f} GB")
print(f"  Parameters:     {result.num_params_billion:.2f}B")
print(f"  Model size:     {result.model_size_gb:.2f} GB")

# Full comparison across bit-widths
print("\nRunning comparison (FP16, BnB-8bit, BnB-4bit)...")
print("  (This loads the model three times, may take a while)")
try:
    from quantllm.benchmark import BenchmarkComparer
    comparer = BenchmarkComparer(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        verbose=True,
    )
    results = comparer.run_comparison(
        prompt="The meaning of life is",
        max_new_tokens=64,
    )
    print("\nComparison report:")
    print(comparer.generate_comparison_report())
except Exception as exc:
    print(f"  Comparison skipped: {exc}")

print("\nBenchmark complete.")
