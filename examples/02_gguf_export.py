"""
QuantLLM v2.2 -- GGUF Export

Export a model to GGUF format at multiple quantization levels.
All levels come from the package's GGUF_QUANT_TYPES constant.
"""

from quantllm import turbo, GGUF_QUANT_TYPES, QUANT_RECOMMENDATIONS

# Show what the package supports
print(f"Available quantisation types: {len(GGUF_QUANT_TYPES)}")
for qt in GGUF_QUANT_TYPES:
    print(f"  {qt}")

print("\nRecommended by use case:")
for use_case, qt in QUANT_RECOMMENDATIONS.items():
    print(f"  {use_case:12} -> {qt}")

# Load model
print("\nLoading TinyLlama...")
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Export at multiple levels
for quant in ("Q4_K_M", "Q8_0", "F16"):
    filename = f"tinyllama-{quant.lower()}.gguf"
    print(f"\nExporting {quant} -> {filename} ...")
    path = model.export("gguf", filename, quantization=quant)
    print(f"  {path}")

# Export using shared config (auto-naming)
print("\nExport using shared config...")
model2 = turbo(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    config={"format": "gguf", "quantization": "Q4_K_M", "output_dir": "./exports"},
)
path = model2.export()
print(f"  {path}")

print("\nAll exports complete.")
