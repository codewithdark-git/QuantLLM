"""
QuantLLM v2.1 - GGUF Export Example

Export models to GGUF format for use with llama.cpp, Ollama, LM Studio.
No external dependencies required!
"""

from quantllm import turbo, GGUF_QUANT_TYPES, QUANT_RECOMMENDATIONS

# ============================================
# Show Available Quantization Types
# ============================================
print("📦 Available quantization types:\n")
for qt in GGUF_QUANT_TYPES:
    print(f"  {qt}")

print("\n📦 Recommended quantization types:\n")
for use_case, qt in QUANT_RECOMMENDATIONS.items():
    print(f"  {use_case:12} → {qt}")

# ============================================
# Load Model
# ============================================
print("\n\n📦 Loading model...")
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ============================================
# Export to GGUF
# ============================================

# Option 1: Quick export (default Q4_K_M)
print("\n🚀 Exporting to GGUF (Q4_K_M)...")
model.export("gguf", "tinyllama-q4.gguf", quantization="Q4_K_M")

# Option 2: High quality (Q8_0)
print("\n🚀 Exporting to GGUF (Q8_0)...")
model.export("gguf", "tinyllama-q8.gguf", quantization="Q8_0")

# Option 3: Half precision (F16)
print("\n🚀 Exporting to GGUF (F16)...")
model.export("gguf", "tinyllama-f16.gguf", quantization="F16")

print("\n✅ All exports complete!")
print("\nUse these files with:")
print("  - llama.cpp: ./llama-cli -m tinyllama-q4.gguf -p 'Hello!'")
print("  - Ollama: ollama create mymodel -f Modelfile")
print("  - LM Studio: Import the .gguf file")
