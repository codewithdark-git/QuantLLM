"""
QuantLLM v2.0 - GGUF Export Example

Export models to GGUF format for use with llama.cpp, Ollama, LM Studio.
No external dependencies required!
"""

from quantllm import turbo, list_quant_types

# ============================================
# Show Available Quantization Types
# ============================================
print("📦 Available quantization types:\n")
for name, desc in list_quant_types().items():
    print(f"  {name:12} - {desc}")

# ============================================
# Load Model
# ============================================
print("\n\n📦 Loading model...")
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ============================================
# Export to GGUF
# ============================================

# Option 1: Quick export (default q4_0)
print("\n🚀 Exporting to GGUF (q4_0)...")
model.export("gguf", "tinyllama-q4.gguf")

# Option 2: High quality (q8_0)
print("\n🚀 Exporting to GGUF (q8_0)...")
model.export("gguf", "tinyllama-q8.gguf", quantization="q8_0")

# Option 3: Half precision (f16)
print("\n🚀 Exporting to GGUF (f16)...")
model.export("gguf", "tinyllama-f16.gguf", quantization="f16")

# ============================================
# Using convert_to_gguf Directly
# ============================================
from quantllm import convert_to_gguf
from transformers import AutoModelForCausalLM, AutoTokenizer

print("\n🔧 Using convert_to_gguf directly...")

# Load with transformers
hf_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Convert
convert_to_gguf(
    model=hf_model,
    tokenizer=tokenizer,
    output_path="tinyllama-direct.gguf",
    quant_type="q4_0",
    verbose=True
)

print("\n✅ All exports complete!")
print("\nUse these files with:")
print("  - llama.cpp: ./main -m tinyllama-q4.gguf")
print("  - Ollama: ollama create mymodel -f Modelfile")
print("  - LM Studio: Import the .gguf file")
