"""
QuantLLM v2.1 - Push to HuggingFace Hub

Push your models to HuggingFace Hub with auto-generated model cards.
"""

from quantllm import turbo

# ============================================
# 1. Load and Prepare Model
# ============================================
print("📦 Loading model...")
model = turbo(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    bits=4,
    config={"format": "gguf", "quantization": "Q4_K_M", "push_format": "gguf"},
)

# ============================================
# 2. Push to Hub (GGUF format)
# ============================================
print("\n🚀 Pushing GGUF to HuggingFace Hub...")

# Uses shared config — format and quantization from config={}
model.push(
    "YOUR_USERNAME/my-quantized-tinyllama-gguf",
    license="apache-2.0",
    # token="hf_..."  # Or set HF_TOKEN env var
)

print("\n✅ GGUF pushed to Hub!")

# ============================================
# 3. Push SafeTensors format
# ============================================
print("\n📦 Pushing SafeTensors to HuggingFace Hub...")

model.push(
    "YOUR_USERNAME/my-quantized-tinyllama",
    format="safetensors",
    license="apache-2.0",
)

print("\n✅ SafeTensors pushed to Hub!")
print("   Visit: https://huggingface.co/YOUR_USERNAME/my-quantized-tinyllama")
