"""
QuantLLM v2.0 - Push to HuggingFace Hub

Push your models to HuggingFace Hub.
"""

from quantllm import turbo, HubManager

# ============================================
# 1. Load and Prepare Model
# ============================================
print("📦 Loading model...")
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0", bits=4)

# ============================================
# 2. Save Locally First
# ============================================
print("\n💾 Saving model locally...")
model.export("safetensors", "./my_quantized_model/")

# ============================================
# 3. Push to Hub
# ============================================
print("\n🚀 Pushing to HuggingFace Hub...")

hub = HubManager(token="YOUR_HF_TOKEN")  # Or set HF_TOKEN env var

# Push the saved model
hub.push_model(
    model_path="./my_quantized_model/",
    repo_name="my-quantized-tinyllama",
    private=False,
    commit_message="Upload quantized model via QuantLLM"
)

print("\n✅ Model pushed to Hub!")
print("   Visit: https://huggingface.co/YOUR_USERNAME/my-quantized-tinyllama")

# ============================================
# Alternative: Push GGUF File
# ============================================
print("\n📦 Creating and pushing GGUF...")

# Export to GGUF
model.export("gguf", "tinyllama-q4.gguf")

# Push GGUF file
hub.push_model(
    model_path="tinyllama-q4.gguf",
    repo_name="my-gguf-model",
    private=False
)

print("\n✅ GGUF pushed to Hub!")
