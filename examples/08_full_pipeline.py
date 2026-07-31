"""
QuantLLM v2.2 -- Full Pipeline

Complete workflow: load -> generate -> chat -> export -> push.
Each step uses real package APIs with no hardcoded data.
"""

import os

from quantllm import turbo

# Shared configuration
CONFIG = {
    "format": "gguf",
    "quantization": "Q4_K_M",
    "output_dir": "./output",
    "push_format": "gguf",
}

# ------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------
print("1. Loading TinyLlama with shared config...")
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0", config=CONFIG)
print(f"   bits={model.config.bits}, device={model.config.device}")
print(f"   quant={model.config.quant_type}, group={model.config.group_size}")

# ------------------------------------------------------------------
# 2. Generate
# ------------------------------------------------------------------
print("\n2. Generating text...")
resp = model.generate("What is the difference between AI and ML?", max_new_tokens=100, temperature=0.7)
print(f"   {resp[:120]}...")

# ------------------------------------------------------------------
# 3. Chat
# ------------------------------------------------------------------
print("\n3. Chat...")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain Python in one sentence."},
]
resp = model.chat(messages, max_tokens=100)
print(f"   {resp[:120]}...")

# ------------------------------------------------------------------
# 4. Export
# ------------------------------------------------------------------
print("\n4. Exporting (uses shared config)...")
path = model.export()
print(f"   {path}")

# Also export ONNX for cross-format comparison
print("\n   Exporting ONNX...")
model2 = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0", config={"format": "onnx", "quantization": "Q4_K_M"})
path2 = model2.export()
print(f"   {path2}")

# ------------------------------------------------------------------
# 5. Push (optional, requires HF_TOKEN)
# ------------------------------------------------------------------
token = os.environ.get("HF_TOKEN")
if token:
    print("\n5. Pushing to HuggingFace Hub...")
    model.push(
        "YOUR_USERNAME/my-quantized-tinyllama-gguf",
        license="apache-2.0",
        token=token,
    )
    print("   https://huggingface.co/YOUR_USERNAME/my-quantized-tinyllama-gguf")
else:
    print("\n5. Push skipped (set HF_TOKEN to enable).")

# ------------------------------------------------------------------
# 6. Serve (run separately: quantllm serve TinyLlama/... --port 8080)
# ------------------------------------------------------------------
print("\n6. To serve:")
print("   quantllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8080")

print("\nPipeline complete.")
