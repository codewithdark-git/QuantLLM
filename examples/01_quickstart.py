"""
QuantLLM v2.2 -- Quick Start

Load a model, generate text, chat, stream, and export.
This is the simplest way to verify the package works.
"""

from quantllm import turbo

# Load a model with automatic 4-bit quantization
print("Loading TinyLlama...")
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(f"  device: {model.config.device}, dtype: {model.config.dtype}")
print(f"  bits: {model.config.bits}, quant: {model.config.quant_type}")

# Generate text
print("\nGenerating...")
response = model.generate(
    "Explain quantum computing in simple terms.",
    max_new_tokens=100,
    temperature=0.7,
)
print(f"  {response}")

# Chat
print("\nChat...")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]
response = model.chat(messages, max_new_tokens=100)
print(f"  {response}")

# Streaming
print("\nStreaming...")
for token in model.generate(
    "Count from 1 to 5:", stream=True, max_new_tokens=30
):
    print(token, end="", flush=True)
print()

# Export
print("\nExporting to GGUF (Q4_K_M)...")
path = model.export("gguf", "tinyllama-q4.gguf", quantization="Q4_K_M")
print(f"  Exported to {path}")

print("\nQuick start complete.")
