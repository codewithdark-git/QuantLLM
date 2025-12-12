"""
QuantLLM v2.0 - Quick Start Example

The simplest way to use QuantLLM.
"""

from quantllm import turbo

# ============================================
# 1. Load a Model (One Line!)
# ============================================
print("Loading model...")
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Show configuration
model.config.print_summary()

# ============================================
# 2. Generate Text
# ============================================
print("\n📝 Generating text...")

response = model.generate(
    "Explain quantum computing in simple terms.",
    max_new_tokens=100,
    temperature=0.7,
)

print(f"\nResponse:\n{response}")

# ============================================
# 3. Chat Mode
# ============================================
print("\n💬 Chat mode...")

messages = [
    {"role": "user", "content": "What is Python?"},
]

response = model.chat(messages, max_new_tokens=100)
print(f"\nChat response:\n{response}")

# ============================================
# 4. Streaming Generation
# ============================================
print("\n🔄 Streaming generation...")

print("Response: ", end="", flush=True)
for token in model.generate("Count from 1 to 10:", stream=True, max_new_tokens=50):
    print(token, end="", flush=True)
print("\n")

print("✅ Quick start complete!")
