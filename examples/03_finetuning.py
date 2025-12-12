"""
QuantLLM v2.0 - Fine-tuning Example

Fine-tune a quantized model using LoRA.
"""

from quantllm import turbo

# ============================================
# 1. Load Model
# ============================================
print("📦 Loading model...")
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0", bits=4)

# ============================================
# 2. Prepare Training Data
# ============================================
# Simple format: list of dicts with "text" key
training_data = [
    {"text": "### Instruction:\nWhat is Python?\n\n### Response:\nPython is a high-level programming language."},
    {"text": "### Instruction:\nExplain AI.\n\n### Response:\nAI is artificial intelligence, machines that can learn."},
    {"text": "### Instruction:\nWhat is machine learning?\n\n### Response:\nML is a subset of AI where systems learn from data."},
    # Add more examples...
]

# Or use a HuggingFace dataset
# training_data = "tatsu-lab/alpaca"

# ============================================
# 3. Fine-tune
# ============================================
print("\n🎯 Starting fine-tuning...")

result = model.finetune(
    data=training_data,
    epochs=1,
    batch_size=2,
    learning_rate=2e-4,
    lora_r=8,
    lora_alpha=16,
    output_dir="./finetuned_model",
)

print(f"\n✅ Training complete!")
print(f"   Loss: {result['train_loss']:.4f}")
print(f"   Output: {result['output_dir']}")

# ============================================
# 4. Test the Fine-tuned Model
# ============================================
print("\n📝 Testing fine-tuned model...")

response = model.generate(
    "What is Python?",
    max_new_tokens=50
)
print(f"Response: {response}")

# ============================================
# 5. Export Fine-tuned Model
# ============================================
print("\n📦 Exporting to GGUF...")
model.export("gguf", "finetuned-q4.gguf")

print("\n✅ Done!")
