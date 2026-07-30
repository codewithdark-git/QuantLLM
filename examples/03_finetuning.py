"""
QuantLLM v2.2 -- Fine-tuning with LoRA

Fine-tune a model on a small custom dataset, test the result,
then export to GGUF.
"""

from quantllm import turbo

# Sample training data (instruction format)
training_data = [
    {
        "text": (
            "### Instruction:\n"
            "What is Python?\n\n"
            "### Response:\n"
            "Python is a high-level, interpreted programming language."
        )
    },
    {
        "text": (
            "### Instruction:\n"
            "Explain AI.\n\n"
            "### Response:\n"
            "AI is artificial intelligence -- machines that can learn and reason."
        )
    },
    {
        "text": (
            "### Instruction:\n"
            "What is machine learning?\n\n"
            "### Response:\n"
            "ML is a subset of AI where systems learn patterns from data."
        )
    },
]

# Load model
print("Loading TinyLlama...")
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Fine-tune
print("\nFine-tuning (1 epoch, 3 samples)...")
result = model.finetune(
    data=training_data,
    epochs=1,
    batch_size=2,
    learning_rate=2e-4,
    lora_r=8,
    lora_alpha=16,
    output_dir="./finetuned_model",
)
print(f"  train_loss: {result['train_loss']:.4f}")
print(f"  output_dir: {result['output_dir']}")

# Test
print("\nTesting fine-tuned model...")
response = model.generate("What is Python?", max_new_tokens=50)
print(f"  {response}")

# Export
print("\nExporting to GGUF...")
path = model.export("gguf", "finetuned-q4.gguf")
print(f"  {path}")

print("\nFine-tuning complete.")
