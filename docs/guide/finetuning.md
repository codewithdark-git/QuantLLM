# Fine-tuning

Train your quantized model using LoRA (Low-Rank Adaptation).

## Quick Start

```python
from quantllm import turbo

model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0", bits=4)

training_data = [
    {"text": "Q: What is AI? A: Artificial Intelligence."},
    {"text": "Q: What is ML? A: Machine Learning."},
]

result = model.finetune(data=training_data, epochs=3)
```

## Data Formats

### Simple Text

```python
data = [
    {"text": "Your training text here..."},
    {"text": "Another example..."},
]
```

### Instruction Format

```python
data = [
    {
        "instruction": "Explain Python",
        "output": "Python is a programming language..."
    },
]
```

### HuggingFace Dataset

```python
# From Hub
result = model.finetune(data="tatsu-lab/alpaca", epochs=1)

# From local file
result = model.finetune(data="./my_data.json", epochs=1)
```

## Training Parameters

```python
result = model.finetune(
    data=training_data,
    epochs=3,                   # Training epochs
    batch_size=4,               # Batch size
    learning_rate=2e-4,         # Learning rate
    lora_r=8,                   # LoRA rank
    lora_alpha=16,              # LoRA alpha
    output_dir="./output",      # Save directory
)
```

## LoRA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 8 | Rank of LoRA matrices |
| `lora_alpha` | 16 | LoRA scaling factor |
| Typical ratio | α = 2×r | Common practice |

Higher `r` = more parameters = better quality but slower.

## After Training

```python
# Test the model
response = model.generate("What is Python?")
print(response)

# Export to GGUF
model.export("gguf", "finetuned.gguf")

# Save to HuggingFace format
model.export("safetensors", "./my-finetuned-model/")
```

## Tips

1. **Start small**: Use small datasets and few epochs first
2. **Monitor loss**: Training loss should decrease
3. **Learning rate**: 1e-4 to 3e-4 works well for most cases
4. **LoRA rank**: 8-16 is usually sufficient
5. **Batch size**: Reduce if you run out of memory
