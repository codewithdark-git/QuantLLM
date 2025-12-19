# 🎓 Fine-Tuning

Train your model on custom data using LoRA (Low-Rank Adaptation).

---

## Quick Start

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3.2-3B")

# Fine-tune with your data
model.finetune("training_data.json", epochs=3)

# Test the result
response = model.generate("Your custom prompt")
print(response)
```

---

## Data Formats

### Instruction Format (Recommended)

Best for Q&A and task-oriented training:

```json
[
  {
    "instruction": "What is Python?",
    "output": "Python is a high-level programming language known for its simplicity and readability."
  },
  {
    "instruction": "Explain machine learning",
    "output": "Machine learning is a subset of AI that enables systems to learn from data."
  }
]
```

### Simple Text Format

For language modeling and general text:

```json
[
  {"text": "This is the first training example. It can be any text."},
  {"text": "This is another example for training the model."}
]
```

### Prompt-Completion Format

Alternative to instruction format:

```json
[
  {
    "prompt": "Question: What is AI?\nAnswer:",
    "completion": "AI stands for Artificial Intelligence."
  }
]
```

### HuggingFace Datasets

Load directly from HuggingFace:

```python
# From Hub
model.finetune("tatsu-lab/alpaca", epochs=1)

# Or use datasets library
from datasets import load_dataset
dataset = load_dataset("your-dataset")
model.finetune(dataset, epochs=3)
```

---

## Training Parameters

### Basic Training

```python
model.finetune(
    "training_data.json",
    epochs=3,                    # Number of training epochs
    batch_size=4,                # Batch size (reduce if OOM)
    learning_rate=2e-4,          # Learning rate
    output_dir="./output",       # Save directory
)
```

### Advanced Training

```python
model.finetune(
    "training_data.json",
    epochs=5,
    batch_size=4,
    learning_rate=2e-4,
    
    # LoRA parameters
    lora_r=16,                   # LoRA rank (higher = more capacity)
    lora_alpha=32,               # LoRA scaling (typically 2x lora_r)
    lora_dropout=0.1,            # Dropout for regularization
    
    # Training options
    warmup_steps=100,            # Learning rate warmup
    max_steps=-1,                # Max steps (-1 for full epochs)
    gradient_accumulation=4,     # Accumulate gradients
    
    # Output
    output_dir="./finetuned",
    save_steps=500,              # Save checkpoint every N steps
    logging_steps=10,            # Log every N steps
)
```

---

## LoRA Configuration

LoRA (Low-Rank Adaptation) enables efficient fine-tuning:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 8 | Rank of LoRA matrices (4, 8, 16, 32) |
| `lora_alpha` | 16 | LoRA scaling factor (typically 2×r) |
| `lora_dropout` | 0.1 | Dropout for regularization |

### Choosing LoRA Rank

| Rank | Parameters | Use Case |
|------|------------|----------|
| 4 | Minimal | Simple adaptations |
| 8 | Low | **Default, good balance** |
| 16 | Medium | More complex tasks |
| 32 | High | Maximum quality |

**Rule of thumb**: Higher rank = more parameters = better quality but slower training.

---

## Training with Hub Integration

Track your training and push to HuggingFace:

```python
from quantllm import turbo, QuantLLMHubManager

model = turbo("meta-llama/Llama-3.2-3B")

# Create hub manager
manager = QuantLLMHubManager(
    repo_id="your-username/finetuned-model",
    hf_token="hf_..."
)

# Train with automatic tracking
model.finetune(
    "training_data.json",
    epochs=3,
    hub_manager=manager  # Automatically tracks hyperparameters
)

# Push the result
manager.save_final_model(model)
manager.push()
```

---

## After Training

### Test Your Model

```python
# Generate with fine-tuned model
response = model.generate("Your custom prompt")
print(response)

# Compare responses
original = turbo("meta-llama/Llama-3.2-3B")
print("Original:", original.generate("prompt"))
print("Fine-tuned:", model.generate("prompt"))
```

### Export the Model

```python
# Export to GGUF
model.export("gguf", "finetuned.Q4_K_M.gguf", quantization="Q4_K_M")

# Export to SafeTensors
model.export("safetensors", "./finetuned-model/")

# Push to HuggingFace
model.push("your-username/finetuned-model", format="gguf")
```

### Save and Load

```python
# Save locally
model.save("./my-finetuned-model/")

# Load later
from quantllm import TurboModel
model = TurboModel.from_pretrained("./my-finetuned-model/")
```

---

## Tips & Best Practices

### Data Quality

1. **Clean your data** — Remove duplicates, errors, and noise
2. **Consistent format** — Use the same format throughout
3. **Balanced dataset** — Mix different types of examples
4. **Minimum 100 examples** — More is generally better

### Training Settings

1. **Start small** — Use few epochs and small data first
2. **Monitor loss** — Training loss should decrease steadily
3. **Learning rate** — 1e-4 to 3e-4 works for most cases
4. **Batch size** — Reduce if you run out of memory

### Memory Management

```python
# If you run out of memory:
model.finetune(
    data,
    batch_size=1,                  # Smaller batch
    gradient_accumulation=8,       # Accumulate gradients
)
```

### Avoiding Overfitting

1. **Limit epochs** — 1-5 epochs is usually enough
2. **Use dropout** — `lora_dropout=0.1`
3. **Validate** — Test on held-out data
4. **Early stopping** — Stop when validation loss increases

---

## Common Issues

### Out of Memory

```python
# Solution 1: Reduce batch size
model.finetune(data, batch_size=1)

# Solution 2: Use gradient accumulation
model.finetune(data, batch_size=1, gradient_accumulation=8)

# Solution 3: Use smaller LoRA rank
model.finetune(data, lora_r=4)
```

### Training Loss Not Decreasing

```python
# Try higher learning rate
model.finetune(data, learning_rate=3e-4)

# Or more epochs
model.finetune(data, epochs=10)
```

### Model Outputs Garbage

- Check your data format
- Reduce epochs (overfitting)
- Use lower learning rate

---

## Next Steps

- [GGUF Export →](gguf-export.md) — Export your fine-tuned model
- [Hub Integration →](hub-integration.md) — Push to HuggingFace
- [API Reference →](../api/turbomodel.md) — Full API documentation
