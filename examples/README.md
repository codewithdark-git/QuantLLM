# QuantLLM v2.0 Examples

## Quick Start

```python
from quantllm import turbo

# Load any model in one line
model = turbo("meta-llama/Llama-3-8B")

# Generate text
response = model.generate("Explain quantum computing")
print(response)

# Chat format
response = model.chat([
    {"role": "user", "content": "Hello!"}
])
```

## Fine-Tuning

```python
from quantllm import turbo

model = turbo("mistralai/Mistral-7B")

# One-line fine-tuning with auto-configuration
model.finetune("my_data.json", epochs=3)

# Or with custom settings
model.finetune(
    "my_data.json",
    epochs=5,
    learning_rate=2e-4,
    lora_r=32,
)
```

## Export to GGUF

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3-8B")

# Export for llama.cpp / Ollama
model.export("gguf", "llama3-q4.gguf", quantization="Q4_K_M")
```

## Hub Integration

```python
from quantllm.hub import QuantLLMHubManager

manager = QuantLLMHubManager(
    repo_id="username/my-model",
    hf_token="your_token",
    auto_push=False
)

# Track training
manager.track_hyperparameters({"lr": 0.001, "epochs": 10})

for epoch in range(10):
    # ... training code ...
    manager.log_metrics({"loss": loss}, step=epoch)
    manager.save_checkpoint(model, optimizer, epoch=epoch)

# Push to Hub
manager.push(push_final_model=True)
```

## Memory Optimization

```python
from quantllm.core import setup_memory_efficient_training

# Auto-configure for memory-efficient training
components = setup_memory_efficient_training(
    model,
    gradient_checkpointing=True,
    cpu_offload_optimizer=True,
)

optimizer = components['optimizer']
```

## Available Examples

| File | Description |
|------|-------------|
| `turbo_quickstart.py` | Basic turbo API usage |
| `hub_example.py` | HuggingFace Hub integration |
| `finetune_example.py` | Fine-tuning with LoRA |
| `export_example.py` | Export to multiple formats |
| `benchmark_example.py` | Performance benchmarking |

## More Information

See the [documentation](../docs/) for complete API reference.