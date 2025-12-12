# Quick Start

Get up and running with QuantLLM in 5 minutes.

## Basic Usage

### Load a Model

```python
from quantllm import turbo

# Load any HuggingFace model with auto-quantization
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

The `turbo()` function automatically:
- Detects your hardware (GPU memory, CUDA version)
- Chooses optimal quantization (4-bit, 8-bit, etc.)
- Enables Flash Attention if available
- Configures memory settings

### Generate Text

```python
response = model.generate(
    "Explain quantum computing in simple terms.",
    max_new_tokens=100,
    temperature=0.7,
)
print(response)
```

### Chat Mode

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

response = model.chat(messages, max_new_tokens=100)
print(response)
```

### Streaming Output

```python
for token in model.generate("Count to 10:", stream=True):
    print(token, end="", flush=True)
```

## Export to GGUF

Export your model for use with llama.cpp, Ollama, or LM Studio:

```python
# Export with 4-bit quantization
model.export("gguf", "my-model.gguf", quantization="q4_0")

# Higher quality 8-bit
model.export("gguf", "my-model-q8.gguf", quantization="q8_0")
```

**No llama.cpp installation required!** QuantLLM uses pure Python for GGUF conversion.

## Fine-tuning

Train your model with LoRA:

```python
training_data = [
    {"text": "Question: What is AI?\nAnswer: Artificial Intelligence."},
    {"text": "Question: What is ML?\nAnswer: Machine Learning."},
]

result = model.finetune(
    data=training_data,
    epochs=3,
    lora_r=8,
)
```

## Configuration

View or customize the auto-detected configuration:

```python
# Print configuration summary
model.config.print_summary()

# Manual configuration
from quantllm import turbo

model = turbo(
    "meta-llama/Llama-2-7b",
    bits=4,                    # Force 4-bit quantization
    max_seq_length=4096,       # Context length
    device="cuda:0",           # Specific GPU
)
```

## Next Steps

- [Loading Models](guide/loading-models.md) - Advanced model loading options
- [GGUF Export](guide/gguf-export.md) - All quantization types
- [Fine-tuning](guide/finetuning.md) - Training guide
- [API Reference](api/turbomodel.md) - Full API documentation
