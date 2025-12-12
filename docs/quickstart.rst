# Quick Start Guide

## Load Any Model

```python
from quantllm import turbo

# Auto-configured loading with optimal settings
model = turbo("meta-llama/Llama-3-8B")

# Override settings if needed
model = turbo(
    "mistralai/Mistral-7B",
    bits=4,           # Quantization bits
    max_length=4096,  # Max sequence length
)
```

## Generate Text

```python
# Simple generation
response = model.generate("Explain machine learning")
print(response)

# With parameters
response = model.generate(
    "Write a Python function",
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
)
```

## Chat Format

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

response = model.chat(messages)
print(response)
```

## Fine-Tuning

```python
# Simple - auto-configured
model.finetune("my_data.json", epochs=3)

# Your data format (JSON):
# [
#   {"instruction": "What is AI?", "output": "AI is..."},
#   {"instruction": "Explain ML", "output": "ML is..."}
# ]
```

## Export

```python
# GGUF for llama.cpp / Ollama
model.export("gguf", "model.gguf", quantization="Q4_K_M")

# ONNX
model.export("onnx", "model.onnx")

# SafeTensors
model.export("safetensors", "./my-model/")
```

## What's Next?

- See [examples/](../examples/) for more use cases
- Read [API Reference](api/) for all options
- Check [Hub Integration](api/hub) for HuggingFace
