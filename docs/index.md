# QuantLLM Documentation

Welcome to QuantLLM - Ultra-fast LLM Quantization with Pure Python GGUF Export.

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guide/loading-models
guide/generation
guide/gguf-export
guide/finetuning
guide/hub-integration
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/turbo
api/turbomodel
api/gguf
api/hub
```

## Quick Example

```python
from quantllm import turbo

# Load model with auto-quantization
model = turbo("meta-llama/Llama-2-7b")

# Generate text
response = model.generate("What is AI?")
print(response)

# Export to GGUF (no llama.cpp needed!)
model.export("gguf", "llama2.gguf", quantization="q4_0")
```

## Key Features

- **One-Line Loading**: `turbo("model-name")` handles everything
- **Pure Python GGUF**: No llama.cpp compilation required
- **Auto-Configuration**: Detects your hardware and optimizes settings
- **Easy Fine-tuning**: LoRA training with `model.finetune(data)`
- **Multiple Exports**: GGUF, SafeTensors, ONNX formats

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
