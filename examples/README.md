# QuantLLM v2.0 Examples

Simple examples for the new TurboModel API.

## Quick Start

```python
from quantllm import turbo

# Load model with auto-quantization
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generate text
response = model.generate("What is Python?")
print(response)

# Export to GGUF (no llama.cpp needed!)
model.export("gguf", "model.gguf", quantization="q4_0")
```

## Examples

| File | Description |
|------|-------------|
| `01_quickstart.py` | Basic loading and generation |
| `02_gguf_export.py` | Export to GGUF format |
| `03_finetuning.py` | Fine-tune with LoRA |
| `04_hub_push.py` | Push to HuggingFace Hub |

## Requirements

```bash
pip install quantllm torch transformers
```
