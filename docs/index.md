# 🚀 QuantLLM Documentation

<div align="center">
  <strong>The Ultra-Fast LLM Quantization & Export Library</strong>
  <br/>
  <em>Load → Quantize → Fine-tune → Export — All in One Line</em>
</div>

---

## Welcome to QuantLLM v2.1 (pre-release)

QuantLLM makes working with large language models simple. Load any model, quantize it automatically, fine-tune with your data, and export to any format — all with just a few lines of code.

```python
from quantllm import turbo

# Load with shared export/push defaults
model = turbo(
    "meta-llama/Llama-3.2-3B",
    config={"format": "gguf", "quantization": "Q4_K_M", "push_format": "gguf"},
)

# Generate text
print(model.generate("Explain quantum computing"))

# Export to GGUF for Ollama/llama.cpp
model.export()

# Push to HuggingFace with auto-generated model card
model.push("username/my-model")
```

---

## 📚 Documentation

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
guide/finetuning
guide/gguf-export
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

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔥 **TurboModel API** | One unified interface for everything |
| 📦 **Multi-Format Export** | GGUF, ONNX, MLX, SafeTensors |
| ⚡ **Auto-Optimization** | Flash Attention, torch.compile, dynamic padding |
| 🎨 **Beautiful UI** | Orange-themed progress bars and logging |
| 🤗 **Hub Integration** | One-click push with auto model cards |
| 🧠 **45+ Architectures** | Llama, Mistral, Qwen, Phi, Gemma, and more |

---

## 🚀 Quick Examples

### Load Any Model
```python
from quantllm import turbo

model = turbo("mistralai/Mistral-7B")
model = turbo("Qwen/Qwen2-7B", bits=4)
model = turbo("microsoft/phi-3-mini")
```

### Export to Any Format
```python
model = turbo(
    "meta-llama/Llama-3.2-3B",
    config={"format": "gguf", "quantization": "Q4_K_M", "push_format": "gguf"},
)
model.export()
model.export("onnx", "./model-onnx/")
model.export("mlx", "./model-mlx/", quantization="4bit")
```

### Fine-tune in One Line
```python
model.finetune("training_data.json", epochs=3)
```

### Push to HuggingFace
```python
model.push("username/my-model")
```

---

## 💻 System Requirements

- **Python**: 3.10+
- **PyTorch**: 2.0+
- **GPU**: NVIDIA with 6GB+ VRAM (recommended)
- **Platforms**: Windows, Linux, macOS

---

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
