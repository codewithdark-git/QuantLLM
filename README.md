<div align="center">
  <img src="docs/images/logo.png" alt="QuantLLM Logo" width="180"/>
  
  # 🚀 QuantLLM v2.0
  
  <p align="center">
    <strong>The Ultra-Fast LLM Quantization & Export Library</strong>
  </p>

  <p align="center">
    <a href="https://pepy.tech/projects/quantllm"><img src="https://static.pepy.tech/badge/quantllm" alt="Downloads"/></a>
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/quantllm?logo=pypi&label=version&color=orange"/>
    <img alt="Python" src="https://img.shields.io/badge/python-3.10+-orange.svg"/>
    <img alt="License" src="https://img.shields.io/badge/license-MIT-orange.svg"/>
    <img alt="Stars" src="https://img.shields.io/github/stars/codewithdark-git/QuantLLM?style=social"/>
  </p>

  <p align="center">
    <b>Load → Quantize → Fine-tune → Export</b> — All in One Line
  </p>
  
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-features">Features</a> •
    <a href="#-export-formats">Export Formats</a> •
    <a href="#-examples">Examples</a> •
    <a href="https://quantllm.readthedocs.io">Documentation</a>
  </p>
</div>

---

## 🎯 Why QuantLLM?

<table>
<tr>
<td width="50%">

### ❌ Without QuantLLM
```python
# 50+ lines of configuration...
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
    # ... more config
)
# Then llama.cpp compilation for GGUF...
# Then manual tensor conversion...
```

</td>
<td width="50%">

### ✅ With QuantLLM
```python
from quantllm import turbo

# One line does everything
model = turbo("meta-llama/Llama-3-8B")

# Generate
print(model.generate("Hello!"))

# Fine-tune
model.finetune(dataset, epochs=3)

# Export to any format
model.export("gguf", quantization="Q4_K_M")
```

</td>
</tr>
</table>

---

## ⚡ Quick Start

### Installation

```bash
# Recommended installation
pip install git+https://github.com/codewithdark-git/QuantLLM.git

# With all export formats
pip install "quantllm[full] @ git+https://github.com/codewithdark-git/QuantLLM.git"
```

### Your First Model

```python
from quantllm import turbo

# Load any model with automatic optimization
model = turbo("meta-llama/Llama-3.2-3B")

# Generate text
response = model.generate("Explain quantum computing simply")
print(response)

# Export to GGUF for Ollama/llama.cpp
model.export("gguf", "model.Q4_K_M.gguf", quantization="Q4_K_M")
```

**QuantLLM automatically:**
- ✅ Detects your GPU and available memory
- ✅ Applies optimal 4-bit quantization
- ✅ Enables Flash Attention 2 when available
- ✅ Configures memory management

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔥 TurboModel API
```python
# One unified API for everything
model = turbo("mistralai/Mistral-7B")
model.generate("Hello!")
model.finetune(data, epochs=3)
model.export("gguf", quantization="Q4_K_M")
model.push("user/repo", format="gguf")
```

</td>
<td width="50%">

### ⚡ Performance
- **Flash Attention 2** — Auto-enabled
- **torch.compile** — 2x faster training
- **Dynamic Padding** — 50% less VRAM
- **Triton Kernels** — Fused operations

</td>
</tr>
<tr>
<td>

### 🧠 45+ Model Architectures
Llama 2/3, Mistral, Mixtral, Qwen 1/2, Phi 1/2/3, Gemma, Falcon, DeepSeek, Yi, StarCoder, ChatGLM, InternLM, Baichuan, StableLM, BLOOM, OPT, MPT, GPT-NeoX...

</td>
<td>

### 📦 Multi-Format Export
- **GGUF** — llama.cpp, Ollama, LM Studio
- **ONNX** — ONNX Runtime, TensorRT
- **MLX** — Apple Silicon (M1/M2/M3/M4)
- **SafeTensors** — HuggingFace

</td>
</tr>
<tr>
<td>

### 🎨 Beautiful UI
```
╔════════════════════════════════════╗
║  🚀 QuantLLM v2.0                  ║
║  ✓ GGUF  ✓ ONNX  ✓ MLX             ║
╚════════════════════════════════════╝

📊 Model: meta-llama/Llama-3.2-3B
   Parameters: 3.21B
   Memory: 6.4 GB → 1.9 GB (70% saved)
```

</td>
<td>

### 🤗 One-Click Hub Publishing
```python
# Auto-generates model cards with:
# - YAML frontmatter
# - Usage examples  
# - "Use this model" button

model.push("user/my-model", format="gguf")
```

</td>
</tr>
</table>

---

## 📦 Export Formats

Export to any deployment target with a single line:

```python
from quantllm import turbo

model = turbo("microsoft/phi-3-mini")

# GGUF — For llama.cpp, Ollama, LM Studio
model.export("gguf", "model.Q4_K_M.gguf", quantization="Q4_K_M")

# ONNX — For ONNX Runtime, TensorRT  
model.export("onnx", "./model-onnx/")

# MLX — For Apple Silicon Macs
model.export("mlx", "./model-mlx/", quantization="4bit")

# SafeTensors — For HuggingFace
model.export("safetensors", "./model-hf/")
```

### GGUF Quantization Types

| Type | Bits | Quality | Use Case |
|------|------|---------|----------|
| `Q2_K` | 2-bit | Low | Minimum size |
| `Q3_K_M` | 3-bit | Fair | Very constrained |
| `Q4_K_M` | 4-bit | Good | **Recommended** ⭐ |
| `Q5_K_M` | 5-bit | High | Quality-focused |
| `Q6_K` | 6-bit | Very High | Near-original |
| `Q8_0` | 8-bit | Excellent | Best quality |

---

## 🎮 Examples

### Chat with Any Model

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3.2-3B")

# Simple generation
response = model.generate(
    "Write a Python function for fibonacci",
    max_new_tokens=200,
    temperature=0.7,
)
print(response)

# Chat format
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I read a file in Python?"},
]
response = model.chat(messages)
print(response)
```

### Load GGUF Models from HuggingFace

```python
from quantllm import TurboModel

# Load any GGUF model directly
model = TurboModel.from_gguf(
    "TheBloke/Llama-2-7B-Chat-GGUF", 
    filename="llama-2-7b-chat.Q4_K_M.gguf"
)

print(model.generate("Hello!"))
```

### Fine-Tune with Your Data

```python
from quantllm import turbo

model = turbo("mistralai/Mistral-7B")

# Simple — everything auto-configured
model.finetune("training_data.json", epochs=3)

# Advanced — full control
model.finetune(
    "training_data.json",
    epochs=5,
    learning_rate=2e-4,
    lora_r=32,
    lora_alpha=64,
    batch_size=4,
)
```

**Supported data formats:**
```json
[
  {"instruction": "What is Python?", "output": "Python is..."},
  {"text": "Full text for language modeling"},
  {"prompt": "Question", "completion": "Answer"}
]
```

### Push to HuggingFace Hub

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3.2-3B")

# Push with auto-generated model card
model.push(
    "your-username/my-model",
    format="gguf",
    quantization="Q4_K_M",
    license="apache-2.0"
)
```

The model card includes:
- ✅ Proper YAML frontmatter (`library_name`, `tags`, `base_model`)
- ✅ Format-specific usage examples
- ✅ "Use this model" button compatibility
- ✅ Quantization details

---

## 💻 Hardware Requirements

| Configuration | GPU VRAM | Models |
|---------------|----------|--------|
| 🟢 **Entry** | 6-8 GB | 1-7B (4-bit) |
| 🟡 **Mid-Range** | 12-24 GB | 7-30B (4-bit) |
| 🔴 **High-End** | 24-80 GB | 70B+ |

**Tested GPUs:** RTX 3060/3070/3080/3090/4070/4080/4090, A100, H100, Apple M1/M2/M3/M4

---

## 📦 Installation Options

```bash
# Basic installation
pip install git+https://github.com/codewithdark-git/QuantLLM.git

# With specific features
pip install "quantllm[gguf]"     # GGUF export
pip install "quantllm[onnx]"     # ONNX export  
pip install "quantllm[mlx]"      # MLX export (Apple Silicon)
pip install "quantllm[triton]"   # Triton kernels
pip install "quantllm[full]"     # Everything
```

---

## 🏗️ Architecture

```
quantllm/
├── core/                    # Core functionality
│   ├── turbo_model.py      # TurboModel unified API
│   ├── smart_config.py     # Auto-configuration
│   └── export.py           # Universal exporter
├── quant/                   # Quantization
│   └── llama_cpp.py        # GGUF conversion
├── hub/                     # HuggingFace integration
│   ├── hub_manager.py      # Push/pull models
│   └── model_card.py       # Auto model cards
├── kernels/                 # Custom kernels
│   └── triton/             # Fused operations
└── utils/                   # Utilities
    └── progress.py         # Beautiful UI
```

---

## 🤝 Contributing

```bash
git clone https://github.com/codewithdark-git/QuantLLM.git
cd QuantLLM
pip install -e ".[dev]"
pytest
```

**Areas for contribution:**
- 🆕 New model architectures
- 🔧 Performance optimizations
- 📚 Documentation
- 🐛 Bug fixes

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  
  ### Made with 🧡 by [Dark Coder](https://github.com/codewithdark-git)

  <a href="https://github.com/codewithdark-git/QuantLLM">⭐ Star on GitHub</a> •
  <a href="https://github.com/codewithdark-git/QuantLLM/issues">🐛 Report Bug</a> •
  <a href="https://github.com/sponsors/codewithdark-git">💖 Sponsor</a>

  **Happy Quantizing! 🚀**

</div>
