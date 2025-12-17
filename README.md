<div align="center">
  <img src="docs/images/logo.png" alt="QuantLLM Logo" width="200"/>
  <h1>🧠 QuantLLM v2.0</h1>
  <p align="center">
    <img src="https://img.shields.io/badge/QuantLLM-v2.0-blue?style=for-the-badge" alt="QuantLLM v2.0"/>
    <br/>
    <strong>🚀 One Line to Rule Them All</strong>
  </p>

  <p align="center">
    <a href="https://pepy.tech/projects/quantllm"><img src="https://static.pepy.tech/badge/quantllm" alt="Downloads"/></a>
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/quantllm?logo=pypi&label=version"/>
    <img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue.svg"/>
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"/>
    <img alt="Stars" src="https://img.shields.io/github/stars/codewithdark-git/QuantLLM?style=social"/>
  </p>

  <p align="center">
    <b>Load → Quantize → Fine-tune → Export</b> · Any LLM · One Line Each
  </p>
  
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-features">Features</a> •
    <a href="#-examples">Examples</a> •
    <a href="#-supported-models">Models</a> •
    <a href="#-documentation">Docs</a>
  </p>
</div>

---

## 🤔 Why QuantLLM?

| Challenge | Without QuantLLM | With QuantLLM |
|-----------|------------------|---------------|
| **Loading 7B model** | 50+ lines of config | `turbo("model")` |
| **Quantization setup** | Complex BitsAndBytes config | Automatic |
| **Fine-tuning** | LoRA config + Trainer setup | `model.finetune(data)` |
| **GGUF export** | Manual llama.cpp workflow | `model.export("gguf")` |
| **Memory management** | Manual offloading code | Built-in |

**QuantLLM handles the complexity so you can focus on building.**

---

## ⚡ Quick Start

### Installation

```bash
# From GitHub (recommended)
pip install git+https://github.com/codewithdark-git/QuantLLM.git

# With all features
pip install "quantllm[full] @ git+https://github.com/codewithdark-git/QuantLLM.git"
```

### Your First Model in 3 Lines

```python
from quantllm import turbo

# Load with automatic 4-bit quantization, Flash Attention, optimal settings
model = turbo("meta-llama/Llama-3-8B")

# Generate text
print(model.generate("Explain quantum computing in simple terms"))
```

That's it. QuantLLM automatically:
- ✅ Detects your GPU and memory
- ✅ Chooses optimal quantization (4-bit on most GPUs)
- ✅ Enables Flash Attention 2 if available
- ✅ Configures batch size and memory management

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Ultra-Simple API
```python
# One line - everything automatic
model = turbo("mistralai/Mistral-7B")

# Override if needed
model = turbo("Qwen/Qwen2-7B", bits=4, max_length=8192)
```

</td>
<td width="50%">

### ⚡ Speed Optimizations
- **Triton Kernels** - Fused dequant+matmul
- **torch.compile** - Graph optimization
- **Flash Attention 2** - Fast attention
- **Weight Caching** - No re-dequantization

</td>
</tr>
<tr>
<td>

### 🧠 45+ Model Architectures
Llama 2/3, Mistral, Mixtral, Qwen/Qwen2, Phi-1/2/3, Gemma, Falcon, GPT-NeoX, StableLM, ChatGLM, Yi, DeepSeek, InternLM, Baichuan, StarCoder, BLOOM, OPT, MPT...

</td>
<td>

### 📦 6 Export Formats
- **GGUF** - Optimized export via llama.cpp integration
- **ONNX** - ONNX Runtime, TensorRT
- **SafeTensors** - HuggingFace
- **MLX** - Apple Silicon
- **AWQ** - AutoAWQ
- **PyTorch** - Standard .pt

</td>
</tr>
<tr>
<td>

### 🔧 Zero-Config Smart Defaults
- **SmartConfig Stats Panel** (See size before loading)
- Hardware auto-detection & optimization
- Automatic quantization selection
- Memory-aware loading

</td>
<td>

### 💾 Memory Optimizations
- **Dynamic Padding** (Efficient training)
- **OOM Prevention** (Expandable segments)
- Dynamic CPU ↔ GPU offloading
- Gradient checkpointing
- CPU optimizer states

</td>
</tr>
</table>

---

## 🎮 Usage Examples

### Chat with Any Model

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3-8B")

# Simple generation
response = model.generate(
    "Write a Python function to calculate fibonacci numbers",
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

### Load GGUF Models
```python
# Load a GGUF model directly from HuggingFace
model = turbo.from_gguf(
    "TheBloke/Llama-2-7B-Chat-GGUF", 
    filename="llama-2-7b-chat.Q4_K_M.gguf"
)

print(model.generate("Hello!"))
```

### Fine-Tune with Your Data

```python
from quantllm import turbo

model = turbo("mistralai/Mistral-7B")

# Simple - everything auto-configured
model.finetune("training_data.json", epochs=3)

# With Auto-Tracking to Hub
from quantllm import QuantLLMHubManager
manager = QuantLLMHubManager("user/repo", "hf_token")

# Automatically tracks params (epochs, lr, etc) to manager
model.finetune("training_data.json", hub_manager=manager)

# Advanced - full control
model.finetune(
    "training_data.json",
    epochs=5,
    learning_rate=2e-4,
    lora_r=32,
    lora_alpha=64,
    batch_size=4,
    output_dir="./fine-tuned-model",
)
```

**Supported data formats:**
```json
[
  {"instruction": "What is Python?", "output": "Python is a programming language..."},
  {"text": "Full text for language modeling"},
  {"prompt": "Question here", "completion": "Answer here"}
]
```

### Export to Multiple Formats

```python
from quantllm import turbo

model = turbo("microsoft/phi-3-mini")

# GGUF for llama.cpp / Ollama / LM Studio
model.export("gguf", "phi3-q4.gguf", quantization="Q4_K_M")

# GGUF quantization options:
# Q2_K, Q3_K_S, Q3_K_M, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0

# ONNX for TensorRT / ONNX Runtime
model.export("onnx", "phi3.onnx")

# SafeTensors for HuggingFace
model.export("safetensors", "./phi3-hf/")

# MLX for Apple Silicon Macs
model.export("mlx", "./phi3-mlx/", quantization="4bit")
```

### Push to HuggingFace Hub

#### Method 1: The One-Liner (Recommended)
```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3-8B")

# Standard Push
model.push("username/my-model", token="hf_...")

# GGUF Push (Auto-Export + Upload)
model.push("username/my-gguf-model", format="gguf", quantization="Q4_K_M")
```

#### Method 2: Advanced Control
```python
from quantllm import turbo, QuantLLMHubManager

model = turbo("microsoft/phi-2")
manager = QuantLLMHubManager("username/my-repo", "hf_token")

# Auto-train & track params
model.finetune("my_data.json", hub_manager=manager)

# Manual tracking
manager.track_hyperparameters({"custom_metric": 0.95})

# Save & Push
manager.save_final_model(model.model)
manager.push()
```

---

## 🧠 Supported Models

QuantLLM supports **45+ model architectures** out of the box:

| Category | Models |
|----------|--------|
| **Llama Family** | Llama 2, Llama 3, CodeLlama |
| **Mistral Family** | Mistral 7B, Mixtral 8x7B |
| **Qwen Family** | Qwen, Qwen2, Qwen2-MoE |
| **Microsoft** | Phi-1, Phi-2, Phi-3 |
| **Google** | Gemma, Gemma 2 |
| **Falcon** | Falcon 7B/40B/180B |
| **Code Models** | StarCoder, StarCoder2, CodeGen |
| **Chinese** | ChatGLM, Yi, Baichuan, InternLM |
| **Other** | DeepSeek, StableLM, MPT, BLOOM, OPT, GPT-NeoX |

---

## 📦 Installation Options

```bash
# Basic installation
pip install git+https://github.com/codewithdark-git/QuantLLM.git

# With GGUF export support
pip install "quantllm[gguf] @ git+https://github.com/codewithdark-git/QuantLLM.git"

# With Triton kernels (Linux only)
pip install "quantllm[triton] @ git+https://github.com/codewithdark-git/QuantLLM.git"

# With Flash Attention
pip install "quantllm[flash] @ git+https://github.com/codewithdark-git/QuantLLM.git"

# Full installation (all features)
pip install "quantllm[full] @ git+https://github.com/codewithdark-git/QuantLLM.git"

# Hub lifecycle (for HuggingFace integration)
pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git
```

---

## 💻 Hardware Requirements

| Configuration | RAM | GPU VRAM | Recommended For |
|---------------|-----|----------|-----------------|
| 🔵 **Entry GPU** | 16GB | 6-8GB | 1-7B models (4-bit) |
| 🟣 **Mid-Range** | 32GB | 12-24GB | 13B-30B models |
| 🟠 **High-End** | 64GB+ | 24-80GB | 70B+ models |

### Tested GPUs
- NVIDIA: RTX 3060, 3070, 3080, 3090, 4070, 4080, 4090, A100, H100
- AMD: RX 7900 XTX (with ROCm)
- Apple: M1, M2, M3 (via MLX export)

---

## 📚 Documentation

| Resource | Description |
|----------|-------------|
| 📖 [Examples](./examples/) | Working code examples |
| 📚 [API Reference](https://quantllm.readthedocs.io/en/latest/) | Full API documentation |
| 🐛 [Issues](https://github.com/codewithdark-git/QuantLLM/issues) | Report bugs |

---

## 🏗️ Architecture

```
quantllm/
├── core/                    # Core functionality
│   ├── turbo_model.py      # Main TurboModel API
│   ├── smart_config.py     # Auto-configuration
│   ├── hardware.py         # Hardware detection
│   ├── compilation.py      # torch.compile integration
│   ├── flash_attention.py  # Flash Attention 2
│   ├── memory.py           # Memory optimization
│   ├── training.py         # Training utilities
│   └── export.py           # Universal exporter
├── kernels/                # Custom kernels
│   └── triton/             # Triton fused kernels
├── quant/                  # Quantization
│   ├── gguf_converter.py   # GGUF export (45 models)
│   └── quantization_engine.py
├── hub/                    # HuggingFace integration
│   └── hf_manager.py       # Lifecycle management
└── utils/                  # Utilities
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

```bash
# Clone the repository
git clone https://github.com/codewithdark-git/QuantLLM.git
cd QuantLLM

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black quantllm/
isort quantllm/
```

### Areas for Contribution
- 🆕 New model architecture support
- 🔧 Performance optimizations
- 📚 Documentation improvements
- 🐛 Bug fixes
- ✨ New export formats

---

## 📈 Benchmarks

Coming soon! We're working on comprehensive benchmarks comparing:
- Inference speed vs vanilla transformers
- Memory usage comparisons
- Quantization quality metrics
- Export format performance

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p>
    <strong>Made with ❤️ by <a href="https://github.com/codewithdark-git">Dark Coder</a></strong>
  </p>
  <p>
    <a href="https://github.com/codewithdark-git/QuantLLM">⭐ Star us on GitHub</a> •
    <a href="https://github.com/sponsors/codewithdark-git">💖 Sponsor</a>
  </p>
</div>
