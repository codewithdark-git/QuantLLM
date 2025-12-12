# Installation

## Basic Installation

Install QuantLLM from GitHub:

```bash
pip install git+https://github.com/codewithdark-git/QuantLLM.git
```

## Installation Options

### With GGUF Support
For exporting to GGUF format (llama.cpp, Ollama):

```bash
pip install "quantllm[gguf] @ git+https://github.com/codewithdark-git/QuantLLM.git"
```

### With Hub Lifecycle
For HuggingFace Hub integration:

```bash
pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git
```

### Full Installation
All optional dependencies:

```bash
pip install "quantllm[full] @ git+https://github.com/codewithdark-git/QuantLLM.git"
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (recommended for GPU acceleration)

## Verify Installation

```python
from quantllm import turbo, __version__
print(f"QuantLLM v{__version__}")
```

## Hardware Requirements

| Setup | RAM | GPU | Models |
|-------|-----|-----|--------|
| Minimal | 8GB | None | 1-3B |
| Recommended | 16GB | 8GB VRAM | 7B |
| Optimal | 32GB+ | 24GB+ | 70B+ |
