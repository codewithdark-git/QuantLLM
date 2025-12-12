# QuantLLM Documentation

Welcome to QuantLLM v2.0 - Ultra-fast LLM quantization with one line of code.

## Quick Start

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3-8B")
model.generate("Hello, world!")
model.finetune("data.json")
model.export("gguf", "model.gguf")
```

## Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/turbo
api/training
api/export
api/hub
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/basic
examples/finetuning
examples/export
```

## Features

- **45 Model Architectures** - Llama, Mistral, Qwen, Phi, Gemma, and more
- **6 Export Formats** - GGUF, ONNX, SafeTensors, MLX, AWQ, PyTorch
- **Auto Configuration** - Zero-config with smart defaults
- **Speed Optimized** - Triton kernels, Flash Attention 2
- **Memory Efficient** - Dynamic offloading, gradient checkpointing