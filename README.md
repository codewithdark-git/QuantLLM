<div align="center">

# QuantLLM

**Load, quantise, serve, benchmark, and push any LLM -- in one line.**

[![PyPI](https://img.shields.io/pypi/v/quantllm?logo=pypi&label=version&color=orange)](https://pypi.org/project/quantllm/)
[![Python](https://img.shields.io/badge/python-3.10+-orange.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

[Quick Start](#quick-start) • [CLI](#cli) • [API](#api) • [Examples](examples/) • [Custom Workflow](#custom-workflow)

</div>

---

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3.2-3B")
model.generate("Explain quantum computing simply")
model.export("gguf", "model.gguf")
model.push("your-username/my-model")
```

That is the entire API. One function loads any HuggingFace model, auto-detects your hardware, picks optimal quantization, enables Flash Attention, and configures memory. No config objects, no boilerplate.

## Quick Start

```bash
pip install quantllm                    # core
pip install "quantllm[full]"            # everything (GGUF, ONNX, MLX, server)
```

```python
from quantllm import turbo

# Load with auto-quantization
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generate, chat, stream
model.generate("What is Python?")
model.chat([{"role": "user", "content": "Hello"}])
for token in model.generate("Count to 5", stream=True):
    print(token, end="")

# Export to any format
model.export("gguf", "model.gguf", quantization="Q4_K_M")
model.export("onnx", "model.onnx")
model.export("safetensors", "./safetensors/")

# Push to HuggingFace Hub (auto-generates model card)
model.push("your-username/my-quantized-model", license="apache-2.0")
```

Everything is optional. Skip the model and it auto-detects `TinyLlama-1.1B`. Skip the quantization and it picks `Q4_K_M`. Skip the config and `SmartConfig` reads your hardware and picks optimal settings.

## What It Does

| Operation | One line | What happens |
|-----------|----------|-------------|
| **Load** | `turbo("org/model")` | Detects GPU/CPU, picks bits (2-16), dtype, group size, flash attention, offloading |
| **Quantise** | Built into load | BnB 4/8-bit or HQQ 2-8 bit, no calibration data needed |
| **Generate** | `model.generate(...)` | Text generation, chat, streaming |
| **Export** | `model.export("gguf")` | GGUF, ONNX, MLX, SafeTensors -- no SDK required |
| **Fine-tune** | `model.finetune(data)` | LoRA on quantized models, any dataset format |
| **Serve** | `quantllm serve ...` | OpenAI-compatible API, auto-detects GGUF backend |
| **Benchmark** | `quantllm bench ...` | Tokens/sec, latency, VRAM -- with hardware comparison |
| **Push** | `model.push("user/repo")` | Auto-generated model card, any format |

## CLI

```bash
quantllm                  # Full CLI
  version                 # Show version
  info                    # Show system info
  convert <model> -o ...  # Convert to GGUF
  finetune <model> ...    # Fine-tune with LoRA
  export <model> -f ...   # Export to any format
  serve <model> --port 8080   # Start inference server
  bench <model> ...       # Run benchmarks
  compare                 # Hardware auto-benchmark
  models list             # List curated models
  models search <query>   # Search models
  register <id> ...       # Register model on HF Hub
```

```bash
quantllm info
quantllm serve TinyLlama-1.1B --port 8080
quantllm bench TinyLlama-1.1B --max-tokens 128 --save
quantllm models list --family llama
```

## API Quick Reference

### `turbo(model_id, bits=None, config=None, device=None, quantize=True, verbose=False)`

| Param | Default | Description |
|-------|---------|-------------|
| `model_id` | `"TinyLlama-1.1B"` | HF model name or local path |
| `bits` | auto | 2-16; auto-selected by `SmartConfig.detect()` |
| `config` | `{}` | Export/push config: `{format, quantization, output_dir, push_format}` |
| `device` | auto | `"cuda:0"` or `"cpu"` |
| `quantize` | `True` | Whether to apply BitsAndBytes quantization |
| `verbose` | `False` | Show loading details |

### `SmartConfig` (auto-decided)

| Setting | Decision logic |
|---------|---------------|
| `bits` | `_choose_bits()`: model size vs GPU memory, 3x headroom for training, 1.5x for inference |
| `quant_type` | `_choose_quant_type()`: `Q4_K_M` default, varies by bits |
| `group_size` | `_choose_group_size()`: 64 for >30B, 128 otherwise |
| `dtype` | bf16 (Ampere+) > fp16 (CUDA) > fp32 (CPU) |
| `device` | CUDA if available, else CPU |
| `use_flash_attention` | Compute capability >= 8.0 |
| `use_fused_kernels` | Ampere+ GPU |
| `compile_model` | GPU with >= 16 GB VRAM |
| `cpu_offload` | Enabled when model exceeds available VRAM |

### `model.push(repo_id, format=None, private=False, license="mit", token=None)`

Pushes to HuggingFace Hub. Auto-exports before push. Generates a model card with:
quantization details, performance benchmarks, usage instructions, and metadata.

### `model.finetune(data, epochs=3, batch_size=4, learning_rate=2e-4, lora_r=8, lora_alpha=16, output_dir="./finetuned")`

Fine-tunes with LoRA. Accepts list of dicts (`{"text": "..."}`), JSON file path, or HuggingFace dataset name.

## Quantization

Quantization is automatic on load. You can also control it explicitly:

```python
# BitsAndBytes (4-bit or 8-bit, via transformers)
model = turbo("meta-llama/Llama-3.2-3B", bits=4)

# HQQ (2-8 bit, native, no calibration data)
from quantllm.quant import HQQQuantizer, HQQConfig

model = turbo("TinyLlama-1.1B", bits=16)  # load unquantized
quantizer = HQQQuantizer(HQQConfig(nbits=4, group_size=64))
qmodel = quantizer.quantize_model(model.model)
```

**HQQ Performance** (tested on H100, 7B model):
| Bits | Time | Memory | Quality |
|------|------|--------|---------|
| 2-bit | 1.8s | 1.86 GB | Functional |
| 4-bit | 1.0s | 3.72 GB | Coherent |
| 8-bit | 1.7s | 7.45 GB | Coherent |

## Inference Server

```bash
quantllm serve meta-llama/Llama-3.2-3B --port 8080
```

Auto-detects GGUF files and uses `llama-cpp-python`; otherwise uses Transformers.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Hardware Requirements

| GPU VRAM | Models |
|----------|--------|
| 6-8 GB | 1-7B (4-bit) |
| 12-24 GB | 7-30B (4-bit) |
| 24-80 GB | 70B+ |

Tested on RTX 3060/3070/3080/3090/4070/4080/4090, A100, H100, Apple M1-M4.

## Custom Workflow

Complete end-to-end pipeline demonstrating QuantLLM on a new/public model, run on Modal GPU (H100).

### 1. Load & Auto-Quantize
```python
from quantllm import turbo
model = turbo("Qwen/Qwen2.5-7B-Instruct")  # 7.62B params, BnB 8-bit on H100
```

### 2. Test Quantization (BnB + HQQ)
```python
# BitsAndBytes runtime quantization
model_bnb4 = turbo("Qwen/Qwen2.5-7B-Instruct", bits=4)
model_bnb8 = turbo("Qwen/Qwen2.5-7B-Instruct", bits=8)

# HQQ native quantization (export/optimization)
from quantllm.quant import HQQQuantizer, HQQConfig
import torch

weight = torch.randn(256, 512)
for bits in [2, 4, 8]:
    config = HQQConfig(bits=bits, group_size=64, optimize=True)
    hqq_layer = HQQLinear(weight, config)
```

### 3. Export to All Formats
```python
# GGUF (llama.cpp) - multiple quant types
model.export("gguf", "model.Q4_K_M.gguf", quantization="Q4_K_M")
model.export("gguf", "model.Q5_K_M.gguf", quantization="Q5_K_M")
model.export("gguf", "model.Q8_0.gguf", quantization="Q8_0")

# SafeTensors (HuggingFace native)
model.export("safetensors", "./exports/safetensors/")

# ONNX (Optimum)
model.export("onnx", "./exports/onnx/", quantization="int8")

# MLX (Apple Silicon)
model.export("mlx", "./exports/mlx/", quantization="4bit")
```

### 4. Push to HF Hub with Model Cards
```python
model.push("QuantLLM/Qwen2.5-7B-Instruct-GGUF", format="gguf", quantization="Q4_K_M")
model.push("QuantLLM/Qwen2.5-7B-Instruct-SafeTensors", format="safetensors")
model.push("QuantLLM/Qwen2.5-7B-Instruct-ONNX", format="onnx", quantization="int8")
model.push("QuantLLM/Qwen2.5-7B-Instruct-MLX", format="mlx", quantization="4bit")
```

### 5. Register in QuantLLM Registry
```python
from quantllm.registry import register_model

register_model("QuantLLM/Qwen2.5-7B-Instruct-GGUF", family="qwen", params=7.0, verified=True, recommended="Q4_K_M")
register_model("QuantLLM/Qwen2.5-7B-Instruct-SafeTensors", family="qwen", params=7.0, verified=True, recommended="int8")
# ... etc
```

### 6. Validate from Registry
```bash
quantllm models list
# QuantLLM/Qwen2.5-7B-Instruct-GGUF          qwen    7.0  4.0  4.5  Yes

model = turbo("QuantLLM/Qwen2.5-7B-Instruct-GGUF")
model.generate("Test")
```

### Full Workflow Script
```bash
python examples/09_custom_workflow.py
```

Or run on Modal GPU:
```bash
modal run modal_custom_workflow.py::run_workflow
```

## Examples

Full, runnable examples in [`examples/`](examples/):

```bash
python examples/01_quickstart.py      # Load, generate, chat, stream, export
python examples/05_hqq_quantization.py # HQQ 2-8 bit quantization
python examples/07_benchmark.py        # Live benchmark + comparison
python examples/08_full_pipeline.py    # Load -> Generate -> Chat -> Export -> Push -> Serve
python examples/09_custom_workflow.py  # Complete custom model workflow
```

---

<div align="center">

[MIT License](LICENSE) • [GitHub](https://github.com/codewithdark-git/QuantLLM) • [Report Bug](https://github.com/codewithdark-git/QuantLLM/issues)

</div>