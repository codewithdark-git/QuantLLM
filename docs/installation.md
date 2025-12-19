# 📦 Installation

Get QuantLLM up and running in minutes.

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+ (for GPU acceleration) |

---

## Quick Install

### From GitHub (Recommended)

```bash
pip install git+https://github.com/codewithdark-git/QuantLLM.git
```

### From PyPI

```bash
pip install quantllm
```

---

## Installation Options

Choose the features you need:

```bash
# Basic installation
pip install quantllm

# With GGUF export support
pip install "quantllm[gguf]"

# With ONNX export support  
pip install "quantllm[onnx]"

# With MLX export (Apple Silicon)
pip install "quantllm[mlx]"

# With Triton kernels (Linux, faster inference)
pip install "quantllm[triton]"

# With Flash Attention
pip install "quantllm[flash]"

# Full installation (everything)
pip install "quantllm[full]"
```

---

## From Source (Development)

```bash
git clone https://github.com/codewithdark-git/QuantLLM.git
cd QuantLLM
pip install -e ".[dev]"
```

---

## Verify Installation

```python
import quantllm

# Check version
print(f"QuantLLM v{quantllm.__version__}")

# Show banner
quantllm.show_banner()

# Quick test
from quantllm import turbo
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(model.generate("Hello!"))
```

Expected output:
```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   🚀 QuantLLM v2.0.0                                       ║
║   Ultra-fast LLM Quantization & Export                     ║
║                                                            ║
║   ✓ GGUF  ✓ ONNX  ✓ MLX  ✓ SafeTensors                     ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

## Optional Dependencies

### Flash Attention (Faster Inference)

```bash
pip install flash-attn --no-build-isolation
```

> **Note**: Requires CUDA toolkit installed on your system.

### Triton Kernels (GPU Optimization)

```bash
pip install triton>=2.1.0
```

> **Note**: Linux only. Provides fused quantization kernels.

---

## Troubleshooting

### CUDA Not Available

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, reinstall PyTorch with CUDA:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Flash Attention Build Errors

Flash Attention requires NVIDIA CUDA toolkit:

```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Then install
pip install flash-attn --no-build-isolation
```

### Memory Issues

If you encounter OOM errors:

```python
# Use 4-bit quantization
model = turbo("model-name", bits=4)

# Or use a smaller model
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### Windows Issues

Some features require Visual C++ Build Tools:

1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install "Desktop development with C++"
3. Restart your terminal

---

## Hardware Requirements

| GPU VRAM | Recommended Models |
|----------|-------------------|
| 6-8 GB | 1-7B models (4-bit) |
| 12-24 GB | 7-30B models (4-bit) |
| 24-80 GB | 70B+ models |

**Tested GPUs:**
- NVIDIA: RTX 3060, 3070, 3080, 3090, 4070, 4080, 4090, A100, H100
- AMD: RX 7900 XTX (with ROCm)
- Apple: M1, M2, M3, M4 (via MLX export)

---

## Next Steps

- [Quick Start →](quickstart.md)
- [Loading Models →](guide/loading-models.md)
- [GGUF Export →](guide/gguf-export.md)
