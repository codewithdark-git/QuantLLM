# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

## Quick Install

```bash
pip install quantllm
```

## From Source

```bash
git clone https://github.com/codewithdark-git/QuantLLM.git
cd QuantLLM
pip install -e .
```

## Optional Dependencies

### Flash Attention (Recommended for Speed)

```bash
pip install flash-attn --no-build-isolation
```

### Triton Kernels (GPU Optimization)

```bash
pip install triton
```

### Full Installation

```bash
pip install quantllm[full]
```

This includes:
- `flash-attn` - Flash Attention for faster inference
- `triton` - GPU kernel optimization
- `bitsandbytes` - 4/8-bit quantization

## Verify Installation

```python
import quantllm
print(quantllm.__version__)  # Should print 3.0.0

# Check system info
from quantllm.cli import cmd_info
cmd_info()
```

## Troubleshooting

### CUDA Not Available

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Flash Attention Build Errors

Flash Attention requires CUDA toolkit. Install it first:
```bash
# Ubuntu
sudo apt install nvidia-cuda-toolkit

# Then install flash-attn
pip install flash-attn --no-build-isolation
```
