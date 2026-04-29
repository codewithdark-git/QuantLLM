# 🖥️ Consumer Hardware Guide

QuantLLM is designed to run modern LLMs on the kind of hardware most
developers actually have — gaming GPUs, laptops, Apple Silicon Macs and
even pure-CPU machines. This page is a flat index of what works, where
to expect compromises, and which knobs to turn for each tier.

`turbo()` auto-detects your hardware via :class:`SmartConfig` and picks
sensible defaults, but you can always override any value explicitly.

---

## Quick decision table

| Hardware                     | Recommended bits | Recommended path                           | Notes                                                           |
|------------------------------|------------------|--------------------------------------------|-----------------------------------------------------------------|
| CPU only / no GPU            | 4 (GGUF)         | `TurboModel.from_gguf(...)` (llama.cpp)    | BitsAndBytes is GPU-only; GGUF runs on CPU.                     |
| Apple Silicon (M-series)     | 4 (MLX or GGUF)  | Export to MLX or GGUF                      | Native Metal kernels; see *Apple Silicon* below.                |
| ≤ 4 GB VRAM (GTX 1650, etc.) | 4 (BnB) + offload| `turbo(..., bits=4)` with CPU offload      | Auto-enabled by SmartConfig; expect slower generation.          |
| 6 – 8 GB VRAM (3060/3070)    | 4 (BnB)          | `turbo(..., bits=4)`                       | The default sweet spot for 7B-class models.                     |
| 12 GB VRAM (3060 12 / 3080)  | 4 or 8           | `turbo(...)` (auto)                        | 13B at 4-bit fits comfortably; 7B can run at 8-bit.             |
| 16 – 24 GB VRAM (4080/4090)  | 4 or 8           | `turbo(...)` (auto)                        | 70B-class needs 4-bit + cpu_offload; 13B at 8-bit fits.         |
| Multi-GPU / server           | 4 / 8 / 16       | `turbo(..., device_map="auto")`            | `accelerate` shards weights across visible GPUs.                |

---

## CPU-only inference

If `torch.cuda.is_available()` is `False`, BitsAndBytes is unavailable
(it depends on CUDA). Use one of:

```python
# Option A: load full precision on CPU (slow but works)
model = turbo("microsoft/phi-2", quantize=False, device="cpu")

# Option B (recommended): load a pre-quantized GGUF via llama.cpp
from quantllm import TurboModel
model = TurboModel.from_gguf(
    "TheBloke/phi-2-GGUF",
    filename="phi-2.Q4_K_M.gguf",
)
```

`from_gguf` uses `llama-cpp-python` under the hood, which produces real
4-bit / 5-bit / 8-bit inference on CPU at usable speeds.

---

## ≤ 8 GB VRAM (gaming laptops, GTX 1660 / RTX 2060 / 3060 6 GB)

```python
from quantllm import turbo

# 4-bit NF4 with double quant, automatic CPU offload, fp16 compute
model = turbo("meta-llama/Llama-3.2-3B-Instruct", bits=4)

# Fine-tune with LoRA at 4-bit (QLoRA-style)
model.finetune("my_data.json", epochs=3)
```

Inspect the actual loaded state via `model.report()` if generation feels
slow — the `device` and `quant_method` keys tell you whether the model
ended up on GPU or partly offloaded.

### What SmartConfig does for small VRAM

* Picks `bits=4` and `BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)`.
* Sets `cpu_offload=True` automatically when the model exceeds ~90% of
  free VRAM (with a 1.5× headroom factor for inference, 3× for training).
* Falls back to `torch.float16` compute when bf16 is not supported.
* Keeps `compile_model=False` (Torch compile only kicks in at ≥ 16 GB
  total VRAM, where its compile-time cost amortises).

---

## 12 – 24 GB VRAM (consumer flagship: 3060 12 / 3080 / 4080 / 4090)

```python
# Auto: SmartConfig will pick 4-bit for 13B, 8-bit for 7B
model = turbo("mistralai/Mistral-7B-Instruct-v0.3")

# Or be explicit
model = turbo("meta-llama/Llama-3.1-8B-Instruct", bits=8)
```

At 24 GB you can fit:
* 70B-class models at 4-bit *with* CPU offload for the largest layers.
* 13B-class models at 8-bit comfortably.
* 7B-class models at 16-bit if you want zero-loss inference for
  benchmarking.

---

## Apple Silicon (M1 / M2 / M3 / M4)

There is no CUDA on Apple Silicon and BitsAndBytes does not run there.
The two supported paths are:

```python
# Export an MLX-format model that uses Apple's native Metal kernels
model = turbo("microsoft/phi-2", quantize=False)   # load on CPU first
model.export(format="mlx", path="./phi-2-mlx")

# Or download a GGUF and run it through llama.cpp's Metal backend
from quantllm import TurboModel
model = TurboModel.from_gguf(
    "TheBloke/phi-2-GGUF",
    filename="phi-2.Q4_K_M.gguf",
)
```

`pip install quantllm[mlx]` pulls `mlx` and `mlx-lm`. `from_gguf` uses
`llama-cpp-python`, which auto-detects Metal at install time when built
with `CMAKE_ARGS="-DLLAMA_METAL=on"`.

---

## Multi-GPU & server-class hardware

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model = turbo(
    "meta-llama/Llama-3.1-70B-Instruct",
    bits=4,
    # Pass through to ``transformers``: shard weights across all GPUs
    device_map="auto",
)
```

For training jobs, prefer `accelerate launch` with a YAML config; the
SmartConfig auto-tuner only inspects GPU 0 and won't pick up multi-GPU
batch-size headroom on its own.

---

## When QuantLLM cannot quantize

A few situations produce a full-precision model even when you asked for
4-bit. They are now reflected honestly in `model.is_quantized`:

| Situation                                | `is_quantized` | What QuantLLM does                                                |
|------------------------------------------|----------------|--------------------------------------------------------------------|
| `bitsandbytes` not installed (no CUDA)   | `False`        | Logs a warning and loads in fp16 / bf16.                           |
| Model already shipped with `quantization_config` (GPTQ / AWQ / etc.) | `True` | Honours the embedded config; skips dynamic BnB.                    |
| `from_config_only=True`                  | `False`        | Returns random-init weights; warns that the model is not usable.   |
| `bits=16` (explicit full precision)      | `False`        | Loads the original checkpoint without any quantization layer.      |

Always use `model.report()` if you need to *programmatically* assert
which quantization path actually ran — it is the canonical source of
truth.

---

## Troubleshooting

### `ImportError: bitsandbytes is not installed`

```bash
pip install bitsandbytes
# or, on Windows:
pip install bitsandbytes --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui
```

If you cannot install it (no CUDA, ROCm, MPS), use the GGUF path above.

### CUDA OOM on a 6 – 8 GB GPU

1. Confirm you got 4-bit via `model.report()`. If not, force `bits=4`.
2. Close other GPU processes (`nvidia-smi`); VRAM is sticky.
3. Lower `max_length` (e.g. `turbo(..., max_length=2048)`).
4. Set `device_map="auto"` to let `accelerate` offload entire blocks.

### Slow generation on consumer GPUs

1. Make sure `flash-attn` is installed (`pip install quantllm[flash]`).
2. Check that the model lives on GPU: `model.report()["device"]` should
   start with `cuda`. If you see `cpu`, free up VRAM or drop to a
   smaller model.
3. For longer-running deployments, export to GGUF and serve through
   `llama.cpp` — it produces ~2–3× the tokens/sec of HF on the same
   hardware below 7B.
