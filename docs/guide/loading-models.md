# 📥 Loading Models

QuantLLM provides flexible model loading with automatic optimization.

---

## Basic Loading

### The `turbo()` Function

The simplest way to load any model:

```python
from quantllm import turbo

# Load from HuggingFace Hub
model = turbo("meta-llama/Llama-3.2-3B")

# Load from local path
model = turbo("./my-local-model/")
```

### What Happens Automatically

When you call `turbo()`, QuantLLM:

1. **Detects your hardware** — GPU memory, CUDA version, capabilities
2. **Chooses quantization** — 4-bit for most GPUs, 8-bit for high-memory systems
3. **Enables optimizations** — Flash Attention 2, gradient checkpointing
4. **Configures memory** — Automatic offloading if needed

---

## Quantization Options

### Automatic (Recommended)

```python
# Let QuantLLM choose the best quantization
model = turbo("meta-llama/Llama-3.2-3B")
```

### Manual Bit-Width

```python
# Force specific quantization
model = turbo("meta-llama/Llama-3.2-3B", bits=4)   # 4-bit (smallest)
model = turbo("meta-llama/Llama-3.2-3B", bits=8)   # 8-bit (balanced)
model = turbo("meta-llama/Llama-3.2-3B", bits=16)  # FP16 (highest quality)
```

### Disable Quantization

```python
# Load in full precision (requires more memory)
model = turbo("meta-llama/Llama-3.2-3B", quantize=False)
```

---

## Configuration Options

### Common Options

```python
model = turbo(
    "meta-llama/Llama-3.2-3B",
    bits=4,                      # Quantization bits (4, 8, 16)
    max_length=4096,             # Maximum context length
    device="cuda:0",             # Device (cuda, cpu, auto)
    dtype="bfloat16",            # Data type (float16, bfloat16)
    trust_remote_code=True,      # For custom model architectures
    verbose=True,                # Show loading progress
)
```

### New Architecture Fallbacks (for very recent model releases)

QuantLLM ships a built-in fallback table covering common model-type
suffixes — `qwen3` → `qwen2`, `llama4` → `llama`, `phi4` → `phi3`,
`gemma3` → `gemma2`, and many others — so brand-new releases load with
the same one-line API as established models:

```python
from quantllm import turbo

# Works without manual registration: qwen3 falls back to qwen2 automatically
model = turbo("Qwen/Qwen3-8B", trust_remote_code=True)
```

When the built-in mapping does not cover your model, register an
explicit fallback before loading:

```python
from quantllm import turbo, register_architecture

# Map a brand-new architecture/model_type to a compatible base family
register_architecture("newmodel", base_model_type="llama")

# Optionally provide an explicit ``model_class`` (most useful for
# fine-tuned variants that ship their own modelling code):
from transformers import LlamaForCausalLM
register_architecture(
    "newmodel",
    base_model_type="llama",
    model_class=LlamaForCausalLM,
)

model = turbo(
    "new-model-org/NewModel-7B",
    model_type_override="llama",     # optional explicit override
    base_model_fallback=True,        # enabled by default; can be disabled
    trust_remote_code=True,
)
```

> ⚠️ **Security note:** `trust_remote_code=True` executes model-provided code.
> Only enable it for trusted publishers, especially when loading unregistered or very new architectures.

#### Pre-quantized HuggingFace repos

QuantLLM detects pre-quantized repository names (Unsloth `*-bnb-4bit` /
`*-bnb-8bit`, AWQ, GPTQ, AQLM, HQQ, FP8, EETQ, etc.) and lets the model's
own `quantization_config` win — so you don't accidentally re-quantize a
model that ships at-rest in 4-bit:

```python
# Loaded as 4-bit BitsAndBytes from the repo's embedded config -- no
# additional dynamic quantization is applied on top.
model = turbo("unsloth/Llama-3.2-3B-Instruct-bnb-4bit")

# Verify what actually got loaded:
print(model.report())
# {'quant_method': 'bitsandbytes', 'is_quantized': True, ...}
```

#### `from_config_only` is for skeleton inspection only

```python
# Loads a randomly-initialised model from the config -- useful for
# inspecting layer shapes or wiring up tests, NOT for inference.
model = turbo(
    "new-model-org/NewModel-7B",
    from_config_only=True,
    trust_remote_code=True,
)

# ``model.is_quantized`` will correctly report False here even when you
# also passed ``bits=4`` -- there are no real weights to quantize.
```

#### Fast contribution template for new architectures

1. Add a registration in your code or PR:
   - `register_architecture("new-arch", base_model_type="llama")`
2. Validate loading with:
   - `turbo("org/model", base_model_fallback=True, trust_remote_code=True)`
3. Add/extend a focused test in `tests/test_architecture_fallback.py`
   or `tests/test_resolve_model_type.py`.

#### Inspecting the loaded state

```python
model = turbo("Qwen/Qwen3-8B", bits=4)

report = model.report()
# {
#   'model_id': 'Qwen/Qwen3-8B',
#   'params_billion': 8.0,
#   'requested_bits': 4,
#   'effective_loading_bits': 4,
#   'is_quantized': True,
#   'quant_method': 'bitsandbytes',
#   'device': 'cuda:0',
#   'dtype': 'torch.bfloat16',
#   'finetuned': False,
#   'lora_applied': False,
# }
```

`model.is_quantized` is derived from the actual loaded model state
(`config.quantization_config` and BitsAndBytes layer types). It is not
a cached snapshot of your load-time intent, so `from_config_only=True`
or a missing `bitsandbytes` install will correctly report `False`.

### Memory Options

```python
model = turbo(
    "meta-llama/Llama-3.2-3B",
    bits=4,
    device_map="auto",           # Automatic device mapping
    low_cpu_mem_usage=True,      # Reduce CPU memory during loading
)
```

---

## Using TurboModel Directly

For more control, use the `TurboModel` class:

```python
from quantllm import TurboModel, SmartConfig

# Create custom config
config = SmartConfig.detect("meta-llama/Llama-3.2-3B", bits=4)

# Load with custom config
model = TurboModel.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    config=config,
)
```

---

## Load GGUF Models

Load pre-quantized GGUF models directly from HuggingFace:

```python
from quantllm import TurboModel

# From HuggingFace Hub
model = TurboModel.from_gguf(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf"
)

# From local file
model = TurboModel.from_gguf("./models/my-model.gguf")
```

### List Available GGUF Files

```python
files = TurboModel.list_gguf_files("TheBloke/Llama-2-7B-Chat-GGUF")
print(files)
# ['llama-2-7b-chat.Q2_K.gguf', 'llama-2-7b-chat.Q4_K_M.gguf', ...]
```

---

## Supported Models

QuantLLM supports **45+ model architectures**:

| Family | Models |
|--------|--------|
| **Llama** | Llama 2, Llama 3, Llama 3.1, Llama 3.2, CodeLlama |
| **Mistral** | Mistral 7B, Mixtral 8x7B, Mixtral 8x22B |
| **Qwen** | Qwen, Qwen2, Qwen2.5, Qwen2-MoE |
| **Microsoft** | Phi-1, Phi-2, Phi-3 |
| **Google** | Gemma, Gemma 2 |
| **Falcon** | Falcon 7B, 40B, 180B |
| **Code Models** | StarCoder, StarCoder2, CodeGen |
| **Chinese** | ChatGLM, Yi, Baichuan, InternLM |
| **Other** | DeepSeek, StableLM, MPT, BLOOM, OPT, GPT-NeoX |

---

## Memory Optimization

### For Large Models

```python
# Enable gradient checkpointing (for training)
model = turbo("meta-llama/Llama-3-70B", bits=4)

# Use CPU offloading
model = turbo(
    "meta-llama/Llama-3-70B",
    bits=4,
    device_map="auto",  # Automatic CPU/GPU split
)
```

### Memory Usage Estimates

| Model Size | 4-bit | 8-bit | FP16 |
|------------|-------|-------|------|
| 3B | ~2 GB | ~4 GB | ~6 GB |
| 7B | ~4 GB | ~8 GB | ~14 GB |
| 13B | ~8 GB | ~14 GB | ~26 GB |
| 70B | ~40 GB | ~70 GB | ~140 GB |

---

## Best Practices

1. **Start with automatic settings** — Let QuantLLM detect your hardware
2. **Use 4-bit for most cases** — Best balance of quality and memory
3. **Check memory first** — `turbo()` shows memory stats before loading
4. **Use GGUF for inference** — Pre-quantized GGUF models load faster

---

## Next Steps

- [Text Generation →](generation.md) — Generate text with your model
- [Fine-tuning →](finetuning.md) — Train with your own data
- [GGUF Export →](gguf-export.md) — Export for deployment
