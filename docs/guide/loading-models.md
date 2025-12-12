# Loading Models

QuantLLM provides flexible model loading with automatic optimization.

## Basic Loading

```python
from quantllm import turbo

# Load from HuggingFace Hub
model = turbo("meta-llama/Llama-2-7b")

# Load from local path
model = turbo("./my-local-model/")
```

## Quantization Options

### Automatic (Recommended)

```python
# Auto-detect best quantization for your hardware
model = turbo("meta-llama/Llama-2-7b")
```

### Manual Bit-Width

```python
# Force specific bit-width
model = turbo("meta-llama/Llama-2-7b", bits=4)   # 4-bit
model = turbo("meta-llama/Llama-2-7b", bits=8)   # 8-bit
model = turbo("meta-llama/Llama-2-7b", bits=16)  # FP16
```

## Configuration Options

```python
model = turbo(
    "meta-llama/Llama-2-7b",
    bits=4,                     # Quantization bits
    max_seq_length=4096,        # Context length
    device="cuda:0",            # Device (cuda, cpu, auto)
    dtype="float16",            # Data type (float16, bfloat16)
    trust_remote_code=True,     # For custom models
)
```

## Using TurboModel Directly

For more control, use the `TurboModel` class:

```python
from quantllm import TurboModel, SmartConfig

# Create custom config
config = SmartConfig.detect("meta-llama/Llama-2-7b", bits=4)
config.use_flash_attention = True
config.gradient_checkpointing = True

# Load with custom config
model = TurboModel.from_pretrained(
    "meta-llama/Llama-2-7b",
    config=config,
)
```

## Supported Models

QuantLLM supports all HuggingFace causal language models:

| Model Family | Example |
|-------------|---------|
| Llama | `meta-llama/Llama-2-7b` |
| Mistral | `mistralai/Mistral-7B-v0.1` |
| Qwen | `Qwen/Qwen2-7B` |
| Phi | `microsoft/phi-2` |
| Gemma | `google/gemma-7b` |
| TinyLlama | `TinyLlama/TinyLlama-1.1B` |

## Memory Optimization

For large models on limited GPU memory:

```python
# Enable CPU offloading
model = turbo("meta-llama/Llama-2-70b", bits=4)
model.config.cpu_offload = True

# Enable gradient checkpointing for training
model.config.gradient_checkpointing = True
```
