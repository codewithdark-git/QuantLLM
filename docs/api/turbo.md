# 🚀 turbo()

The main entry point for QuantLLM — load any model in one line.

---

## Signature

```python
def turbo(
    model: str,
    *,
    bits: Optional[int] = None,
    max_length: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    quantize: bool = True,
    trust_remote_code: bool = False,
    verbose: bool = True,
    **kwargs
) -> TurboModel
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | HuggingFace model name or local path |
| `bits` | int | auto | Quantization bits (4, 8, 16) |
| `max_length` | int | auto | Maximum context length |
| `device` | str | auto | Device ("cuda", "cpu", "cuda:0", "auto") |
| `dtype` | str | auto | Data type ("float16", "bfloat16") |
| `quantize` | bool | True | Whether to apply quantization |
| `trust_remote_code` | bool | False | Trust remote code in model |
| `verbose` | bool | True | Show loading progress and stats |

---

## Returns

A [`TurboModel`](turbomodel.md) instance ready for generation, fine-tuning, and export.

---

## Examples

### Basic Usage

```python
from quantllm import turbo

# Load with automatic optimization
model = turbo("meta-llama/Llama-3.2-3B")

# Generate text
response = model.generate("What is machine learning?")
print(response)
```

### With Custom Settings

```python
model = turbo(
    "meta-llama/Llama-3.2-3B",
    bits=4,                    # Force 4-bit quantization
    max_length=4096,           # Context length
    device="cuda:0",           # Specific GPU
    dtype="bfloat16",          # Use bfloat16
)
```

### Without Quantization

```python
# Load in full precision
model = turbo("meta-llama/Llama-3.2-3B", quantize=False)
```

### Local Model

```python
model = turbo("./my-local-model/")
```

### Silent Loading

```python
model = turbo("meta-llama/Llama-3.2-3B", verbose=False)
```

---

## Auto-Configuration

When parameters are not specified, `turbo()` automatically:

1. **Detects hardware**
   - GPU memory and CUDA version
   - CPU cores and available RAM
   - Flash Attention availability

2. **Analyzes model**
   - Parameter count and size
   - Architecture type
   - Optimal settings

3. **Chooses quantization**
   - 4-bit if GPU memory < 16GB
   - 8-bit if GPU memory >= 16GB
   - No quantization if explicitly disabled

4. **Enables optimizations**
   - Flash Attention 2 when available
   - torch.compile for training
   - Dynamic memory management

---

## Output

When `verbose=True` (default), you'll see:

```
╔════════════════════════════════════════════════════════════╗
║  🚀 QuantLLM v2.0.0                                        ║
╚════════════════════════════════════════════════════════════╝

📊 Loading: meta-llama/Llama-3.2-3B
   Parameters: 3.21B
   Original: 6.4 GB
   Quantized: 1.9 GB (70% saved)
   
✓ Model loaded successfully
```

---

## See Also

- [TurboModel](turbomodel.md) — Full class documentation
- [SmartConfig](turbomodel.md#smartconfig) — Configuration details
- [Loading Models Guide](../guide/loading-models.md) — Detailed loading guide
