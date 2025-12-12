# turbo()

The main entry point for QuantLLM - a one-liner to load any model.

## Signature

```python
def turbo(
    model_name: str,
    bits: Optional[int] = None,
    max_seq_length: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs
) -> TurboModel
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | required | HuggingFace model name or local path |
| `bits` | int | auto | Quantization bits (2, 3, 4, 5, 6, 8, 16) |
| `max_seq_length` | int | auto | Maximum sequence length |
| `device` | str | auto | Device ("cuda", "cpu", "cuda:0") |
| `dtype` | str | auto | Data type ("float16", "bfloat16") |
| `trust_remote_code` | bool | False | Trust remote code in model |

## Returns

A `TurboModel` instance ready for generation.

## Examples

### Basic Usage

```python
from quantllm import turbo

model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### With Options

```python
model = turbo(
    "meta-llama/Llama-2-7b",
    bits=4,
    max_seq_length=4096,
    device="cuda:0",
)
```

### Local Model

```python
model = turbo("./my-local-model/")
```

## Auto-Configuration

When parameters are not specified, `turbo()` automatically:

1. **Detects hardware** (GPU memory, CUDA version, CPU cores)
2. **Analyzes model** (size, architecture, optimal settings)
3. **Chooses quantization** (4-bit if memory limited, 8-bit if available)
4. **Enables optimizations** (Flash Attention, fused kernels)

## See Also

- [TurboModel](turbomodel.md) - Full class documentation
- [SmartConfig](turbomodel.md#smartconfig) - Configuration details
