# 📦 GGUF API

Export models to GGUF format for llama.cpp, Ollama, and LM Studio.

---

## Quick Reference

```python
from quantllm import turbo, convert_to_gguf, quantize_gguf

# Method 1: Via TurboModel
model = turbo("meta-llama/Llama-3.2-3B")
model.export("gguf", "model.Q4_K_M.gguf", quantization="Q4_K_M")

# Method 2: Direct conversion
convert_to_gguf("meta-llama/Llama-3.2-3B", "model.Q4_K_M.gguf", quant_type="Q4_K_M")

# Method 3: Re-quantize existing GGUF
quantize_gguf("model.F16.gguf", "model.Q4_K_M.gguf", quant_type="Q4_K_M")
```

---

## convert_to_gguf()

Convert a HuggingFace model to GGUF format.

```python
def convert_to_gguf(
    model_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
    model_dtype: str = "auto",
    verbose: bool = True,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | HuggingFace model name or local path |
| `output_path` | str | required | Output .gguf file path |
| `quant_type` | str | "Q4_K_M" | Quantization type |
| `model_dtype` | str | "auto" | Model dtype (auto, f16, f32) |
| `verbose` | bool | True | Show progress |

### Returns

Path to the created GGUF file.

### Example

```python
from quantllm import convert_to_gguf

# Basic conversion
convert_to_gguf(
    "meta-llama/Llama-3.2-3B",
    "llama3.Q4_K_M.gguf",
    quant_type="Q4_K_M"
)

# Higher quality
convert_to_gguf(
    "meta-llama/Llama-3.2-3B",
    "llama3.Q8_0.gguf",
    quant_type="Q8_0"
)
```

---

## quantize_gguf()

Re-quantize an existing GGUF file to a different quantization type.

```python
def quantize_gguf(
    input_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | str | required | Input GGUF file path |
| `output_path` | str | required | Output GGUF file path |
| `quant_type` | str | "Q4_K_M" | Target quantization type |

### Example

```python
from quantllm import quantize_gguf

# Re-quantize F16 to Q4_K_M
quantize_gguf(
    "model.F16.gguf",
    "model.Q4_K_M.gguf",
    quant_type="Q4_K_M"
)
```

---

## GGUF_QUANT_TYPES

Available quantization types.

```python
from quantllm import GGUF_QUANT_TYPES

print(GGUF_QUANT_TYPES)
# ['Q2_K', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L', 'Q4_K_S', 'Q4_K_M', 
#  'Q5_K_S', 'Q5_K_M', 'Q6_K', 'Q8_0', 'F16', 'F32']
```

### Quantization Comparison

| Type | Bits | Quality | Size (7B) | Use Case |
|------|------|---------|-----------|----------|
| `Q2_K` | 2 | Low | ~2 GB | Extreme compression |
| `Q3_K_S` | 3 | Fair | ~2.5 GB | Small devices |
| `Q3_K_M` | 3 | Fair | ~3 GB | Constrained memory |
| `Q4_K_S` | 4 | Good | ~3.5 GB | Balanced (smaller) |
| `Q4_K_M` | 4 | Good | ~4 GB | **Recommended** ⭐ |
| `Q5_K_S` | 5 | High | ~4.5 GB | Quality focus |
| `Q5_K_M` | 5 | High | ~5 GB | Quality balance |
| `Q6_K` | 6 | Very High | ~5.5 GB | Near original |
| `Q8_0` | 8 | Excellent | ~7 GB | Maximum quality |
| `F16` | 16 | Original | ~14 GB | Full precision |

---

## QUANT_RECOMMENDATIONS

Get recommendations based on hardware.

```python
from quantllm import QUANT_RECOMMENDATIONS

print(QUANT_RECOMMENDATIONS)
# {
#     'low_memory': 'Q3_K_M',      # <6 GB VRAM
#     'balanced': 'Q4_K_M',        # 6-12 GB VRAM (recommended)
#     'quality': 'Q5_K_M',         # 12-24 GB VRAM
#     'high_quality': 'Q6_K',      # >24 GB VRAM
#     'maximum': 'Q8_0',           # Maximum quality
# }
```

---

## check_llama_cpp()

Check if llama.cpp is installed.

```python
def check_llama_cpp() -> bool
```

### Example

```python
from quantllm import check_llama_cpp

if check_llama_cpp():
    print("llama.cpp is ready!")
else:
    print("llama.cpp not found")
```

---

## install_llama_cpp()

Install llama.cpp automatically.

```python
def install_llama_cpp(
    install_dir: str = "./llama.cpp",
    force: bool = False,
) -> str
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `install_dir` | str | "./llama.cpp" | Installation directory |
| `force` | bool | False | Force reinstall |

### Example

```python
from quantllm import install_llama_cpp

# Install to default location
install_llama_cpp()

# Install to custom location
install_llama_cpp("./tools/llama.cpp")
```

---

## ensure_llama_cpp_installed()

Ensure llama.cpp is installed, installing if needed.

```python
def ensure_llama_cpp_installed() -> str
```

### Example

```python
from quantllm import ensure_llama_cpp_installed

# Automatically installs if not present
llama_path = ensure_llama_cpp_installed()
print(f"llama.cpp at: {llama_path}")
```

---

## export_to_gguf()

High-level export function (deprecated, use `convert_to_gguf`).

```python
def export_to_gguf(
    model,
    tokenizer,
    output_path: str,
    quant_type: str = "Q4_K_M",
) -> str
```

---

## Using Exported Models

### llama.cpp

```bash
./llama-cli -m model.Q4_K_M.gguf -p "Hello!" -n 100
```

### Ollama

```bash
echo 'FROM ./model.Q4_K_M.gguf' > Modelfile
ollama create mymodel -f Modelfile
ollama run mymodel
```

### LM Studio

1. Import the `.gguf` file
2. Start chatting

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(model_path="model.Q4_K_M.gguf")
output = llm("Hello!", max_tokens=100)
print(output["choices"][0]["text"])
```

---

## See Also

- [GGUF Export Guide](../guide/gguf-export.md) — Detailed guide
- [TurboModel.export()](turbomodel.md#export) — Export via TurboModel
- [Hub Integration](hub.md) — Push GGUF to HuggingFace
