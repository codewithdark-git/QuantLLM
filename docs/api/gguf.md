# GGUF Export API

Pure Python GGUF conversion - no llama.cpp required.

## convert_to_gguf()

High-level function for GGUF conversion.

```python
def convert_to_gguf(
    model,
    tokenizer,
    output_path: str,
    quant_type: str = "q4_0",
    verbose: bool = True,
) -> str
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | PreTrainedModel | required | HuggingFace model |
| `tokenizer` | PreTrainedTokenizer | required | Tokenizer |
| `output_path` | str | required | Output .gguf file path |
| `quant_type` | str | "q4_0" | Quantization type |
| `verbose` | bool | True | Show progress |

**Returns:** Path to created GGUF file.

**Example:**
```python
from quantllm import convert_to_gguf
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B")

convert_to_gguf(model, tokenizer, "output.gguf", "q4_0")
```

---

## list_quant_types()

Get available quantization types.

```python
def list_quant_types() -> Dict[str, str]
```

**Returns:** Dictionary of type names and descriptions.

---

## QUANT_TYPES

Registry of quantization types with details.

```python
QUANT_TYPES = {
    "f32": QuantizationInfo("F32", ...),
    "f16": QuantizationInfo("F16", ...),
    "q8_0": QuantizationInfo("Q8_0", ...),
    "q4_0": QuantizationInfo("Q4_0", ...),
    # ... more types
}
```

---

## GGUFWriter

Low-level GGUF file writer.

```python
class GGUFWriter:
    def __init__(self, output_path: str, arch: str = "llama"):
        ...
    
    def add_architecture(self):
        """Add architecture metadata."""
    
    def add_string(self, key: str, value: str):
        """Add string metadata."""
    
    def add_uint32(self, key: str, value: int):
        """Add uint32 metadata."""
    
    def add_tensor(self, name: str, tensor: Tensor, quant_type: str):
        """Add a tensor to be quantized and written."""
    
    def write(self, show_progress: bool = True):
        """Write the GGUF file."""
```

**Example:**
```python
from quantllm import GGUFWriter

writer = GGUFWriter("custom.gguf", arch="llama")
writer.add_architecture()
writer.add_uint32("llama.context_length", 4096)

for name, param in model.named_parameters():
    writer.add_tensor(name, param, "q4_0")

writer.write()
```

---

## FastQuantizer

Fast quantization kernels.

```python
class FastQuantizer:
    @staticmethod
    def quantize_q8_0(tensor: Tensor) -> bytes:
        """Quantize to Q8_0 format."""
    
    @staticmethod
    def quantize_q4_0(tensor: Tensor) -> bytes:
        """Quantize to Q4_0 format."""
    
    @staticmethod
    def quantize_f16(tensor: Tensor) -> bytes:
        """Convert to FP16."""
```
