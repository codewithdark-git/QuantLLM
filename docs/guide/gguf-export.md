# GGUF Export

Export models to GGUF format for use with llama.cpp, Ollama, and LM Studio.

**No llama.cpp installation required!** QuantLLM uses pure Python for conversion.

## Quick Export

```python
from quantllm import turbo

model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model.export("gguf", "my-model.gguf")
```

## Quantization Types

| Type | Bits | Quality | Size | Use Case |
|------|------|---------|------|----------|
| `f16` | 16 | Highest | Large | Reference |
| `q8_0` | 8 | Very High | Medium | Quality-focused |
| `q5_0` | 5 | High | Small | Balanced |
| `q4_0` | 4 | Good | Smaller | **Recommended** |
| `q4_k` | 4 | Better | Similar | K-quant variant |

```python
# Different quantization types
model.export("gguf", "model-f16.gguf", quantization="f16")
model.export("gguf", "model-q8.gguf", quantization="q8_0")
model.export("gguf", "model-q4.gguf", quantization="q4_0")
```

## List All Types

```python
from quantllm import list_quant_types

for name, desc in list_quant_types().items():
    print(f"{name:12} - {desc}")
```

## Direct Conversion

Convert any HuggingFace model without TurboModel:

```python
from quantllm import convert_to_gguf
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B")

convert_to_gguf(
    model=model,
    tokenizer=tokenizer,
    output_path="output.gguf",
    quant_type="q4_0",
    verbose=True,
)
```

## Using Exported Files

### llama.cpp

```bash
./main -m my-model.gguf -p "Hello, world!"
```

### Ollama

Create a Modelfile:
```
FROM ./my-model.gguf

TEMPLATE """{{ .Prompt }}"""
```

Then:
```bash
ollama create mymodel -f Modelfile
ollama run mymodel
```

### LM Studio

1. Open LM Studio
2. Click "Import"
3. Select your `.gguf` file
4. Start chatting!

## Advanced Options

### GGUFWriter (Low-Level)

For custom conversion:

```python
from quantllm import GGUFWriter

writer = GGUFWriter("output.gguf", arch="llama")

# Add metadata
writer.add_architecture()
writer.add_uint32("llama.context_length", 4096)

# Add tensors
for name, param in model.named_parameters():
    writer.add_tensor(name, param, "q4_0")

# Write file
writer.write()
```
