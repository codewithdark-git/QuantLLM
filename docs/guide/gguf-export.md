# 📦 GGUF Export

Export models to GGUF format for deployment with llama.cpp, Ollama, and LM Studio.

---

## Quick Export

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3.2-3B")

# Export with recommended Q4_K_M quantization
model.export("gguf", "model.Q4_K_M.gguf", quantization="Q4_K_M")
```

**No llama.cpp compilation required!** QuantLLM handles everything automatically.

---

## Quantization Types

Choose the right quantization for your needs:

| Type | Bits | Quality | Size | Use Case |
|------|------|---------|------|----------|
| `Q2_K` | 2-bit | Low | Smallest | Extreme compression |
| `Q3_K_S` | 3-bit | Fair | Very small | Memory constrained |
| `Q3_K_M` | 3-bit | Fair | Small | Balanced for 3-bit |
| `Q4_K_S` | 4-bit | Good | Small | Slightly smaller Q4 |
| `Q4_K_M` | 4-bit | Good | Medium | **Recommended** ⭐ |
| `Q5_K_S` | 5-bit | High | Medium | Quality-focused |
| `Q5_K_M` | 5-bit | High | Medium | Best 5-bit balance |
| `Q6_K` | 6-bit | Very High | Large | Near-original |
| `Q8_0` | 8-bit | Excellent | Largest | Maximum quality |
| `F16` | 16-bit | Original | Full size | Reference |

### Size Comparison (7B Model)

| Quantization | Size | Quality Loss |
|--------------|------|--------------|
| F16 | ~14 GB | 0% |
| Q8_0 | ~7 GB | <1% |
| Q5_K_M | ~5 GB | ~2% |
| Q4_K_M | ~4 GB | ~3% |
| Q3_K_M | ~3 GB | ~5% |
| Q2_K | ~2 GB | ~10% |

---

## Export Examples

### Different Quantization Types

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3.2-3B")

# Recommended for most use cases
model.export("gguf", "model.Q4_K_M.gguf", quantization="Q4_K_M")

# Higher quality
model.export("gguf", "model.Q5_K_M.gguf", quantization="Q5_K_M")
model.export("gguf", "model.Q8_0.gguf", quantization="Q8_0")

# Smaller size
model.export("gguf", "model.Q3_K_M.gguf", quantization="Q3_K_M")
model.export("gguf", "model.Q2_K.gguf", quantization="Q2_K")

# Full precision (largest)
model.export("gguf", "model.F16.gguf", quantization="F16")
```

---

## Using Exported Models

### With llama.cpp

```bash
# Download or build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Run your model
./llama-cli -m model.Q4_K_M.gguf -p "Hello, world!" -n 100
```

### With Ollama

```bash
# Create a Modelfile
echo 'FROM ./model.Q4_K_M.gguf' > Modelfile

# Create the model
ollama create mymodel -f Modelfile

# Run
ollama run mymodel
```

### With LM Studio

1. Open LM Studio
2. Go to "My Models" → "Import"
3. Select your `.gguf` file
4. Start chatting!

### With Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(model_path="model.Q4_K_M.gguf")

output = llm(
    "Write a poem about the ocean:",
    max_tokens=100,
    echo=True
)
print(output["choices"][0]["text"])
```

---

## Push to HuggingFace

Export and push in one step:

```python
model.push(
    "your-username/my-model-gguf",
    format="gguf",
    quantization="Q4_K_M",
    license="apache-2.0"
)
```

The model card is automatically generated with:
- Usage examples for llama.cpp, Ollama, LM Studio
- Quantization details
- "Use this model" button compatibility

---

## Direct Conversion

Convert any HuggingFace model without loading into TurboModel:

```python
from quantllm import convert_to_gguf

convert_to_gguf(
    model_path="meta-llama/Llama-3.2-3B",
    output_path="model.Q4_K_M.gguf",
    quant_type="Q4_K_M",
    verbose=True,
)
```

---

## Quantize Existing GGUF

Re-quantize a GGUF file to a different type:

```python
from quantllm import quantize_gguf

quantize_gguf(
    input_path="model.F16.gguf",
    output_path="model.Q4_K_M.gguf",
    quant_type="Q4_K_M"
)
```

---

## List Available Quantization Types

```python
from quantllm import GGUF_QUANT_TYPES, QUANT_RECOMMENDATIONS

# All available types
print(GGUF_QUANT_TYPES)

# Recommendations
print(QUANT_RECOMMENDATIONS)
```

---

## Troubleshooting

### BitsAndBytes Models

If you loaded a model with BitsAndBytes quantization:

```python
# This works - QuantLLM dequantizes automatically
model = turbo("model-name", bits=4)
model.export("gguf", "model.gguf", quantization="Q4_K_M")
```

### Large Models

For very large models:

```python
# Use lower quantization
model.export("gguf", "model.Q3_K_M.gguf", quantization="Q3_K_M")

# Or export with streaming (reduces memory)
model.export("gguf", "model.gguf", quantization="Q4_K_M", streaming=True)
```

### Windows Issues

If you encounter issues on Windows:

1. Install Visual C++ Build Tools
2. Ensure Python 3.10+ is installed
3. Try running as administrator

---

## Best Practices

1. **Use Q4_K_M** for most deployments (best quality/size balance)
2. **Use Q5_K_M or Q6_K** for quality-critical applications
3. **Use Q2_K or Q3_K_M** only when size is critical
4. **Test output quality** after quantization
5. **Keep the F16 version** as a reference

---

## Next Steps

- [Hub Integration →](hub-integration.md) — Push to HuggingFace
- [Other Export Formats →](../quickstart.md#export-to-different-formats) — ONNX, MLX, SafeTensors
- [API Reference →](../api/gguf.md) — Full GGUF API
