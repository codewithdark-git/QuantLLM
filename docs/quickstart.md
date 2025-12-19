# 🚀 Quick Start

Get up and running with QuantLLM in 5 minutes.

---

## Your First Model

```python
from quantllm import turbo

# Load any HuggingFace model with automatic optimization
model = turbo("meta-llama/Llama-3.2-3B")

# Generate text
response = model.generate("Explain machine learning in simple terms")
print(response)
```

**That's it!** QuantLLM automatically:
- ✅ Detects your GPU and available memory
- ✅ Applies optimal 4-bit quantization
- ✅ Enables Flash Attention 2 when available
- ✅ Configures memory management

---

## Basic Usage

### Generate Text

```python
response = model.generate(
    "Write a Python function to calculate fibonacci numbers",
    max_new_tokens=200,
    temperature=0.7,
)
print(response)
```

### Chat Mode

```python
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I read a file in Python?"},
]

response = model.chat(messages, max_new_tokens=200)
print(response)
```

### Streaming Output

```python
for token in model.generate("Count to 10:", stream=True):
    print(token, end="", flush=True)
```

---

## Export to Different Formats

### GGUF (llama.cpp, Ollama, LM Studio)

```python
# Export with recommended Q4_K_M quantization
model.export("gguf", "model.Q4_K_M.gguf", quantization="Q4_K_M")

# Other quantization options
model.export("gguf", "model.Q8_0.gguf", quantization="Q8_0")   # Higher quality
model.export("gguf", "model.Q2_K.gguf", quantization="Q2_K")   # Smallest size
```

### ONNX (ONNX Runtime, TensorRT)

```python
model.export("onnx", "./model-onnx/")
```

### MLX (Apple Silicon)

```python
model.export("mlx", "./model-mlx/", quantization="4bit")
```

### SafeTensors (HuggingFace)

```python
model.export("safetensors", "./model-hf/")
```

---

## Fine-Tune Your Model

Train with your own data in one line:

```python
# Simple training
model.finetune("training_data.json", epochs=3)

# With more control
model.finetune(
    "training_data.json",
    epochs=5,
    learning_rate=2e-4,
    lora_r=16,
    batch_size=4,
)
```

**Supported data formats:**

```json
[
  {"instruction": "What is Python?", "output": "Python is a programming language..."},
  {"text": "Full text for language modeling"},
  {"prompt": "Question here", "completion": "Answer here"}
]
```

---

## Push to HuggingFace

Share your model with the world:

```python
# Push with auto-generated model card
model.push(
    "your-username/my-awesome-model",
    format="gguf",
    quantization="Q4_K_M",
    license="apache-2.0"
)
```

The model card includes:
- ✅ Proper YAML frontmatter for HuggingFace
- ✅ Format-specific usage examples
- ✅ "Use this model" button compatibility
- ✅ Quantization details

---

## Configuration Options

### Override Auto-Detection

```python
model = turbo(
    "meta-llama/Llama-3.2-3B",
    bits=4,                    # Force 4-bit quantization
    max_length=4096,           # Context length
    device="cuda:0",           # Specific GPU
    dtype="bfloat16",          # Data type
)
```

### View Current Configuration

```python
print(model.config)
```

---

## Load GGUF Models

Load pre-quantized GGUF models directly:

```python
from quantllm import TurboModel

model = TurboModel.from_gguf(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf"
)

print(model.generate("Hello!"))
```

---

## Show the Banner

Display the QuantLLM banner anytime:

```python
import quantllm

quantllm.show_banner()
```

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   🚀 QuantLLM v2.0.0                                       ║
║   Ultra-fast LLM Quantization & Export                     ║
║                                                            ║
║   ✓ GGUF  ✓ ONNX  ✓ MLX  ✓ SafeTensors                     ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

## Next Steps

Now that you know the basics, explore more:

- [Loading Models →](guide/loading-models.md) — Advanced model loading options
- [Text Generation →](guide/generation.md) — Generation parameters and modes
- [GGUF Export →](guide/gguf-export.md) — All quantization types explained
- [Fine-tuning →](guide/finetuning.md) — Training with LoRA
- [Hub Integration →](guide/hub-integration.md) — Push and pull from HuggingFace
- [API Reference →](api/turbomodel.md) — Full API documentation
