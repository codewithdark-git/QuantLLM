# 🤗 Hub API

Push models to HuggingFace Hub with auto-generated model cards.

---

## Quick Reference

```python
from quantllm import turbo, QuantLLMHubManager

# Method 1: TurboModel.push() (Recommended)
model = turbo("meta-llama/Llama-3.2-3B")
model.push("user/my-model", format="gguf", quantization="Q4_K_M")

# Method 2: QuantLLMHubManager (Advanced)
manager = QuantLLMHubManager("user/my-model", hf_token="hf_...")
manager.save_final_model(model)
manager.push()
```

---

## TurboModel.push()

The simplest way to push models.

```python
def push(
    self,
    repo_id: str,
    token: Optional[str] = None,
    format: str = "safetensors",
    quantization: Optional[str] = None,
    license: str = "apache-2.0",
    commit_message: str = "Upload model via QuantLLM",
    **kwargs
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_id` | str | required | HuggingFace repo ID (user/model) |
| `token` | str | None | HF token (or use HF_TOKEN env) |
| `format` | str | "safetensors" | Export format |
| `quantization` | str | None | Quantization type |
| `license` | str | "apache-2.0" | License type |

### Supported Formats

| Format | Description |
|--------|-------------|
| `"safetensors"` | HuggingFace Transformers (default) |
| `"gguf"` | llama.cpp, Ollama, LM Studio |
| `"onnx"` | ONNX Runtime, TensorRT |
| `"mlx"` | Apple Silicon (M1/M2/M3/M4) |

### Examples

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3.2-3B")

# Push as GGUF
model.push(
    "your-username/llama-3.2-3b-gguf",
    format="gguf",
    quantization="Q4_K_M"
)

# Push as ONNX
model.push(
    "your-username/llama-3.2-3b-onnx",
    format="onnx"
)

# Push as MLX
model.push(
    "your-username/llama-3.2-3b-mlx",
    format="mlx",
    quantization="4bit"
)

# Push as SafeTensors (default)
model.push("your-username/llama-3.2-3b")
```

---

## QuantLLMHubManager

Advanced hub management with hyperparameter tracking.

```python
class QuantLLMHubManager:
    def __init__(
        self,
        repo_id: str,
        hf_token: Optional[str] = None,
        organization: Optional[str] = None
    )
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `repo_id` | str | HuggingFace repo ID (user/model) |
| `hf_token` | str | HuggingFace API token |
| `organization` | str | Optional organization name |

### Methods

#### login()

Verify authentication with HuggingFace.

```python
manager.login()
```

#### track_hyperparameters()

Track training hyperparameters for the model card.

```python
def track_hyperparameters(self, params: Dict[str, Any])
```

**Example:**
```python
manager.track_hyperparameters({
    "epochs": 3,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "base_model": "meta-llama/Llama-3.2-3B",
})
```

#### save_final_model()

Save model to staging directory.

```python
def save_final_model(
    self,
    model,
    tokenizer=None,
    format: str = "safetensors"
)
```

#### push()

Push staged model to HuggingFace Hub.

```python
def push(self, commit_message: str = "Upload model via QuantLLM")
```

---

## Complete Workflow

### Fine-Tune and Push

```python
from quantllm import turbo, QuantLLMHubManager

# Load model
model = turbo("meta-llama/Llama-3.2-3B")

# Create manager
manager = QuantLLMHubManager(
    "your-username/my-finetuned-model",
    hf_token="hf_..."
)

# Fine-tune with tracking
model.finetune(
    "data.json",
    epochs=3,
    hub_manager=manager  # Auto-tracks hyperparameters
)

# Save and push
manager.save_final_model(model)
manager.push(commit_message="Fine-tuned on custom dataset")
```

### Export and Push

```python
from quantllm import turbo, QuantLLMHubManager
import os

model = turbo("meta-llama/Llama-3.2-3B")
manager = QuantLLMHubManager("your-username/my-gguf", "hf_...")

# Export multiple quantizations
for quant in ["Q4_K_M", "Q5_K_M", "Q8_0"]:
    output = os.path.join(manager.staging_dir, f"model.{quant}.gguf")
    model.export("gguf", output, quantization=quant)

# Track metadata
manager.track_hyperparameters({
    "format": "gguf",
    "base_model": "meta-llama/Llama-3.2-3B",
    "quantizations": ["Q4_K_M", "Q5_K_M", "Q8_0"],
})

manager.push()
```

---

## Auto-Generated Model Cards

QuantLLM automatically generates professional model cards with:

### YAML Frontmatter

```yaml
---
license: apache-2.0
base_model: meta-llama/Llama-3.2-3B
library_name: gguf
language:
  - en
tags:
  - quantllm
  - gguf
  - llama-cpp
  - q4_k_m
---
```

### Format-Specific Usage

For **GGUF**:
```python
from llama_cpp import Llama
llm = Llama.from_pretrained(repo_id="user/model", filename="model.Q4_K_M.gguf")
```

For **MLX**:
```python
from mlx_lm import load, generate
model, tokenizer = load("user/model")
text = generate(model, tokenizer, prompt="Hello!")
```

For **ONNX**:
```python
from optimum.onnxruntime import ORTModelForCausalLM
model = ORTModelForCausalLM.from_pretrained("user/model")
```

---

## ModelCardGenerator

Generate custom model cards.

```python
from quantllm.hub import ModelCardGenerator, generate_model_card

# Quick function
content = generate_model_card(
    repo_id="user/my-model",
    base_model="meta-llama/Llama-3.2-3B",
    format="gguf",
    quantization="Q4_K_M",
    license="apache-2.0",
)

# Or use the class for more control
generator = ModelCardGenerator(
    repo_id="user/my-model",
    base_model="meta-llama/Llama-3.2-3B",
    format="gguf",
    quantization="Q4_K_M",
    license="apache-2.0",
    language=["en", "es"],
    tags=["finetuned", "code"],
)
content = generator.generate()
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token |
| `HUGGINGFACE_TOKEN` | Alternative token variable |
| `HF_HUB_DISABLE_PROGRESS_BARS` | Disable progress bars |

---

## Best Practices

1. **Use descriptive names**: `llama-3.2-3b-code-q4_k_m`
2. **Include format suffix**: `-gguf`, `-onnx`, `-mlx`
3. **Test before pushing**: Verify the model works
4. **Use appropriate license**: Match your base model's license
5. **Write good commit messages**: Describe what changed

---

## See Also

- [Hub Integration Guide](../guide/hub-integration.md) — Detailed guide
- [TurboModel.push()](turbomodel.md#push--push_to_hub) — Push via TurboModel
- [GGUF Export](gguf.md) — GGUF conversion details
