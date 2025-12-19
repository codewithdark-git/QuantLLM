# 🤗 Hub Integration

Push and pull models from HuggingFace Hub with auto-generated model cards.

---

## Quick Push

The easiest way to share your model:

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3.2-3B")

# Push with auto-generated model card
model.push(
    "your-username/my-model",
    token="hf_...",
    format="gguf",
    quantization="Q4_K_M"
)
```

---

## Setup

### Get Your Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "Write" permissions
3. Use it in your code or set as environment variable

```bash
# Option 1: Environment variable
export HF_TOKEN=hf_your_token_here

# Option 2: Pass directly
model.push("user/repo", token="hf_...")
```

---

## Push Methods

### Method 1: TurboModel.push() (Recommended)

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3.2-3B")

# Push as GGUF (for Ollama, llama.cpp, LM Studio)
model.push(
    "your-username/my-model-gguf",
    format="gguf",
    quantization="Q4_K_M",
    license="apache-2.0"
)

# Push as ONNX
model.push(
    "your-username/my-model-onnx",
    format="onnx"
)

# Push as MLX (Apple Silicon)
model.push(
    "your-username/my-model-mlx",
    format="mlx",
    quantization="4bit"
)

# Push as SafeTensors (default)
model.push(
    "your-username/my-model",
    format="safetensors"
)
```

### Method 2: QuantLLMHubManager (Advanced)

For more control:

```python
from quantllm import turbo, QuantLLMHubManager

model = turbo("meta-llama/Llama-3.2-3B")

# Create manager
manager = QuantLLMHubManager(
    repo_id="your-username/my-model",
    hf_token="hf_..."
)

# Track hyperparameters (for fine-tuned models)
manager.track_hyperparameters({
    "epochs": 3,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "base_model": "meta-llama/Llama-3.2-3B"
})

# Save model
manager.save_final_model(model)

# Push
manager.push(commit_message="Upload fine-tuned model")
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

### Format-Specific Usage Examples

For **GGUF** models:
```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="your-username/my-model",
    filename="model.Q4_K_M.gguf",
)
```

For **MLX** models:
```python
from mlx_lm import load, generate

model, tokenizer = load("your-username/my-model")
text = generate(model, tokenizer, prompt="Hello!")
```

For **ONNX** models:
```python
from optimum.onnxruntime import ORTModelForCausalLM

model = ORTModelForCausalLM.from_pretrained("your-username/my-model")
```

---

## Pull Models

Load models from HuggingFace:

```python
from quantllm import turbo, TurboModel

# Load regular models
model = turbo("your-username/my-model")

# Load GGUF models
model = TurboModel.from_gguf(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf"
)
```

### List GGUF Files

```python
files = TurboModel.list_gguf_files("TheBloke/Llama-2-7B-Chat-GGUF")
print(files)
# ['llama-2-7b-chat.Q2_K.gguf', 'llama-2-7b-chat.Q4_K_M.gguf', ...]
```

---

## Private Repositories

```python
# Push to private repo
model.push(
    "your-username/private-model",
    token="hf_...",
    private=True  # Makes repository private
)
```

---

## Fine-Tuning with Hub Tracking

Automatically track training hyperparameters:

```python
from quantllm import turbo, QuantLLMHubManager

model = turbo("meta-llama/Llama-3.2-3B")
manager = QuantLLMHubManager("user/repo", "hf_token")

# Train with automatic tracking
model.finetune(
    "data.json",
    epochs=3,
    hub_manager=manager  # Tracks all hyperparameters
)

# Push with full history
manager.save_final_model(model)
manager.push()
```

---

## Commit Messages

Customize commit messages:

```python
model.push(
    "user/repo",
    commit_message="v2.0 - Improved accuracy on coding tasks"
)
```

---

## Multiple Formats

Upload multiple formats to the same repo:

```python
manager = QuantLLMHubManager("user/my-model", token)

# Export multiple formats to staging
model.export("gguf", f"{manager.staging_dir}/model.Q4_K_M.gguf", quantization="Q4_K_M")
model.export("gguf", f"{manager.staging_dir}/model.Q8_0.gguf", quantization="Q8_0")

manager.push(commit_message="Upload Q4_K_M and Q8_0 variants")
```

---

## Best Practices

1. **Use descriptive names**: `llama-3.2-3b-code-assistant-q4`
2. **Include format in name**: `-gguf`, `-onnx`, `-mlx`
3. **Add quantization**: `-q4_k_m`, `-8bit`
4. **Write good commit messages**: Describe what changed
5. **Test before pushing**: Verify the model works

---

## Troubleshooting

### Authentication Error

```python
# Make sure your token has write permissions
# Check at: huggingface.co/settings/tokens
```

### Repository Already Exists

```python
# Use exist_ok=True (default)
model.push("user/existing-repo")  # Will update existing repo
```

### Large File Issues

```bash
# Install git-lfs for large files
git lfs install
```

---

## Next Steps

- [GGUF Export →](gguf-export.md) — Learn about GGUF quantization
- [Fine-tuning →](finetuning.md) — Train your own model
- [API Reference →](../api/hub.md) — Full Hub API
