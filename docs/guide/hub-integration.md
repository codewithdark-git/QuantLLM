# Hub Integration

Push and pull models from HuggingFace Hub.

## Setup

Get your HuggingFace token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```python
from quantllm import HubManager

hub = HubManager(token="hf_your_token_here")
# Or set HF_TOKEN environment variable
```

## Push Model

### Push SafeTensors

```python
from quantllm import turbo, HubManager

# Load and optionally fine-tune
model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Save locally first
model.export("safetensors", "./my-model/")

# Push to Hub
hub = HubManager()
hub.push_model(
    model_path="./my-model/",
    repo_name="my-quantized-model",
    private=False,
    commit_message="Upload quantized model"
)
```

### Push GGUF

```python
# Export to GGUF
model.export("gguf", "model.gguf", quantization="q4_0")

# Push GGUF file
hub.push_model(
    model_path="model.gguf",
    repo_name="my-gguf-model",
)
```

## Pull Model

```python
# Pull from Hub (uses transformers internally)
from quantllm import turbo

model = turbo("your-username/your-model")
```

## Model Card

The HubManager automatically creates a model card. Customize it:

```python
hub.push_model(
    model_path="./my-model/",
    repo_name="my-model",
    commit_message="Fine-tuned with QuantLLM",
)
```

## Private Models

```python
hub.push_model(
    model_path="./my-model/",
    repo_name="my-private-model",
    private=True,  # Private repository
)
```

## Best Practices

1. **Use descriptive names**: `llama2-7b-q4-finetuned`
2. **Add tags**: Include model type, quantization info
3. **Write good READMEs**: Document training details
4. **Test before pushing**: Verify the model works
