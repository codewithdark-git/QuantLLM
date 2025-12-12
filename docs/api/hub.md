# HubManager API

Push models to HuggingFace Hub.

## HubManager

```python
class HubManager:
    def __init__(self, token: Optional[str] = None):
        """
        Initialize HubManager.
        
        Args:
            token: HuggingFace token. If None, uses HF_TOKEN env var.
        """
```

## Methods

### push_model()

Push a model to HuggingFace Hub.

```python
def push_model(
    self,
    model_path: str,
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload model",
) -> str
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to model directory or file |
| `repo_name` | str | required | Repository name |
| `private` | bool | False | Create private repo |
| `commit_message` | str | "Upload model" | Commit message |

**Returns:** URL of the uploaded model.

**Example:**
```python
from quantllm import HubManager

hub = HubManager(token="hf_xxx")

# Push directory
hub.push_model("./my-model/", "my-username/my-model")

# Push single file
hub.push_model("model.gguf", "my-username/my-gguf")
```

### login()

Login to HuggingFace Hub.

```python
def login(self, token: Optional[str] = None):
    """Login to HuggingFace Hub."""
```

### create_repo()

Create a new repository.

```python
def create_repo(
    self,
    repo_name: str,
    private: bool = False,
) -> str
```

## Environment Variables

- `HF_TOKEN` - HuggingFace authentication token
- `HUGGINGFACE_TOKEN` - Alternative token variable

## Example Workflow

```python
from quantllm import turbo, HubManager

# 1. Load and prepare model
model = turbo("TinyLlama/TinyLlama-1.1B")

# 2. Fine-tune (optional)
model.finetune(data=my_data, epochs=3)

# 3. Export
model.export("safetensors", "./my-model/")

# 4. Push to Hub
hub = HubManager()
url = hub.push_model("./my-model/", "my-finetuned-model")
print(f"Model available at: {url}")
```
