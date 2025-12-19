# 🔥 TurboModel

The unified model class for loading, generating, fine-tuning, and exporting.

---

## Class Overview

```python
class TurboModel:
    """Ultra-fast LLM with auto-configuration."""
    
    model: PreTrainedModel           # The underlying HuggingFace model
    tokenizer: PreTrainedTokenizer   # The tokenizer
    config: SmartConfig              # Auto-detected configuration
```

---

## Class Methods

### from_pretrained()

Load a model from HuggingFace Hub or local path.

```python
@classmethod
def from_pretrained(
    cls,
    model_name: str,
    config: Optional[SmartConfig] = None,
    quantize: bool = True,
    verbose: bool = True,
    **kwargs
) -> "TurboModel"
```

**Example:**
```python
from quantllm import TurboModel, SmartConfig

# With auto-config
model = TurboModel.from_pretrained("meta-llama/Llama-3.2-3B")

# With custom config
config = SmartConfig.detect("meta-llama/Llama-3.2-3B", bits=4)
model = TurboModel.from_pretrained("meta-llama/Llama-3.2-3B", config=config)
```

### from_gguf()

Load a GGUF model from HuggingFace or local file.

```python
@classmethod
def from_gguf(
    cls,
    repo_id_or_path: str,
    filename: Optional[str] = None,
    **kwargs
) -> "TurboModel"
```

**Example:**
```python
# From HuggingFace
model = TurboModel.from_gguf(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf"
)

# From local file
model = TurboModel.from_gguf("./models/my-model.gguf")
```

### list_gguf_files()

List available GGUF files in a HuggingFace repository.

```python
@staticmethod
def list_gguf_files(repo_id: str) -> List[str]
```

**Example:**
```python
files = TurboModel.list_gguf_files("TheBloke/Llama-2-7B-Chat-GGUF")
print(files)
# ['llama-2-7b-chat.Q2_K.gguf', 'llama-2-7b-chat.Q4_K_M.gguf', ...]
```

---

## Instance Methods

### generate()

Generate text from a prompt.

```python
def generate(
    self,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
    stream: bool = False,
    stop_strings: Optional[List[str]] = None,
    **kwargs
) -> Union[str, Generator[str, None, None]]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Input text |
| `max_new_tokens` | int | 256 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `top_k` | int | 50 | Top-k sampling |
| `repetition_penalty` | float | 1.0 | Repetition penalty (1.0-1.5) |
| `stream` | bool | False | Stream tokens as generated |
| `stop_strings` | list | None | Stop generation at these strings |

**Example:**
```python
# Basic generation
response = model.generate("What is AI?")

# With parameters
response = model.generate(
    "Write a poem:",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
)

# Streaming
for token in model.generate("Count to 10:", stream=True):
    print(token, end="", flush=True)
```

### chat()

Chat with the model using messages format.

```python
def chat(
    self,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    stream: bool = False,
    **kwargs
) -> Union[str, Generator[str, None, None]]
```

**Messages format:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
]
```

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a coding expert."},
    {"role": "user", "content": "How do I read a file in Python?"},
]

response = model.chat(messages)
print(response)
```

### finetune()

Fine-tune the model with LoRA.

```python
def finetune(
    self,
    data: Union[str, List[Dict], Dataset],
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    output_dir: Optional[str] = None,
    hub_manager: Optional[QuantLLMHubManager] = None,
    **kwargs
) -> Dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | str/list/Dataset | required | Training data |
| `epochs` | int | 3 | Training epochs |
| `batch_size` | int | 4 | Batch size |
| `learning_rate` | float | 2e-4 | Learning rate |
| `lora_r` | int | 8 | LoRA rank |
| `lora_alpha` | int | 16 | LoRA alpha |
| `output_dir` | str | None | Save directory |

**Returns:** Dictionary with `train_loss`, `epochs`, `output_dir`.

**Example:**
```python
# Simple training
result = model.finetune("data.json", epochs=3)

# Advanced
result = model.finetune(
    "data.json",
    epochs=5,
    learning_rate=2e-4,
    lora_r=16,
    lora_alpha=32,
    batch_size=4,
)
```

### export()

Export the model to various formats.

```python
def export(
    self,
    format: str,
    output_path: str,
    quantization: Optional[str] = None,
    **kwargs
) -> str
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `format` | str | "gguf", "onnx", "mlx", "safetensors" |
| `output_path` | str | Output file or directory |
| `quantization` | str | Quantization type (format-specific) |

**Examples:**
```python
# GGUF
model.export("gguf", "model.Q4_K_M.gguf", quantization="Q4_K_M")

# ONNX
model.export("onnx", "./model-onnx/")

# MLX
model.export("mlx", "./model-mlx/", quantization="4bit")

# SafeTensors
model.export("safetensors", "./model-hf/")
```

### push() / push_to_hub()

Push model to HuggingFace Hub.

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

**Example:**
```python
# Push as GGUF
model.push(
    "your-username/my-model",
    format="gguf",
    quantization="Q4_K_M"
)

# Push as MLX
model.push(
    "your-username/my-model-mlx",
    format="mlx",
    quantization="4bit"
)
```

---

## SmartConfig

Auto-detected configuration for optimal performance.

```python
@dataclass
class SmartConfig:
    bits: int = 4
    quant_type: str = "nf4"
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    compile_model: bool = False
    batch_size: int = 4
    max_seq_length: int = 4096
    device: torch.device = "cuda"
    dtype: torch.dtype = torch.float16
```

### SmartConfig.detect()

Auto-detect optimal configuration.

```python
@classmethod
def detect(
    cls,
    model_name: str,
    bits: Optional[int] = None,
    training: bool = False,
) -> SmartConfig
```

**Example:**
```python
from quantllm import SmartConfig

config = SmartConfig.detect("meta-llama/Llama-3.2-3B")
print(f"Bits: {config.bits}")
print(f"Flash Attention: {config.use_flash_attention}")
```

### print_summary()

Print configuration summary.

```python
config.print_summary()
```

**Output:**
```
╔════════════════════════════════════════════════════╗
║          QUANTLLM CONFIGURATION                    ║
╠════════════════════════════════════════════════════╣
║ 📦 Quantization: 4-bit (nf4)                       ║
║ 💾 Memory: CPU Offload Disabled                    ║
║ ⚡ Speed: Flash Attention Enabled                  ║
╚════════════════════════════════════════════════════╝
```

---

## See Also

- [turbo()](turbo.md) — Quick loading function
- [GGUF API](gguf.md) — GGUF export details
- [Hub API](hub.md) — HuggingFace integration
