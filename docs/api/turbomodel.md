# TurboModel

The main model class for loading, generating, fine-tuning, and exporting.

## Class Definition

```python
class TurboModel:
    """Ultra-fast LLM with auto-configuration."""
    
    model: PreTrainedModel      # The underlying HuggingFace model
    tokenizer: PreTrainedTokenizer  # The tokenizer
    config: SmartConfig         # Auto-detected configuration
```

## Methods

### from_pretrained()

Load a model from HuggingFace Hub or local path.

```python
@classmethod
def from_pretrained(
    cls,
    model_name: str,
    config: Optional[SmartConfig] = None,
    **kwargs
) -> "TurboModel"
```

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

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Input text |
| `max_new_tokens` | int | 256 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.9 | Nucleus sampling |
| `top_k` | int | 50 | Top-k sampling |
| `repetition_penalty` | float | 1.0 | Penalty for repetition |
| `stream` | bool | False | Stream tokens |
| `stop_strings` | list | None | Stop generation strings |

### chat()

Chat with the model using messages format.

```python
def chat(
    self,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    **kwargs
) -> str
```

**Messages format:**
```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
]
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
    output_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]
```

**Returns:** Dictionary with `train_loss`, `epochs`, `output_dir`.

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

**Formats:**
- `"gguf"` - GGUF format (llama.cpp, Ollama)
- `"safetensors"` - SafeTensors format
- `"onnx"` - ONNX format

---

## SmartConfig

Auto-detected configuration for optimal performance.

```python
@dataclass
class SmartConfig:
    bits: int = 4
    quant_type: str = "Q4_K_M"
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    compile_model: bool = False
    batch_size: int = 4
    max_seq_length: int = 4096
    device: torch.device
    dtype: torch.dtype
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

### print_summary()

Print configuration summary.

```python
config.print_summary()
```

Output:
```
==================================================
         QUANTLLM CONFIGURATION          
==================================================

📦 Quantization:
   Bits:       4
   Type:       Q4_K_M

💾 Memory:
   CPU Offload:      Disabled (Fast)
   Grad Checkpoint:  Disabled

⚡ Speed:
   Flash Attention:  Enabled
   Fused Kernels:    Enabled
   torch.compile:    Disabled (Optional)
```
