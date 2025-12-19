"""
Model Card Generator for QuantLLM

Generates proper HuggingFace model cards with format-specific metadata,
usage examples, and "Use this model" button compatibility.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime


class ModelCardGenerator:
    """
    Generate HuggingFace-compatible model cards for different export formats.
    
    Supports:
        - GGUF (llama.cpp, Ollama, LM Studio)
        - ONNX (ONNX Runtime, TensorRT)
        - MLX (Apple Silicon)
        - SafeTensors (Transformers)
    """
    
    # Format-specific library names for HuggingFace
    LIBRARY_NAMES = {
        "gguf": "gguf",
        "onnx": "onnx",
        "mlx": "mlx",
        "safetensors": "transformers",
        "pytorch": "transformers",
    }
    
    # Format-specific tags
    FORMAT_TAGS = {
        "gguf": ["gguf", "llama-cpp", "quantized", "transformers"],
        "onnx": ["onnx", "onnxruntime", "transformers"],
        "mlx": ["mlx", "mlx-lm", "apple-silicon", "transformers"],
        "safetensors": ["transformers", "safetensors"],
        "pytorch": ["transformers", "pytorch"],
    }
    
    def __init__(
        self,
        repo_id: str,
        base_model: str,
        format: str,
        quantization: Optional[str] = None,
        license: str = "apache-2.0",
        language: List[str] = None,
        tags: List[str] = None,
        **kwargs
    ):
        """
        Initialize model card generator.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., "user/model-name")
            base_model: Original base model name
            format: Export format (gguf, onnx, mlx, safetensors)
            quantization: Quantization type (e.g., Q4_K_M, 4bit)
            license: License type
            language: List of languages
            tags: Additional tags
        """
        self.repo_id = repo_id
        self.base_model = base_model
        self.format = format.lower()
        self.quantization = quantization
        self.license = license
        self.language = language or ["en"]
        self.extra_tags = tags or []
        self.extra_params = kwargs
        
    def generate(self) -> str:
        """Generate the complete model card content."""
        sections = [
            self._generate_yaml_header(),
            self._generate_title(),
            self._generate_description(),
            self._generate_usage_section(),
            self._generate_details_section(),
            self._generate_quantization_section(),
            self._generate_footer(),
        ]
        
        return "\n".join(sections)
    
    def _generate_yaml_header(self) -> str:
        """Generate YAML frontmatter for HuggingFace."""
        library_name = self.LIBRARY_NAMES.get(self.format, "transformers")
        
        # Collect all tags
        tags = ["quantllm"]
        tags.extend(self.FORMAT_TAGS.get(self.format, []))
        if self.quantization:
            tags.append(f"{self.quantization.lower()}")
        tags.extend(self.extra_tags)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        yaml_lines = [
            "---",
            f"license: {self.license}",
            f"base_model: {self.base_model}",
            f"library_name: {library_name}",
        ]
        
        # Add language
        if self.language:
            yaml_lines.append("language:")
            for lang in self.language:
                yaml_lines.append(f"  - {lang}")
        
        # Add tags
        yaml_lines.append("tags:")
        for tag in unique_tags:
            yaml_lines.append(f"  - {tag}")
        
        # Add pipeline tag for transformers
        if self.format in ["safetensors", "pytorch"]:
            yaml_lines.append("pipeline_tag: text-generation")
        
        yaml_lines.append("---")
        
        return "\n".join(yaml_lines)
    
    def _generate_title(self) -> str:
        """Generate model title section."""
        model_name = self.repo_id.split('/')[-1]
        format_upper = self.format.upper()
        
        title = f"\n# {model_name}\n"
        
        # Add badges
        badges = []
        badges.append(f"![Format](https://img.shields.io/badge/format-{format_upper}-orange)")
        if self.quantization:
            badges.append(f"![Quantization](https://img.shields.io/badge/quantization-{self.quantization}-blue)")
        badges.append("![QuantLLM](https://img.shields.io/badge/made%20with-QuantLLM-green)")
        
        title += " ".join(badges) + "\n"
        
        return title
    
    def _generate_description(self) -> str:
        """Generate description section."""
        format_desc = {
            "gguf": "GGUF format for use with llama.cpp, Ollama, LM Studio, and other compatible tools",
            "onnx": "ONNX format for use with ONNX Runtime, TensorRT, and other inference engines",
            "mlx": "MLX format optimized for Apple Silicon (M1/M2/M3) Macs",
            "safetensors": "SafeTensors format for use with HuggingFace Transformers",
        }
        
        desc = f"""
## Description

This is **{self.base_model}** converted to {format_desc.get(self.format, self.format.upper() + ' format')}.

- **Base Model**: [{self.base_model}](https://huggingface.co/{self.base_model})
- **Format**: {self.format.upper()}
"""
        if self.quantization:
            desc += f"- **Quantization**: {self.quantization}\n"
        
        desc += f"- **Created with**: [QuantLLM](https://github.com/codewithdark-git/QuantLLM)\n"
        
        return desc
    
    def _generate_usage_section(self) -> str:
        """Generate format-specific usage examples."""
        if self.format == "gguf":
            return self._generate_gguf_usage()
        elif self.format == "onnx":
            return self._generate_onnx_usage()
        elif self.format == "mlx":
            return self._generate_mlx_usage()
        else:
            return self._generate_transformers_usage()
    
    def _generate_gguf_usage(self) -> str:
        """Generate GGUF usage examples."""
        model_name = self.repo_id.split('/')[-1]
        quant = self.quantization or "Q4_K_M"
        filename = f"{model_name}.{quant}.gguf"
        
        return f'''
## Usage

### With llama.cpp

```bash
# Download the model
huggingface-cli download {self.repo_id} {filename} --local-dir .

# Run with llama.cpp
./llama-cli -m {filename} -p "Hello, how are you?" -n 128
```

### With Ollama

```bash
# Create a Modelfile
echo 'FROM ./{filename}' > Modelfile

# Create the model
ollama create {model_name.lower()} -f Modelfile

# Run
ollama run {model_name.lower()}
```

### With LM Studio

1. Download the `.gguf` file from this repository
2. Open LM Studio and go to the Models tab
3. Click "Add Model" and select the downloaded file
4. Start chatting!

### With Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="{self.repo_id}",
    filename="{filename}",
)

output = llm(
    "Write a story about a robot:",
    max_tokens=256,
    echo=True
)
print(output["choices"][0]["text"])
```
'''
    
    def _generate_onnx_usage(self) -> str:
        """Generate ONNX usage examples."""
        return f'''
## Usage

### With Optimum (Recommended)

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model = ORTModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With ONNX Runtime

```python
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")
session = ort.InferenceSession("model.onnx")

inputs = tokenizer("Hello!", return_tensors="np")
outputs = session.run(None, dict(inputs))
```

### With TensorRT (for NVIDIA GPUs)

```python
# Convert ONNX to TensorRT
import tensorrt as trt
# ... TensorRT conversion code
```
'''
    
    def _generate_mlx_usage(self) -> str:
        """Generate MLX usage examples for Apple Silicon."""
        return f'''
## Usage

### Generate text with mlx-lm

```python
from mlx_lm import load, generate

model, tokenizer = load("{self.repo_id}")

prompt = "Write a story about Einstein"
messages = [{{"role": "user", "content": prompt}}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

### With streaming

```python
from mlx_lm import load, stream_generate

model, tokenizer = load("{self.repo_id}")

prompt = "Explain quantum computing"
messages = [{{"role": "user", "content": prompt}}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

for token in stream_generate(model, tokenizer, prompt=prompt, max_tokens=500):
    print(token, end="", flush=True)
```

### Command Line

```bash
# Install mlx-lm
pip install mlx-lm

# Generate text
python -m mlx_lm.generate --model {self.repo_id} --prompt "Hello!"

# Chat mode
python -m mlx_lm.chat --model {self.repo_id}
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13.0 or later
- Python 3.10+
- mlx-lm: `pip install mlx-lm`
'''
    
    def _generate_transformers_usage(self) -> str:
        """Generate Transformers usage examples."""
        return f'''
## Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With QuantLLM

```python
from quantllm import TurboModel

model = TurboModel.from_pretrained("{self.repo_id}")
response = model.generate("Hello, how are you?")
print(response)
```
'''
    
    def _generate_details_section(self) -> str:
        """Generate model details section."""
        return f'''
## Model Details

| Property | Value |
|----------|-------|
| Base Model | [{self.base_model}](https://huggingface.co/{self.base_model}) |
| Format | {self.format.upper()} |
| Quantization | {self.quantization or "N/A"} |
| License | {self.license} |
| Created | {datetime.now().strftime("%Y-%m-%d")} |
'''
    
    def _generate_quantization_section(self) -> str:
        """Generate quantization details for GGUF."""
        if self.format != "gguf" or not self.quantization:
            return ""
        
        quant_info = {
            "Q2_K": ("2-bit", "Smallest size, lowest quality"),
            "Q3_K_S": ("3-bit", "Very small, low quality"),
            "Q3_K_M": ("3-bit", "Small size, acceptable quality"),
            "Q3_K_L": ("3-bit", "Slightly larger, better quality"),
            "Q4_K_S": ("4-bit", "Small size, good quality"),
            "Q4_K_M": ("4-bit", "**Recommended** - Best balance of size and quality"),
            "Q5_K_S": ("5-bit", "Medium size, very good quality"),
            "Q5_K_M": ("5-bit", "Good size-quality trade-off"),
            "Q6_K": ("6-bit", "Large size, excellent quality"),
            "Q8_0": ("8-bit", "Near full precision quality"),
            "F16": ("16-bit", "Full precision, largest size"),
        }
        
        quant = self.quantization.upper()
        info = quant_info.get(quant, ("N/A", "Custom quantization"))
        
        return f'''
## Quantization Details

- **Type**: {quant}
- **Bits**: {info[0]}
- **Description**: {info[1]}

### Available Quantizations

| Quantization | Bits | Use Case |
|--------------|------|----------|
| Q2_K | 2-bit | Minimum size, experimental |
| Q3_K_M | 3-bit | Very constrained environments |
| Q4_K_M | 4-bit | **Recommended** for most users |
| Q5_K_M | 5-bit | Higher quality, more memory |
| Q6_K | 6-bit | Near-original quality |
| Q8_0 | 8-bit | Best quality, largest size |
'''
    
    def _generate_footer(self) -> str:
        """Generate footer section."""
        return f'''
---

## About QuantLLM

This model was converted using [QuantLLM](https://github.com/codewithdark-git/QuantLLM) - 
the ultra-fast LLM quantization and export library.

```python
from quantllm import turbo

# Load and quantize any model
model = turbo("{self.base_model}")

# Export to any format
model.export("{self.format}", quantization="{self.quantization or 'Q4_K_M'}")
```

⭐ Star us on [GitHub](https://github.com/codewithdark-git/QuantLLM)!
'''


def generate_model_card(
    repo_id: str,
    base_model: str,
    format: str,
    quantization: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to generate a model card.
    
    Args:
        repo_id: HuggingFace repo ID
        base_model: Original base model
        format: Export format (gguf, onnx, mlx, safetensors)
        quantization: Quantization type
        **kwargs: Additional parameters
        
    Returns:
        Model card content as string
    """
    generator = ModelCardGenerator(
        repo_id=repo_id,
        base_model=base_model,
        format=format,
        quantization=quantization,
        **kwargs
    )
    return generator.generate()

