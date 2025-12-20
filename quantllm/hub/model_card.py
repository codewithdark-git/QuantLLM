"""
Model Card Generator for QuantLLM

Generates professional HuggingFace model cards with format-specific metadata,
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
            self._generate_header_banner(),
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
    
    def _generate_header_banner(self) -> str:
        """Generate prominent header with QuantLLM branding."""
        model_name = self.repo_id.split('/')[-1]
        format_upper = self.format.upper()
        
        # Format-specific emoji
        format_emoji = {
            "gguf": "🦙",
            "onnx": "⚡",
            "mlx": "🍎",
            "safetensors": "🤗",
        }
        emoji = format_emoji.get(self.format, "📦")
        
        return f'''
<div align="center">

# {emoji} {model_name}

**{self.base_model}** converted to **{format_upper}** format

[![QuantLLM](https://img.shields.io/badge/🚀_Made_with-QuantLLM-orange?style=for-the-badge)](https://github.com/codewithdark-git/QuantLLM)
[![Format](https://img.shields.io/badge/Format-{format_upper}-blue?style=for-the-badge)]()
{f'[![Quantization](https://img.shields.io/badge/Quant-{self.quantization}-green?style=for-the-badge)]()' if self.quantization else ''}

<a href="https://github.com/codewithdark-git/QuantLLM">⭐ Star QuantLLM on GitHub</a>

</div>

---
'''
    
    def _generate_description(self) -> str:
        """Generate description section."""
        format_desc = {
            "gguf": "**GGUF** format for use with llama.cpp, Ollama, LM Studio, and other compatible inference engines",
            "onnx": "**ONNX** format for use with ONNX Runtime, TensorRT, and cross-platform deployment",
            "mlx": "**MLX** format optimized for Apple Silicon (M1/M2/M3/M4) Macs with native acceleration",
            "safetensors": "**SafeTensors** format for use with HuggingFace Transformers and PyTorch",
        }
        
        return f'''
## 📖 About This Model

This model is **[{self.base_model}](https://huggingface.co/{self.base_model})** converted to {format_desc.get(self.format, self.format.upper() + ' format')}.

| Property | Value |
|----------|-------|
| **Base Model** | [{self.base_model}](https://huggingface.co/{self.base_model}) |
| **Format** | {self.format.upper()} |
| **Quantization** | {self.quantization or "None (Full Precision)"} |
| **License** | {self.license} |
| **Created With** | [QuantLLM](https://github.com/codewithdark-git/QuantLLM) |
'''
    
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
## 🚀 Quick Start

### Option 1: Python (llama-cpp-python)

```python
from llama_cpp import Llama

# Load the model
llm = Llama.from_pretrained(
    repo_id="{self.repo_id}",
    filename="{filename}",
)

# Generate text
output = llm(
    "Write a short story about a robot learning to paint:",
    max_tokens=256,
    echo=True
)
print(output["choices"][0]["text"])
```

### Option 2: Ollama

```bash
# Download the model
huggingface-cli download {self.repo_id} {filename} --local-dir .

# Create Modelfile
echo 'FROM ./{filename}' > Modelfile

# Import to Ollama
ollama create {model_name.lower()} -f Modelfile

# Chat with the model
ollama run {model_name.lower()}
```

### Option 3: LM Studio

1. Download the `.gguf` file from the **Files** tab above
2. Open **LM Studio** → **My Models** → **Add Model**
3. Select the downloaded file
4. Start chatting!

### Option 4: llama.cpp CLI

```bash
# Download
huggingface-cli download {self.repo_id} {filename} --local-dir .

# Run inference
./llama-cli -m {filename} -p "Hello! " -n 128
```
'''
    
    def _generate_onnx_usage(self) -> str:
        """Generate ONNX usage examples."""
        return f'''
## 🚀 Quick Start

### Option 1: Optimum (Recommended)

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# Load model and tokenizer
model = ORTModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

# Generate text
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option 2: ONNX Runtime Direct

```python
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer and session
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")
session = ort.InferenceSession("model.onnx")

# Tokenize and run
inputs = tokenizer("Hello!", return_tensors="np")
outputs = session.run(None, dict(inputs))
```

### Requirements

```bash
pip install optimum[onnxruntime] transformers
```
'''
    
    def _generate_mlx_usage(self) -> str:
        """Generate MLX usage examples for Apple Silicon."""
        return f'''
## 🚀 Quick Start

### Generate Text with mlx-lm

```python
from mlx_lm import load, generate

# Load the model
model, tokenizer = load("{self.repo_id}")

# Simple generation
prompt = "Explain quantum computing in simple terms"
messages = [{{"role": "user", "content": prompt}}]
prompt_formatted = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True
)

# Generate response
text = generate(model, tokenizer, prompt=prompt_formatted, verbose=True)
print(text)
```

### Streaming Generation

```python
from mlx_lm import load, stream_generate

model, tokenizer = load("{self.repo_id}")

prompt = "Write a haiku about coding"
messages = [{{"role": "user", "content": prompt}}]
prompt_formatted = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True
)

# Stream tokens as they're generated
for token in stream_generate(model, tokenizer, prompt=prompt_formatted, max_tokens=200):
    print(token, end="", flush=True)
```

### Command Line Interface

```bash
# Install mlx-lm
pip install mlx-lm

# Generate text
python -m mlx_lm.generate --model {self.repo_id} --prompt "Hello!"

# Interactive chat
python -m mlx_lm.chat --model {self.repo_id}
```

### System Requirements

| Requirement | Minimum |
|-------------|---------|
| **Chip** | Apple Silicon (M1/M2/M3/M4) |
| **macOS** | 13.0 (Ventura) or later |
| **Python** | 3.10+ |
| **RAM** | 8GB+ (16GB recommended) |

```bash
# Install dependencies
pip install mlx-lm
```
'''
    
    def _generate_transformers_usage(self) -> str:
        """Generate Transformers usage examples."""
        return f'''
## 🚀 Quick Start

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

# Generate text
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With QuantLLM

```python
from quantllm import TurboModel

# Load with automatic optimization
model = TurboModel.from_pretrained("{self.repo_id}")

# Generate
response = model.generate("Write a poem about coding")
print(response)
```

### Requirements

```bash
pip install transformers torch
```
'''
    
    def _generate_details_section(self) -> str:
        """Generate model details section."""
        return f'''
## 📊 Model Details

| Property | Value |
|----------|-------|
| **Original Model** | [{self.base_model}](https://huggingface.co/{self.base_model}) |
| **Format** | {self.format.upper()} |
| **Quantization** | {self.quantization or "Full Precision"} |
| **License** | `{self.license}` |
| **Export Date** | {datetime.now().strftime("%Y-%m-%d")} |
| **Exported By** | [QuantLLM v2.0](https://github.com/codewithdark-git/QuantLLM) |
'''
    
    def _generate_quantization_section(self) -> str:
        """Generate quantization details for GGUF."""
        if self.format != "gguf" or not self.quantization:
            return ""
        
        quant_info = {
            "Q2_K": ("2-bit", "Smallest file size, experimental quality", "🔴"),
            "Q3_K_S": ("3-bit", "Very small, reduced quality", "🟠"),
            "Q3_K_M": ("3-bit", "Small size, acceptable quality", "🟠"),
            "Q3_K_L": ("3-bit", "Slightly larger, better quality", "🟠"),
            "Q4_K_S": ("4-bit", "Good balance, slightly smaller", "🟡"),
            "Q4_K_M": ("4-bit", "⭐ Recommended - Best quality/size balance", "🟢"),
            "Q5_K_S": ("5-bit", "Higher quality, moderate size", "🟢"),
            "Q5_K_M": ("5-bit", "High quality, good performance", "🟢"),
            "Q6_K": ("6-bit", "Very high quality, larger size", "🔵"),
            "Q8_0": ("8-bit", "Near-original quality, largest size", "🔵"),
            "F16": ("16-bit", "Full precision, reference quality", "⚪"),
        }
        
        quant = self.quantization.upper()
        info = quant_info.get(quant, ("N/A", "Custom quantization", "⚪"))
        
        return f'''
## 📦 Quantization Details

This model uses **{quant}** quantization:

| Property | Value |
|----------|-------|
| **Type** | {quant} |
| **Bits** | {info[0]} |
| **Quality** | {info[2]} {info[1]} |

### All Available GGUF Quantizations

| Type | Bits | Quality | Best For |
|------|------|---------|----------|
| Q2_K | 2-bit | 🔴 Lowest | Extreme size constraints |
| Q3_K_M | 3-bit | 🟠 Low | Very limited memory |
| Q4_K_M | 4-bit | 🟢 Good | **Most users** ⭐ |
| Q5_K_M | 5-bit | 🟢 High | Quality-focused |
| Q6_K | 6-bit | 🔵 Very High | Near-original |
| Q8_0 | 8-bit | 🔵 Excellent | Maximum quality |
'''
    
    def _generate_footer(self) -> str:
        """Generate footer section with QuantLLM promotion."""
        return f'''
---

## 🚀 Created with QuantLLM

<div align="center">

[![QuantLLM](https://img.shields.io/badge/🚀_QuantLLM-Ultra--fast_LLM_Quantization-orange?style=for-the-badge)](https://github.com/codewithdark-git/QuantLLM)

**Convert any model to GGUF, ONNX, or MLX in one line!**

```python
from quantllm import turbo

# Load any HuggingFace model
model = turbo("{self.base_model}")

# Export to any format
model.export("{self.format}", quantization="{self.quantization or 'Q4_K_M'}")

# Push to HuggingFace
model.push("your-repo", format="{self.format}")
```

<a href="https://github.com/codewithdark-git/QuantLLM">
  <img src="https://img.shields.io/github/stars/codewithdark-git/QuantLLM?style=social" alt="GitHub Stars">
</a>

**[📚 Documentation](https://github.com/codewithdark-git/QuantLLM#readme)** · 
**[🐛 Report Issue](https://github.com/codewithdark-git/QuantLLM/issues)** · 
**[💡 Request Feature](https://github.com/codewithdark-git/QuantLLM/issues)**

</div>
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
