# 🧠 QuantLLM: Efficient GGUF Model Quantization and Deployment

[![PyPI Downloads](https://static.pepy.tech/badge/quantllm)](https://pepy.tech/projects/quantllm)
<img alt="PyPI - Version" src="https://img.shields.io/pypi/v/quantllm?logo=pypi&label=version&">

## 📌 Overview

**QuantLLM** is a Python library designed for efficient model quantization using the GGUF (GGML Universal Format) method. It provides a robust framework for converting and deploying large language models with minimal memory footprint and optimal performance. Key capabilities include:

- **Memory-efficient GGUF quantization** with multiple precision options (2-bit to 8-bit)
- **Chunk-based processing** for handling large models
- **Comprehensive benchmarking** tools
- **Detailed progress tracking** with memory statistics
- **Easy model export** and deployment

## 🎯 Key Features

| Feature                          | Description |
|----------------------------------|-------------|
| ✅ Multiple GGUF Types          | Support for various GGUF quantization types (Q2_K to Q8_0) with different precision-size tradeoffs |
| ✅ Memory Optimization          | Chunk-based processing and CPU offloading for efficient handling of large models |
| ✅ Progress Tracking            | Detailed layer-wise progress with memory statistics and ETA |
| ✅ Benchmarking Tools           | Comprehensive benchmarking suite for performance evaluation |
| ✅ Hardware Optimization        | Automatic device selection and memory management |
| ✅ Easy Deployment              | Simple conversion to GGUF format for deployment |
| ✅ Flexible Configuration       | Customizable quantization parameters and processing options |

## 🚀 Getting Started

### Installation

Basic installation:
```bash
pip install quantllm
```

With GGUF support (recommended):
```bash
pip install quantllm[gguf]
```

### Quick Example

```python
from quantllm import QuantLLM
from transformers import AutoTokenizer

# Load tokenizer and prepare data
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
calibration_text = ["Example text for calibration."] * 10
calibration_data = tokenizer(calibration_text, return_tensors="pt", padding=True)["input_ids"]

# Quantize model
quantized_model, benchmark_results = QuantLLM.quantize_from_pretrained(
    model_name_or_path=model_name,
    bits=4,                    # Quantization bits (2-8)
    group_size=32,            # Group size for quantization
    quant_type="Q4_K_M",      # GGUF quantization type
    calibration_data=calibration_data,
    benchmark=True,           # Run benchmarks
    benchmark_input_shape=(1, 32)
)

# Save and convert to GGUF
QuantLLM.save_quantized_model(model=quantized_model, output_path="quantized_model")
QuantLLM.convert_to_gguf(model=quantized_model, output_path="model.gguf")
```

For detailed usage examples and API documentation, please refer to our:
- 📚 [Official Documentation](https://quantllm.readthedocs.io/)
- 🎓 [Tutorials](https://quantllm.readthedocs.io/tutorials/)
- 📖 [API Reference](https://quantllm.readthedocs.io/api/)

## 💻 Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 16GB+
- **Storage**: 10GB+ free space
- **Python**: 3.10+

### Recommended for Large Models
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.7+
- **Storage**: 20GB+ free space

### GGUF Quantization Types

| Type    | Bits | Description           | Use Case                    |
|---------|------|-----------------------|-----------------------------|
| Q2_K    | 2    | Extreme compression   | Size-critical deployment   |
| Q3_K_S  | 3    | Small size           | Limited storage            |
| Q4_K_M  | 4    | Balanced quality     | General use                |
| Q5_K_M  | 5    | Higher quality       | Quality-sensitive tasks    |
| Q8_0    | 8    | Best quality         | Accuracy-critical tasks    |

## 🔄 Version Compatibility

| QuantLLM | Python | PyTorch | Transformers | CUDA  |
|----------|--------|----------|--------------|-------|
| 1.2.0    | ≥3.10  | ≥2.0.0   | ≥4.30.0     | ≥11.7 |

## 🗺 Roadmap

- [ ] Support for more GGUF model architectures
- [ ] Enhanced benchmarking capabilities
- [ ] Multi-GPU processing support
- [ ] Advanced memory optimization techniques
- [ ] Integration with more deployment platforms
- [ ] Custom quantization kernels

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTE.md](CONTRIBUTE.md) for guidelines and setup instructions.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF format
- [HuggingFace](https://huggingface.co/) for Transformers library
- [CTransformers](https://github.com/marella/ctransformers) for GGUF support

## 📫 Contact & Support

- GitHub Issues: [Create an issue](https://github.com/yourusername/QuantLLM/issues)
- Documentation: [Read the docs](https://quantllm.readthedocs.io/)
- Discord: [Join our community](https://discord.gg/quantllm)
- Email: support@quantllm.ai
