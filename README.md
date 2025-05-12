# 🧠 QuantLLM: Lightweight Library for Quantized LLM Fine-Tuning and Deployment

[![PyPI Downloads](https://static.pepy.tech/badge/quantllm)](https://pepy.tech/projects/quantllm)
<img alt="PyPI - Version" src="https://img.shields.io/pypi/v/quantllm?logo=pypi&label=version&">


## 📌 Overview

**QuantLLM** is a Python library designed for developers, researchers, and teams who want to fine-tune and deploy large language models (LLMs) **efficiently** using **4-bit and 8-bit quantization** techniques. It provides a modular and flexible framework for:

- **Loading and quantizing models** with advanced configurations
- **LoRA / QLoRA-based fine-tuning** with customizable parameters
- **Dataset management** with preprocessing and splitting
- **Training and evaluation** with comprehensive metrics
- **Model checkpointing** and versioning
- **Hugging Face Hub integration** for model sharing

The goal of QuantLLM is to **democratize LLM training**, especially in low-resource environments, while keeping the workflow intuitive, modular, and production-ready.

## 🎯 Key Features

| Feature                          | Description |
|----------------------------------|-------------|
| ✅ Quantized Model Loading       | Load any HuggingFace model in 4-bit or 8-bit precision with customizable quantization settings |
| ✅ Advanced Dataset Management   | Load, preprocess, and split datasets with flexible configurations |
| ✅ LoRA / QLoRA Fine-Tuning      | Memory-efficient fine-tuning with customizable LoRA parameters |
| ✅ Comprehensive Training        | Advanced training loop with mixed precision, gradient accumulation, and early stopping |
| ✅ Model Evaluation             | Flexible evaluation with custom metrics and batch processing |
| ✅ Checkpoint Management        | Save, resume, and manage training checkpoints with versioning |
| ✅ Hub Integration              | Push models and checkpoints to Hugging Face Hub with authentication |
| ✅ Configuration Management     | YAML/JSON config support for reproducible experiments |
| ✅ Logging and Monitoring       | Comprehensive logging and Weights & Biases integration |

## 🚀 Getting Started

### Installation

```bash
pip install quantllm
```

For detailed usage examples and API documentation, please refer to our:
- 📚 [Official Documentation](https://quantllm.readthedocs.io/)
- 🎓 [Tutorials](https://quantllm.readthedocs.io/tutorials/)
- 📖 [API Reference](https://quantllm.readthedocs.io/api/)

## 💻 Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Storage**: 20GB free space
- **Python**: 3.8+

### Recommended Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 32GB
- **Storage**: 50GB+ SSD
- **CUDA**: 11.7+

### Resource Usage Guidelines
| Model Size | 4-bit (GPU RAM) | 8-bit (GPU RAM) | CPU RAM (min) |
|------------|----------------|-----------------|---------------|
| 3B params  | ~6GB          | ~9GB           | 16GB         |
| 7B params  | ~12GB         | ~18GB          | 32GB         |
| 13B params | ~20GB         | ~32GB          | 64GB         |
| 70B params | ~90GB         | ~140GB         | 256GB        |

## 🔄 Version Compatibility

| QuantLLM | Python | PyTorch | Transformers | CUDA  |
|----------|--------|----------|--------------|-------|
| latest    | ≥3.10   | ≥2.0.0   | ≥4.30.0     | ≥11.7 |

## 🗺 Roadmap

- [ ] Multi-GPU training support
- [ ] AutoML for hyperparameter tuning
- [ ] More quantization methods
- [ ] Custom model architecture support
- [ ] Enhanced logging and visualization
- [ ] Model compression techniques
- [ ] Deployment optimizations

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTE.md](CONTRIBUTE.md) for guidelines and setup instructions.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for their amazing Transformers library
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization
- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning
- [Weights & Biases](https://wandb.ai/) for experiment tracking

## 📫 Contact & Support

- GitHub Issues: [Create an issue](https://github.com/codewithdark-git/QuantLLM/issues)
- Documentation: [Read the docs](https://quantllm.readthedocs.io/)
- Discord: [Join our community](https://discord.gg/quantllm)
- Email: 
