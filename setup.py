"""
QuantLLM - Ultra-Fast LLM Quantization and Deployment

Installation: pip install git+https://github.com/codewithdark-git/QuantLLM.git
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantllm",
    version="2.0.0",
    author="Dark Coder",
    author_email="codewithdark90@gmail.com",
    description="Ultra-fast LLM quantization, fine-tuning, and deployment with one line of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/codewithdark-git/QuantLLM",
    project_urls={
        "Homepage": "https://github.com/codewithdark-git/QuantLLM",
        "Documentation": "https://quantllm.readthedocs.io/",
        "Bug Tracker": "https://github.com/codewithdark-git/QuantLLM/issues",
        "Sponsor": "https://github.com/sponsors/codewithdark-git",
    },
    packages=find_packages(exclude=["test", "test.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "llm", "quantization", "gguf", "transformers", "pytorch",
        "fine-tuning", "lora", "inference", "llama", "mistral",
        "machine-learning", "deep-learning", "nlp"
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "bitsandbytes>=0.41.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "einops>=0.7.0",
        "psutil>=5.9.0",
        "huggingface-hub>=0.19.0",
        "rich>=13.0.0",  # For beautiful progress bars
        "gguf>=0.10.0",  # For GGUF loading/export
    ],
    extras_require={
        "gguf": [
            "gguf>=0.10.0",
        ],
        "onnx": [
            "onnx>=1.14.0",
            "onnxruntime>=1.16.0",
            "optimum[onnxruntime]>=1.14.0",
            "onnxscript>=0.1.0",
        ],
        "mlx": [
            "mlx>=0.1.0",
            "mlx-lm>=0.1.0",
        ],
        "triton": [
            "triton>=2.1.0",
        ],
        "flash": [
            "flash-attn>=2.3.0",
        ],
        "hub": [
            "hf-lifecycle @ git+https://github.com/codewithdark-git/huggingface-lifecycle.git",
        ],
        "full": [
            "gguf>=0.10.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.16.0",
            "optimum[onnxruntime]>=1.14.0",
            "onnxscript>=0.1.0",
            "triton>=2.1.0",
            "wandb>=0.15.0",
            "tensorboard>=2.14.0",
            "evaluate>=0.4.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-copybutton>=0.5.2",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantllm=quantllm.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)