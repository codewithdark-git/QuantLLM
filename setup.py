from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantllm",
    version="1.2.0",
    author="Dark Coder",
    author_email="codewithdark90@gmail.com",
    description="A lightweight library for quantized LLM fine-tuning and deployment with GGUF support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Homepage": "https://github.com/codewithdark-git/DiffusionLM",
        "Sponsor": "https://github.com/sponsors/codewithdark-git",  # 💰
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "datasets>=2.12.0",
            "accelerate>=0.20.0",            
            "peft>=0.4.0",
            "bitsandbytes>=0.40.0",
            "scipy>=1.10.0",
            "scikit-learn>=1.0.0",
            "tqdm>=4.65.0",
            "numpy>=1.24.0",
            "wandb>=0.15.0",
            "sentencepiece>=0.1.99",
            "protobuf>=3.20.0",
            "einops>=0.6.1",
            "evaluate>=0.4.0",
            "tensorboard>=2.13.0",
            "psutil>=5.9.0",
            "pandas>=1.5.0",
    ],
    extras_require={
        "dev": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-copybutton>=0.5.0",
            "sphinx-autodoc-typehints>=1.18.3",
            "myst-parser>=0.18.1",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "gguf": [
            "ctransformers>=0.2.24",
            "llama-cpp-python>=0.2.11",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)