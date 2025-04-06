from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantllm",
    version="0.0.1b",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.39.0",
        "huggingface_hub>=0.15.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
    ],
    author="Dark Coder",
    author_email="codewithdark90@gmail.com",
    description="Lightweight Library for Quantized LLM Fine-Tuning and Deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Homepage": "https://github.com/codewithdark-git/QuantLLM",
        "Sponsor": "https://github.com/sponsors/codewithdark-git",  # 💰
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
) 