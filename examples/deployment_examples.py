#!/usr/bin/env python3
"""
QuantLLM Deployment Examples

This script demonstrates how to quantize models for different deployment
platforms and inference engines.
"""

import os
import json
import time
import torch
from pathlib import Path
from quantllm import QuantLLM
from quantllm.deployment import DeploymentOptimizer
from quantllm.conversion import convert_for_platform

def setup_logging():
    """Set up logging for the examples."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def quantize_for_llama_cpp():
    """
    Example: Quantizing for llama.cpp deployment
    
    llama.cpp is optimized for CPU inference and supports GGUF format.
    """
    print("\n" + "="*60)
    print("LLAMA.CPP DEPLOYMENT EXAMPLE")
    print("="*60)
    
    model_name = "gpt2"
    output_dir = "./deployment_models/llama_cpp"
    
    print(f"Quantizing {model_name} for llama.cpp deployment...")
    
    # Optimize for llama.cpp
    result = QuantLLM.optimize_for_deployment(
        model=model_name,
        target_platform="llama_cpp",
        constraints={
            "target_device": "cpu",
            "memory_limit": "8GB",
            "optimize_for": "inference_speed"
        },
        output_dir=output_dir
    )
    
    print(f"Model quantized and optimized for llama.cpp")
    print(f"Output format: {result.format}")
    print(f"Quantization type: {result.quant_type}")
    print(f"File size: {result.file_size_mb:.1f} MB")
    
    # Generate llama.cpp configuration
    llama_config = {
        "model_path": f"{output_dir}/model.gguf",
        "n_ctx": 2048,
        "n_threads": os.cpu_count(),
        "n_gpu_layers": 0,  # CPU-only
        "use_mmap": True,
        "use_mlock": False,
        "verbose": False
    }
    
    config_path = f"{output_dir}/llama_config.json"
    with open(config_path, 'w') as f:
        json.dump(llama_config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    
    # Generate usage example
    usage_script = f"""#!/bin/bash
# llama.cpp usage example
# Make sure llama.cpp is installed and compiled

# Interactive chat
./llama-cpp-chat -m {output_dir}/model.gguf -c 2048 -t {os.cpu_count()}

# Server mode
./llama-cpp-server -m {output_dir}/model.gguf -c 2048 -t {os.cpu_count()} --port 8080

# Single completion
echo "The future of AI is" | ./llama-cpp-main -m {output_dir}/model.gguf -p -
"""
    
    with open(f"{output_dir}/usage_example.sh", 'w') as f:
        f.write(usage_script)
    
    print(f"Usage examples saved to: {output_dir}/usage_example.sh")
    
    return result

def quantize_for_vllm():
    """
    Example: Quantizing for vLLM deployment
    
    vLLM is optimized for high-throughput GPU inference.
    """
    print("\n" + "="*60)
    print("vLLM DEPLOYMENT EXAMPLE")
    print("="*60)
    
    model_name = "gpt2"
    output_dir = "./deployment_models/vllm"
    
    print(f"Quantizing {model_name} for vLLM deployment...")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("Warning: vLLM requires CUDA. This example will prepare the model but may not run.")
    
    # Optimize for vLLM
    result = QuantLLM.optimize_for_deployment(
        model=model_name,
        target_platform="vllm",
        constraints={
            "target_device": "cuda",
            "batch_size": 32,
            "max_sequence_length": 2048,
            "optimize_for": "throughput"
        },
        output_dir=output_dir
    )
    
    print(f"Model quantized and optimized for vLLM")
    print(f"Output format: {result.format}")
    print(f"Quantization method: {result.method}")
    print(f"Model size: {result.file_size_mb:.1f} MB")
    
    # Generate vLLM configuration
    vllm_config = {
        "model": output_dir,
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.9,
        "quantization": result.method if result.method != "none" else None
    }
    
    config_path = f"{output_dir}/vllm_config.json"
    with open(config_path, 'w') as f:
        json.dump(vllm_config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    
    # Generate usage example
    usage_script = f'''#!/usr/bin/env python3
"""
vLLM usage example
Make sure vLLM is installed: pip install vllm
"""

from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(
    model="{output_dir}",
    tensor_parallel_size=1,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.9
)

# Set up sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

# Generate text
prompts = [
    "The future of artificial intelligence is",
    "In a world where technology advances rapidly,",
    "The most important aspect of machine learning is"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {{prompt}}")
    print(f"Generated: {{generated_text}}")
    print("-" * 50)

# Server mode example
# python -m vllm.entrypoints.api_server --model {output_dir} --port 8000
'''
    
    with open(f"{output_dir}/usage_example.py", 'w') as f:
        f.write(usage_script)
    
    print(f"Usage examples saved to: {output_dir}/usage_example.py")
    
    return result

def quantize_for_tensorrt():
    """
    Example: Quantizing for TensorRT-LLM deployment
    
    TensorRT-LLM provides optimized inference for NVIDIA GPUs.
    """
    print("\n" + "="*60)
    print("TENSORRT-LLM DEPLOYMENT EXAMPLE")
    print("="*60)
    
    model_name = "gpt2"
    output_dir = "./deployment_models/tensorrt"
    
    print(f"Quantizing {model_name} for TensorRT-LLM deployment...")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("Warning: TensorRT-LLM requires CUDA. This example will prepare the model but may not run.")
    
    # Optimize for TensorRT-LLM
    result = QuantLLM.optimize_for_deployment(
        model=model_name,
        target_platform="tensorrt_llm",
        constraints={
            "target_device": "cuda",
            "precision": "fp16",
            "max_batch_size": 8,
            "max_sequence_length": 2048,
            "optimize_for": "latency"
        },
        output_dir=output_dir
    )
    
    print(f"Model quantized and optimized for TensorRT-LLM")
    print(f"Output format: {result.format}")
    print(f"Precision: {result.precision}")
    print(f"Model size: {result.file_size_mb:.1f} MB")
    
    # Generate TensorRT-LLM build configuration
    trt_config = {
        "architecture": "GPTForCausalLM",
        "dtype": "float16",
        "logits_dtype": "float32",
        "vocab_size": 50257,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "n_positions": 1024,
        "max_batch_size": 8,
        "max_input_len": 1024,
        "max_output_len": 1024,
        "max_beam_width": 1,
        "plugin_config": {
            "gpt_attention_plugin": "float16",
            "gemm_plugin": "float16",
            "lookup_plugin": "float16"
        }
    }
    
    config_path = f"{output_dir}/trt_config.json"
    with open(config_path, 'w') as f:
        json.dump(trt_config, f, indent=2)
    
    print(f"TensorRT configuration saved to: {config_path}")
    
    # Generate build script
    build_script = f'''#!/bin/bash
# TensorRT-LLM build script
# Make sure TensorRT-LLM is installed

# Convert to TensorRT-LLM format
python convert_checkpoint.py \\
    --model_dir {output_dir} \\
    --output_dir {output_dir}/trt_checkpoint \\
    --dtype float16

# Build TensorRT engine
trtllm-build \\
    --checkpoint_dir {output_dir}/trt_checkpoint \\
    --output_dir {output_dir}/trt_engines \\
    --gemm_plugin float16 \\
    --gpt_attention_plugin float16 \\
    --max_batch_size 8 \\
    --max_input_len 1024 \\
    --max_output_len 1024

echo "TensorRT engine built successfully!"
'''
    
    with open(f"{output_dir}/build.sh", 'w') as f:
        f.write(build_script)
    
    print(f"Build script saved to: {output_dir}/build.sh")
    
    return result

def quantize_for_transformers():
    """
    Example: Quantizing for HuggingFace Transformers deployment
    
    Standard deployment using HuggingFace Transformers library.
    """
    print("\n" + "="*60)
    print("HUGGINGFACE TRANSFORMERS DEPLOYMENT EXAMPLE")
    print("="*60)
    
    model_name = "gpt2"
    output_dir = "./deployment_models/transformers"
    
    print(f"Quantizing {model_name} for Transformers deployment...")
    
    # Optimize for Transformers
    result = QuantLLM.optimize_for_deployment(
        model=model_name,
        target_platform="transformers",
        constraints={
            "target_device": "auto",  # Auto-detect best device
            "memory_limit": "8GB",
            "optimize_for": "balanced"
        },
        output_dir=output_dir
    )
    
    print(f"Model quantized and optimized for Transformers")
    print(f"Output format: {result.format}")
    print(f"Quantization method: {result.method}")
    print(f"Model size: {result.file_size_mb:.1f} MB")
    
    # Generate usage example
    usage_script = f'''#!/usr/bin/env python3
"""
HuggingFace Transformers usage example
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained(
    "{output_dir}",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=False
)

tokenizer = AutoTokenizer.from_pretrained("{output_dir}")

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate text
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompts = [
    "The future of artificial intelligence is",
    "In a world where technology advances rapidly,",
    "The most important aspect of machine learning is"
]

for prompt in prompts:
    generated = generate_text(prompt)
    print(f"Prompt: {{prompt}}")
    print(f"Generated: {{generated}}")
    print("-" * 50)

# Model information
print(f"Model device: {{next(model.parameters()).device}}")
print(f"Model dtype: {{next(model.parameters()).dtype}}")
print(f"Model memory usage: {{model.get_memory_footprint() / 1024**2:.1f}} MB")
'''
    
    with open(f"{output_dir}/usage_example.py", 'w') as f:
        f.write(usage_script)
    
    print(f"Usage examples saved to: {output_dir}/usage_example.py")
    
    return result

def quantize_for_mobile_deployment():
    """
    Example: Quantizing for mobile deployment
    
    Mobile deployment requires aggressive optimization for size and efficiency.
    """
    print("\n" + "="*60)
    print("MOBILE DEPLOYMENT EXAMPLE")
    print("="*60)
    
    model_name = "gpt2"  # Using smaller model for mobile
    output_dir = "./deployment_models/mobile"
    
    print(f"Quantizing {model_name} for mobile deployment...")
    
    # Optimize for mobile
    result = QuantLLM.optimize_for_deployment(
        model=model_name,
        target_platform="mobile",
        constraints={
            "target_device": "cpu",
            "memory_limit": "1GB",
            "model_size_limit": "100MB",
            "optimize_for": "size"
        },
        output_dir=output_dir
    )
    
    print(f"Model quantized and optimized for mobile")
    print(f"Output format: {result.format}")
    print(f"Quantization: {result.bits}-bit")
    print(f"Model size: {result.file_size_mb:.1f} MB")
    print(f"Compression ratio: {result.compression_ratio:.2f}x")
    
    # Generate mobile configuration
    mobile_config = {
        "model_path": f"{output_dir}/model.gguf",
        "max_context_length": 512,  # Reduced for mobile
        "num_threads": 2,  # Conservative for mobile CPUs
        "use_mmap": True,
        "use_mlock": False,
        "memory_pool_size": "256MB"
    }
    
    config_path = f"{output_dir}/mobile_config.json"
    with open(config_path, 'w') as f:
        json.dump(mobile_config, f, indent=2)
    
    print(f"Mobile configuration saved to: {config_path}")
    
    # Generate deployment guide
    deployment_guide = f'''# Mobile Deployment Guide

## Model Information
- Model: {model_name}
- Size: {result.file_size_mb:.1f} MB
- Format: {result.format}
- Quantization: {result.bits}-bit

## Integration Steps

### Android (using llama.cpp Android)
1. Copy model.gguf to Android assets folder
2. Use llama.cpp Android bindings
3. Configure with mobile_config.json settings

### iOS (using llama.cpp iOS)
1. Add model.gguf to iOS bundle
2. Use llama.cpp iOS bindings
3. Configure memory limits appropriately

### React Native
1. Use react-native-llama package
2. Configure model path and settings
3. Handle memory management carefully

## Performance Considerations
- Limit context length to 512 tokens
- Use 2 threads maximum on mobile CPUs
- Monitor memory usage closely
- Consider model caching strategies

## Testing
- Test on actual devices, not simulators
- Monitor battery usage during inference
- Verify performance on low-end devices
- Test memory pressure scenarios
'''
    
    with open(f"{output_dir}/deployment_guide.md", 'w') as f:
        f.write(deployment_guide)
    
    print(f"Deployment guide saved to: {output_dir}/deployment_guide.md")
    
    return result

def create_docker_deployment():
    """
    Example: Creating Docker containers for deployment
    
    Containerized deployment for easy scaling and distribution.
    """
    print("\n" + "="*60)
    print("DOCKER DEPLOYMENT EXAMPLE")
    print("="*60)
    
    model_name = "gpt2"
    output_dir = "./deployment_models/docker"
    
    print(f"Preparing {model_name} for Docker deployment...")
    
    # Quantize model
    result = QuantLLM.quantize(
        model=model_name,
        method="gguf",
        bits=4,
        optimization_target="balanced",
        output_dir=f"{output_dir}/model"
    )
    
    # Create Dockerfile
    dockerfile_content = f'''FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    cmake \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    torch \\
    transformers \\
    quantllm \\
    fastapi \\
    uvicorn

# Create app directory
WORKDIR /app

# Copy model and configuration
COPY model/ /app/model/
COPY server.py /app/
COPY requirements.txt /app/

# Install additional requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    with open(f"{output_dir}/Dockerfile", 'w') as f:
        f.write(dockerfile_content)
    
    # Create FastAPI server
    server_content = f'''#!/usr/bin/env python3
"""
FastAPI server for quantized model inference
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI(title="QuantLLM Inference Server", version="1.0.0")

# Global model and tokenizer
model = None
tokenizer = None

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1

class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    
    model_path = "/app/model"
    
    print(f"Loading model from {{model_path}}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")

@app.get("/health")
async def health_check():
    return {{"status": "healthy", "model_loaded": model is not None}}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                num_return_sequences=request.num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {{
        "model_type": type(model).__name__,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "memory_footprint_mb": model.get_memory_footprint() / 1024**2
    }}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open(f"{output_dir}/server.py", 'w') as f:
        f.write(server_content)
    
    # Create requirements.txt
    requirements = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
torch>=2.0.0
transformers>=4.35.0
quantllm
pydantic>=2.0.0
'''
    
    with open(f"{output_dir}/requirements.txt", 'w') as f:
        f.write(requirements)
    
    # Create docker-compose.yml
    compose_content = f'''version: '3.8'

services:
  quantllm-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - quantllm-server
    restart: unless-stopped
'''
    
    with open(f"{output_dir}/docker-compose.yml", 'w') as f:
        f.write(compose_content)
    
    # Create nginx configuration
    nginx_config = '''events {
    worker_connections 1024;
}

http {
    upstream quantllm {
        server quantllm-server:8000;
    }

    server {
        listen 80;
        
        location / {
            proxy_pass http://quantllm;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
'''
    
    with open(f"{output_dir}/nginx.conf", 'w') as f:
        f.write(nginx_config)
    
    # Create deployment script
    deploy_script = '''#!/bin/bash
# Docker deployment script

echo "Building QuantLLM Docker image..."
docker build -t quantllm-server .

echo "Starting services..."
docker-compose up -d

echo "Waiting for services to start..."
sleep 30

echo "Testing health endpoint..."
curl -f http://localhost:8000/health

echo "Testing generation endpoint..."
curl -X POST "http://localhost:8000/generate" \\
     -H "Content-Type: application/json" \\
     -d '{"prompt": "The future of AI is", "max_length": 50}'

echo "Deployment complete!"
echo "Server available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
'''
    
    with open(f"{output_dir}/deploy.sh", 'w') as f:
        f.write(deploy_script)
    
    os.chmod(f"{output_dir}/deploy.sh", 0o755)
    
    print(f"Docker deployment files created in: {output_dir}")
    print("Files created:")
    print("- Dockerfile")
    print("- server.py")
    print("- requirements.txt")
    print("- docker-compose.yml")
    print("- nginx.conf")
    print("- deploy.sh")
    
    return result

def main():
    """Run all deployment examples."""
    logger = setup_logging()
    
    print("QuantLLM Deployment Examples")
    print("=" * 60)
    print("This script demonstrates quantization for different deployment platforms")
    
    # Create output directories
    os.makedirs("./deployment_models", exist_ok=True)
    
    try:
        # Run deployment examples
        logger.info("Starting llama.cpp deployment example...")
        llama_result = quantize_for_llama_cpp()
        
        logger.info("Starting vLLM deployment example...")
        vllm_result = quantize_for_vllm()
        
        logger.info("Starting TensorRT-LLM deployment example...")
        tensorrt_result = quantize_for_tensorrt()
        
        logger.info("Starting Transformers deployment example...")
        transformers_result = quantize_for_transformers()
        
        logger.info("Starting mobile deployment example...")
        mobile_result = quantize_for_mobile_deployment()
        
        logger.info("Creating Docker deployment...")
        docker_result = create_docker_deployment()
        
        print("\n" + "="*60)
        print("ALL DEPLOYMENT EXAMPLES COMPLETED!")
        print("="*60)
        print("\nGenerated deployment configurations:")
        print("- ./deployment_models/llama_cpp/ - CPU inference with llama.cpp")
        print("- ./deployment_models/vllm/ - High-throughput GPU inference")
        print("- ./deployment_models/tensorrt/ - Optimized NVIDIA GPU inference")
        print("- ./deployment_models/transformers/ - Standard HuggingFace deployment")
        print("- ./deployment_models/mobile/ - Mobile-optimized deployment")
        print("- ./deployment_models/docker/ - Containerized deployment")
        
        print("\nDeployment recommendations:")
        print("1. Use llama.cpp for CPU-only environments")
        print("2. Use vLLM for high-throughput GPU serving")
        print("3. Use TensorRT-LLM for lowest latency on NVIDIA GPUs")
        print("4. Use Transformers for standard PyTorch deployment")
        print("5. Use mobile configs for edge deployment")
        print("6. Use Docker for scalable cloud deployment")
        
    except Exception as e:
        logger.error(f"Deployment example failed: {e}")
        raise

if __name__ == "__main__":
    main()