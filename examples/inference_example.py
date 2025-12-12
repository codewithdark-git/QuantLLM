"""Example demonstrating inference with quantized models."""

import time

import torch
from transformers import AutoTokenizer

from quantllm import QuantLLM
from quantllm.utils.logger import logger


def run_inference(model, tokenizer, text: str, device: str = "cuda"):
    """Run inference on the given text."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        start_time = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        end_time = time.perf_counter()
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference_time = (end_time - start_time) * 1000  # Convert to ms
    
    return generated_text, inference_time

def main():
    # Load quantized model
    try:
        logger.log_info("Loading quantized model...")
        model_path = "quantized_model_highlevel"  # Path from quantization_example.py
        
        # Load model and tokenizer
        quantized_model = QuantLLM.load_quantized_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        quantized_model = quantized_model.to(device)
        quantized_model.eval()
        
        # Example prompts for testing
        prompts = [
            "Once upon a time in a distant galaxy,",
            "The secret to successful machine learning is",
            "In the year 2050, artificial intelligence has"
        ]
        
        logger.log_info("\nRunning inference tests...")
        
        # Run inference on each prompt
        for i, prompt in enumerate(prompts, 1):
            logger.log_info(f"\nTest {i}: Processing prompt: {prompt}")
            
            generated_text, inference_time = run_inference(
                quantized_model,
                tokenizer,
                prompt,
                device
            )
            
            logger.log_info(f"Generated text: {generated_text}")
            logger.log_info(f"Inference time: {inference_time:.2f}ms")
        
        # Load GGUF model for comparison
        try:
            from ctransformers import AutoModelForCausalLM as CTAutoModel
            
            logger.log_info("\nTesting GGUF model inference...")
            gguf_model = CTAutoModel.from_pretrained(
                "quantized_model.gguf",  # Path from quantization_example.py
                model_type="llama"
            )
            
            for i, prompt in enumerate(prompts, 1):
                logger.log_info(f"\nGGUF Test {i}: Processing prompt: {prompt}")
                
                start_time = time.perf_counter()
                generated_text = gguf_model(prompt, max_new_tokens=100)
                inference_time = (time.perf_counter() - start_time) * 1000
                
                logger.log_info(f"Generated text: {generated_text}")
                logger.log_info(f"Inference time: {inference_time:.2f}ms")
                
        except ImportError:
            logger.log_warning("ctransformers not installed. Skipping GGUF model inference.")
        
    except Exception as e:
        logger.log_error(f"Inference failed: {str(e)}")

if __name__ == "__main__":
    main() 