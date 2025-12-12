"""Example demonstrating model quantization using QuantLLM."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantllm import QuantLLM
from quantllm.quant import GGUFQuantizer
from quantllm.utils.logger import logger


def main():
    # Example 1: Using High-Level API
    logger.log_info("Example 1: Using High-Level API")
    
    # Load model and tokenizer
    model_name = "facebook/opt-125m"  # Small model for demonstration
    logger.log_info(f"Loading model: {model_name}")
    
    # Generate some dummy calibration data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    calibration_text = ["This is an example text for calibration."] * 10
    calibration_data = tokenizer(calibration_text, return_tensors="pt", padding=True)["input_ids"]
    
    try:
        # Quantize model using high-level API
        quantized_model, benchmark_results = QuantLLM.quantize_from_pretrained(
            model_name_or_path=model_name,
            bits=4,
            group_size=32,
            quant_type="Q4_K_M",
            calibration_data=calibration_data,
            benchmark=True,
            benchmark_input_shape=(1, 32),
            benchmark_steps=50
        )
        
        # Save the quantized model
        output_dir = "quantized_model_highlevel"
        QuantLLM.save_quantized_model(
            model=quantized_model,
            output_path=output_dir,
            save_tokenizer=True
        )
        logger.log_success(f"High-level API: Model saved to {output_dir}")
        
    except Exception as e:
        logger.log_error(f"High-level API quantization failed: {str(e)}")
    
    # Example 2: Using Direct Quantization
    logger.log_info("\nExample 2: Using Direct Quantization")
    
    try:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Initialize quantizer
        quantizer = GGUFQuantizer(
            model_name=model,
            bits=4,
            group_size=32,
            quant_type="Q4_K_M",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Quantize model
        quantized_model = quantizer.quantize(calibration_data=calibration_data)
        
        # Save as GGUF format
        gguf_path = "quantized_model.gguf"
        quantizer.convert_to_gguf(gguf_path)
        logger.log_success(f"Direct quantization: Model saved to {gguf_path}")
        
    except Exception as e:
        logger.log_error(f"Direct quantization failed: {str(e)}")

if __name__ == "__main__":
    main() 