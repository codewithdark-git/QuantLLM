#!/usr/bin/env python3
"""
QuantLLM Examples for Different Model Sizes

This script demonstrates how to quantize models of different sizes with
appropriate configurations and optimizations.
"""

import os
import time
import torch
from quantllm import QuantLLM
from quantllm.utils import get_model_info, estimate_memory_usage
from datasets import load_dataset

def setup_logging():
    """Set up logging for the examples."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_calibration_data(num_samples=512):
    """Get calibration data for quantization."""
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        return dataset.select(range(min(num_samples, len(dataset))))
    except Exception as e:
        print(f"Warning: Could not load calibration data: {e}")
        return None

def quantize_small_model():
    """
    Example: Quantizing small models (< 1B parameters)
    
    Small models like GPT-2 can be quantized quickly with standard settings.
    """
    print("\n" + "="*60)
    print("SMALL MODEL EXAMPLE: GPT-2 (117M parameters)")
    print("="*60)
    
    model_name = "gpt2"
    
    # Get model information
    info = get_model_info(model_name)
    print(f"Model: {info['name']}")
    print(f"Parameters: {info['parameters']:,}")
    print(f"Model size: {info['size_mb']:.1f} MB")
    
    # Estimate memory usage
    memory_est = estimate_memory_usage(model_name, method="gptq", bits=4)
    print(f"Estimated memory usage: {memory_est['total_mb']:.1f} MB")
    
    # Get calibration data
    calibration_data = get_calibration_data(256)  # Fewer samples for small models
    
    # Quantize with standard settings
    start_time = time.time()
    
    result = QuantLLM.quantize(
        model=model_name,
        method="gptq",
        bits=4,
        group_size=128,
        calibration_data=calibration_data,
        optimization_target="balanced",
        output_dir="./quantized_models/gpt2_4bit"
    )
    
    end_time = time.time()
    
    print(f"\nQuantization completed in {end_time - start_time:.1f} seconds")
    print(f"Original size: {result.original_size_mb:.1f} MB")
    print(f"Quantized size: {result.quantized_size_mb:.1f} MB")
    print(f"Compression ratio: {result.compression_ratio:.2f}x")
    print(f"Quality score: {result.quality_metrics.get('perplexity_ratio', 'N/A')}")
    
    return result

def quantize_medium_model():
    """
    Example: Quantizing medium models (1B-7B parameters)
    
    Medium models require more careful memory management and optimization.
    """
    print("\n" + "="*60)
    print("MEDIUM MODEL EXAMPLE: GPT-2 Large (774M parameters)")
    print("="*60)
    
    model_name = "gpt2-large"
    
    # Get model information
    info = get_model_info(model_name)
    print(f"Model: {info['name']}")
    print(f"Parameters: {info['parameters']:,}")
    print(f"Model size: {info['size_mb']:.1f} MB")
    
    # Check available memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Available GPU memory: {gpu_memory:.1f} GB")
    
    # Get calibration data
    calibration_data = get_calibration_data(512)
    
    # Quantize with memory optimization
    start_time = time.time()
    
    result = QuantLLM.quantize(
        model=model_name,
        method="gptq",
        bits=4,
        group_size=64,  # Smaller group size for better quality
        calibration_data=calibration_data,
        optimization_target="quality",
        hardware_config={
            "enable_cpu_offload": True,  # Enable if GPU memory is limited
            "mixed_precision": True,
            "gradient_checkpointing": True
        },
        output_dir="./quantized_models/gpt2_large_4bit"
    )
    
    end_time = time.time()
    
    print(f"\nQuantization completed in {end_time - start_time:.1f} seconds")
    print(f"Original size: {result.original_size_mb:.1f} MB")
    print(f"Quantized size: {result.quantized_size_mb:.1f} MB")
    print(f"Compression ratio: {result.compression_ratio:.2f}x")
    print(f"Quality score: {result.quality_metrics.get('perplexity_ratio', 'N/A')}")
    
    return result

def quantize_large_model():
    """
    Example: Quantizing large models (7B+ parameters)
    
    Large models require aggressive memory optimization and may need
    special handling depending on available hardware.
    """
    print("\n" + "="*60)
    print("LARGE MODEL EXAMPLE: Microsoft DialoGPT Large (774M parameters)")
    print("Note: This example uses DialoGPT as a proxy for larger models")
    print("="*60)
    
    model_name = "microsoft/DialoGPT-large"
    
    # Get model information
    info = get_model_info(model_name)
    print(f"Model: {info['name']}")
    print(f"Parameters: {info['parameters']:,}")
    print(f"Model size: {info['size_mb']:.1f} MB")
    
    # Check system resources
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Available GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 16:
            print("Warning: Large models may require 16GB+ GPU memory")
            print("Enabling aggressive memory optimization...")
    
    # Get calibration data
    calibration_data = get_calibration_data(1000)  # More samples for large models
    
    # Quantize with aggressive optimization
    start_time = time.time()
    
    # Choose method based on available memory
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 16 * 1024**3:
        method = "gptq"
        print("Using GPTQ method (sufficient GPU memory)")
    else:
        method = "gguf"
        print("Using GGUF method (memory-efficient)")
    
    result = QuantLLM.quantize(
        model=model_name,
        method=method,
        bits=4,
        group_size=32 if method == "gptq" else None,
        calibration_data=calibration_data,
        optimization_target="memory",
        hardware_config={
            "enable_cpu_offload": True,
            "streaming_quantization": True,
            "chunk_size": 1024,
            "mixed_precision": True,
            "gradient_checkpointing": True,
            "use_8bit_adam": True
        },
        output_dir=f"./quantized_models/diallogpt_large_{method}_4bit"
    )
    
    end_time = time.time()
    
    print(f"\nQuantization completed in {end_time - start_time:.1f} seconds")
    print(f"Original size: {result.original_size_mb:.1f} MB")
    print(f"Quantized size: {result.quantized_size_mb:.1f} MB")
    print(f"Compression ratio: {result.compression_ratio:.2f}x")
    print(f"Quality score: {result.quality_metrics.get('perplexity_ratio', 'N/A')}")
    
    return result

def quantize_with_different_bit_widths():
    """
    Example: Comparing different bit-widths on the same model
    
    This shows the trade-offs between compression and quality.
    """
    print("\n" + "="*60)
    print("BIT-WIDTH COMPARISON EXAMPLE")
    print("="*60)
    
    model_name = "gpt2"
    calibration_data = get_calibration_data(256)
    
    bit_widths = [8, 4, 3, 2]
    results = {}
    
    for bits in bit_widths:
        print(f"\nQuantizing with {bits}-bit precision...")
        
        start_time = time.time()
        
        try:
            result = QuantLLM.quantize(
                model=model_name,
                method="gptq",
                bits=bits,
                group_size=128,
                calibration_data=calibration_data,
                optimization_target="balanced",
                output_dir=f"./quantized_models/gpt2_{bits}bit"
            )
            
            end_time = time.time()
            
            results[bits] = {
                'time': end_time - start_time,
                'size_mb': result.quantized_size_mb,
                'compression_ratio': result.compression_ratio,
                'quality_score': result.quality_metrics.get('perplexity_ratio', 1.0)
            }
            
            print(f"  Time: {results[bits]['time']:.1f}s")
            print(f"  Size: {results[bits]['size_mb']:.1f} MB")
            print(f"  Compression: {results[bits]['compression_ratio']:.2f}x")
            print(f"  Quality: {results[bits]['quality_score']:.3f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[bits] = None
    
    # Summary comparison
    print(f"\n{'Bits':<6} {'Time(s)':<8} {'Size(MB)':<10} {'Compression':<12} {'Quality':<8}")
    print("-" * 50)
    
    for bits in bit_widths:
        if results[bits]:
            r = results[bits]
            print(f"{bits:<6} {r['time']:<8.1f} {r['size_mb']:<10.1f} "
                  f"{r['compression_ratio']:<12.2f} {r['quality_score']:<8.3f}")
        else:
            print(f"{bits:<6} {'Failed':<8} {'N/A':<10} {'N/A':<12} {'N/A':<8}")
    
    return results

def quantize_for_specific_hardware():
    """
    Example: Hardware-specific optimizations
    
    This shows how to optimize quantization for different hardware configurations.
    """
    print("\n" + "="*60)
    print("HARDWARE-SPECIFIC OPTIMIZATION EXAMPLE")
    print("="*60)
    
    model_name = "gpt2"
    calibration_data = get_calibration_data(256)
    
    # Detect hardware configuration
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("GPU: Not available")
    
    cpu_count = os.cpu_count()
    print(f"CPU Cores: {cpu_count}")
    
    # Configure based on hardware
    if has_cuda and gpu_memory >= 8:
        # High-end GPU configuration
        config = {
            "method": "gptq",
            "hardware_config": {
                "device": "cuda",
                "mixed_precision": True,
                "use_flash_attention": True,
                "optimize_for_throughput": True
            }
        }
        print("\nUsing high-end GPU configuration")
        
    elif has_cuda and gpu_memory >= 4:
        # Mid-range GPU configuration
        config = {
            "method": "gptq",
            "hardware_config": {
                "device": "cuda",
                "enable_cpu_offload": True,
                "mixed_precision": True,
                "gradient_checkpointing": True
            }
        }
        print("\nUsing mid-range GPU configuration")
        
    else:
        # CPU-only configuration
        config = {
            "method": "gguf",  # More efficient for CPU
            "hardware_config": {
                "device": "cpu",
                "num_threads": min(cpu_count, 8),
                "use_mkl": True,
                "streaming_quantization": True
            }
        }
        print("\nUsing CPU-only configuration")
    
    # Perform quantization
    start_time = time.time()
    
    result = QuantLLM.quantize(
        model=model_name,
        bits=4,
        calibration_data=calibration_data,
        optimization_target="balanced",
        output_dir="./quantized_models/gpt2_hardware_optimized",
        **config
    )
    
    end_time = time.time()
    
    print(f"\nQuantization completed in {end_time - start_time:.1f} seconds")
    print(f"Method used: {config['method'].upper()}")
    print(f"Quantized size: {result.quantized_size_mb:.1f} MB")
    print(f"Compression ratio: {result.compression_ratio:.2f}x")
    
    return result

def main():
    """Run all model size examples."""
    logger = setup_logging()
    
    print("QuantLLM Model Size Examples")
    print("=" * 60)
    print("This script demonstrates quantization strategies for different model sizes")
    print("and hardware configurations.")
    
    # Create output directory
    os.makedirs("./quantized_models", exist_ok=True)
    
    try:
        # Run examples
        logger.info("Starting small model example...")
        small_result = quantize_small_model()
        
        logger.info("Starting medium model example...")
        medium_result = quantize_medium_model()
        
        logger.info("Starting large model example...")
        large_result = quantize_large_model()
        
        logger.info("Starting bit-width comparison...")
        bitwidth_results = quantize_with_different_bit_widths()
        
        logger.info("Starting hardware-specific optimization...")
        hardware_result = quantize_for_specific_hardware()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated models:")
        print("- ./quantized_models/gpt2_4bit/")
        print("- ./quantized_models/gpt2_large_4bit/")
        print("- ./quantized_models/diallogpt_large_*_4bit/")
        print("- ./quantized_models/gpt2_*bit/")
        print("- ./quantized_models/gpt2_hardware_optimized/")
        
        print("\nKey takeaways:")
        print("1. Small models can use standard settings")
        print("2. Medium models benefit from quality-focused optimization")
        print("3. Large models require aggressive memory management")
        print("4. Lower bit-widths trade quality for size")
        print("5. Hardware-specific optimization improves performance")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    main()