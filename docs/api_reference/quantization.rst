# QuantLLM: GGUF Model Quantization
===================================

💫 Introduction
------------
QuantLLM provides efficient model quantization using the GGUF (GGML Universal Format) method, enabling memory-efficient deployment of large language models. The library focuses on providing robust quantization with comprehensive progress tracking and benchmarking capabilities.

🚀 Getting Started
---------------
QuantLLM offers both high-level and low-level APIs for GGUF quantization. The high-level `QuantLLM` API provides a simple interface, while the low-level `GGUFQuantizer` gives you fine-grained control over the quantization process.

Key Features:
- Multiple GGUF quantization types (Q2_K to Q8_0)
- Memory-efficient chunk-based processing
- Hardware-specific optimizations
- Comprehensive metrics and logging
- Easy model export and deployment

Complete Example
---------------

.. code-block:: python

    import torch
    from quantllm import QuantLLM
    from transformers import AutoTokenizer
    import time

    # 1. Model Selection
    model_name = "facebook/opt-125m"  # Any Hugging Face model

    # 2. Configure GGUF Quantization
    quant_config = {
        "bits": 4,                # Quantization bits (2-8)
        "group_size": 32,        # Size of quantization groups
        "quant_type": "Q4_K_M",  # GGUF quantization type
        "use_packed": True,      # Use packed format
        "cpu_offload": False,    # CPU offloading for large models
        "chunk_size": 1000      # Chunk size for memory efficiency
    }

    # 3. Prepare Calibration Data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Representative text samples for calibration
    calibration_texts = [
        "Translate English to French: Hello, how are you?",
        "Summarize this text: The quick brown fox jumps over the lazy dog",
        "What is the capital of France?",
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing in simple terms."
    ]
    
    # Tokenize with proper padding and attention masks
    inputs = tokenizer(
        calibration_texts, 
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # 4. Model Quantization with Error Handling
    try:
        print("Starting GGUF quantization process...")
        start_time = time.time()
        
        # Perform quantization
        quantized_model, benchmark_results = QuantLLM.quantize_from_pretrained(
            model_name_or_path=model_name,
            bits=4,
            group_size=32,
            quant_type="Q4_K_M",
            calibration_data=inputs["input_ids"],
            benchmark=True,
            benchmark_input_shape=(1, 32),
            benchmark_steps=50,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"Quantization completed in {time.time() - start_time:.2f} seconds")
        
        # 5. Model Validation
        test_input = "Translate this to French: The weather is beautiful today."
        inputs = tokenizer(test_input, return_tensors="pt").to(quantized_model.device)
        
        with torch.no_grad():
            outputs = quantized_model.generate(
                **inputs,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7
            )
            
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test Output: {result}")
        
        # 6. Save and Convert to GGUF
        QuantLLM.save_quantized_model(
            model=quantized_model,
            output_path="./quantized_model",
            save_tokenizer=True
        )
        
        QuantLLM.convert_to_gguf(
            model=quantized_model,
            output_path="model.gguf"
        )
        print("Model saved and converted to GGUF format")
        
    except Exception as e:
        print(f"Error during quantization: {str(e)}")
        raise

Main Parameters of `quantize_from_pretrained`
---------------------------------------------

-   ``model_name_or_path (str)``: Hugging Face model ID (e.g., "facebook/opt-125m") or a local path to a pretrained model.
-   ``bits (int)``: Number of bits for quantization (2-8). Default: 4.
-   ``group_size (int)``: Size of quantization groups. Default: 32.
-   ``quant_type (str)``: GGUF quantization type (e.g., "Q4_K_M"). Optional.
-   ``use_packed (bool)``: Enable weight packing. Default: True.
-   ``cpu_offload (bool)``: Offload layers to CPU for memory efficiency. Default: False.
-   ``chunk_size (int)``: Size of processing chunks. Default: 1000.
-   ``calibration_data (torch.Tensor)``: Input IDs for calibration.
-   ``benchmark (bool)``: Whether to run benchmarks. Default: False.
-   ``benchmark_input_shape (tuple)``: Shape for benchmark inputs.
-   ``benchmark_steps (int)``: Number of benchmark steps.
-   ``device (str)``: Device for quantization ("cpu" or "cuda").

GGUF Quantization Types
----------------------

============  ================  ====================
Bits          Types            Description
============  ================  ====================
2-bit         Q2_K             Extreme compression
3-bit         Q3_K_S           Small size
3-bit         Q3_K_M           Medium accuracy
3-bit         Q3_K_L           Better accuracy
4-bit         Q4_K_S           Standard quality
4-bit         Q4_K_M           Better quality
5-bit         Q5_K_S           High quality
5-bit         Q5_K_M           Higher quality
6-bit         Q6_K             Very high quality
8-bit         Q8_0             Best quality
============  ================  ====================

Direct GGUFQuantizer Usage
=========================

For more fine-grained control, you can use the `GGUFQuantizer` class directly:

.. code-block:: python

    from quantllm.quant import GGUFQuantizer
    
    # Initialize quantizer
    quantizer = GGUFQuantizer(
        model_name="facebook/opt-125m",
        bits=4,
        group_size=32,
        quant_type="Q4_K_M",
        use_packed=True,
        cpu_offload=False,
        chunk_size=1000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Quantize model
    quantized_model = quantizer.quantize(calibration_data=calibration_data)
    
    # Convert to GGUF format
    quantizer.convert_to_gguf("model.gguf")

Memory-Efficient Processing
-------------------------

For large models, QuantLLM provides several memory optimization features:

1. **Chunk-based Processing**
   
   .. code-block:: python

       quantizer = GGUFQuantizer(
           model_name="large-model",
           chunk_size=500,  # Process in smaller chunks
           cpu_offload=True  # Offload to CPU when needed
       )

2. **Progress Tracking**
   
   The quantization process provides detailed progress information:
   - Layer-wise quantization progress
   - Memory usage statistics
   - Estimated time remaining
   - Layer shape information

3. **Benchmarking**
   
   .. code-block:: python

       from quantllm.utils.benchmark import QuantizationBenchmark
       
       benchmark = QuantizationBenchmark(
           model=model,
           calibration_data=calibration_data,
           input_shape=(1, 32),
           num_inference_steps=100
       )
       results = benchmark.run_all_benchmarks()
       benchmark.print_report()

Best Practices
-------------

1. **Memory Management**
   - Use `cpu_offload=True` for models larger than 70% of GPU memory
   - Adjust `chunk_size` based on available memory
   - Monitor memory usage with benchmarking tools

2. **Quantization Type Selection**
   - Use Q4_K_M for general use cases
   - Use Q2_K for extreme compression needs
   - Use Q8_0 for quality-critical applications

3. **Performance Optimization**
   - Run benchmarks to find optimal settings
   - Use appropriate batch sizes
   - Enable progress tracking for monitoring

For detailed examples, check out the `examples/` directory or refer to the getting started guide.
