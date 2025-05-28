Getting Started
===============

Introduction
-----------

QuantLLM is a powerful library for quantizing and deploying large language models with a focus on memory efficiency and performance. The library now supports GGUF format, advanced progress tracking, and comprehensive benchmarking tools.

Installation
-----------

Install the base package:

.. code-block:: bash

    pip install quantllm

For GGUF support, install with extras:

.. code-block:: bash

    pip install quantllm[gguf]

Quick Start
----------

Here's a complete example showcasing GGUF quantization and benchmarking:

.. code-block:: python

    from quantllm import QuantLLM
    from quantllm.quant import GGUFQuantizer
    from transformers import AutoTokenizer

    # 1. Load tokenizer and prepare calibration data
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    calibration_text = ["This is an example text for calibration."] * 10
    calibration_data = tokenizer(calibration_text, return_tensors="pt", padding=True)["input_ids"]

    # 2. Quantize using high-level API
    quantized_model, benchmark_results = QuantLLM.quantize_from_pretrained(
        model_name_or_path=model_name,
        bits=4,                    # Quantization bits (2-8)
        group_size=32,            # Group size for quantization
        quant_type="Q4_K_M",      # GGUF quantization type
        calibration_data=calibration_data,
        benchmark=True,           # Run benchmarks
        benchmark_input_shape=(1, 32),
        benchmark_steps=50,
        cpu_offload=False,       # Set to True for large models
        chunk_size=1000          # Process in chunks for memory efficiency
    )

    # 3. Save the quantized model
    QuantLLM.save_quantized_model(
        model=quantized_model,
        output_path="quantized_model",
        save_tokenizer=True
    )

    # 4. Convert to GGUF format
    QuantLLM.convert_to_gguf(
        model=quantized_model,
        output_path="model.gguf"
    )

Core Features
------------

Advanced GGUF Quantization
~~~~~~~~~~~~~~~~~~~~~~~

The library supports various GGUF quantization types:

* **2-bit Quantization**
    * Q2_K: Best for extreme compression
    * Suitable for smaller models or when size is critical

* **4-bit Quantization**
    * Q4_K_S: Standard 4-bit quantization
    * Q4_K_M: 4-bit quantization with improved accuracy
    * Best balance of size and quality

* **8-bit Quantization**
    * Q8_0: High-precision 8-bit quantization
    * Best for quality-critical applications

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~

* Chunk-based quantization for large models
* Automatic device management
* CPU offloading support
* Progress tracking with memory statistics

Detailed Examples
---------------

1. Direct GGUF Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~

For more control over the quantization process:

.. code-block:: python

    from quantllm.quant import GGUFQuantizer
    import torch

    # Initialize quantizer with detailed configuration
    quantizer = GGUFQuantizer(
        model_name="facebook/opt-125m",
        bits=4,
        group_size=32,
        quant_type="Q4_K_M",
        use_packed=True,
        desc_act=False,
        desc_ten=False,
        legacy_format=False,
        batch_size=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cpu_offload=False,
        gradient_checkpointing=False,
        chunk_size=1000
    )

    # Quantize the model
    quantized_model = quantizer.quantize(calibration_data=calibration_data)

    # Convert to GGUF format with progress tracking
    quantizer.convert_to_gguf("model.gguf")

2. Comprehensive Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate quantization performance:

.. code-block:: python

    from quantllm.utils.benchmark import QuantizationBenchmark

    # Initialize benchmark
    benchmark = QuantizationBenchmark(
        model=model,
        calibration_data=calibration_data,
        input_shape=(1, 32),
        num_inference_steps=100,
        device="cuda",
        num_warmup_steps=10
    )

    # Run benchmarks and get detailed metrics
    results = benchmark.run_all_benchmarks()
    
    # Print detailed report
    benchmark.print_report()

    # Optional: Generate visualization
    benchmark.plot_comparison("benchmark_results.png")

3. Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

For large models with memory constraints:

.. code-block:: python

    # Configure for memory efficiency
    quantizer = GGUFQuantizer(
        model_name="facebook/opt-1.3b",  # Larger model
        bits=4,
        group_size=32,
        cpu_offload=True,      # Enable CPU offloading
        chunk_size=500,        # Smaller chunks for memory efficiency
        gradient_checkpointing=True
    )

    # Process in chunks with progress display
    quantized_model = quantizer.quantize(calibration_data)

Supported GGUF Types
------------------

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

Best Practices
------------

1. **Memory Management**
    * Use `cpu_offload=True` for models larger than 70% of GPU memory
    * Adjust `chunk_size` based on available memory
    * Enable `gradient_checkpointing` for large models

2. **Quantization Selection**
    * Use Q4_K_M for general use cases
    * Use Q2_K for extreme compression needs
    * Use Q8_0 for quality-critical applications

3. **Performance Optimization**
    * Run benchmarks to find optimal settings
    * Use appropriate batch sizes
    * Monitor memory usage with built-in tools

4. **Progress Tracking**
    * Use the built-in progress bars
    * Monitor layer-wise quantization
    * Track memory usage during processing

Next Steps
---------

* Check out our :doc:`tutorials/index` for more examples
* Read the :doc:`api_reference/index` for API details
* See :doc:`advanced_usage/index` for advanced features
* Visit :doc:`deployment` for deployment guides