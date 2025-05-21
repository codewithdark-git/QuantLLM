Quantization Methods
==================

QuantLLM provides three primary methods for model quantization, each with its own advantages:

1. GPTQ (Goyal-Pham-Tan-Quant)
---------------------------------

GPTQ offers Hessian-based quantization with activation ordering for high accuracy:

.. code-block:: python

    from quantllm.quant import GPTQQuantizer
    
    # Initialize quantizer
    quantizer = GPTQQuantizer(
        model=model,
        bits=4,              # Quantization bits (2-8)
        group_size=128,      # Size of quantization groups
        actorder=True,       # Enable activation ordering
        use_triton=True      # Use Triton kernels for acceleration
    )
    
    # Quantize model
    quantized_model = quantizer.quantize(calibration_data=calibration_data)

2. AWQ (Activation-Aware Weight Quantization)
-------------------------------------------

AWQ adapts quantization based on activation patterns:

.. code-block:: python

    from quantllm.quant import AWQQuantizer
    
    quantizer = AWQQuantizer(
        model=model,
        bits=4,              # Quantization bits
        group_size=128,      # Group size for quantization
        zero_point=True,     # Enable zero point computation
        version="v2"         # AWQ version
    )
    
    # Quantize with activation statistics
    quantized_model = quantizer.quantize(
        calibration_data=calibration_data,
        calibration_steps=100
    )

3. GGUF (GGML Universal Format)
-----------------------------

GGUF provides an efficient format with CTransformers integration:

.. code-block:: python

    from quantllm.quant import GGUFQuantizer
    
    quantizer = GGUFQuantizer(
        model=model,
        bits=4,              # Quantization bits
        group_size=32,       # Group size
        use_packed=True      # Enable weight packing
    )
    
    # Quantize model
    quantized_model = quantizer.quantize()
    
    # Export to GGUF format
    quantizer.convert_to_gguf("model-q4.gguf")

Choosing the Right Method
------------------------

- **GPTQ**: Best for highest accuracy with slightly slower quantization
- **AWQ**: Best balance of speed and accuracy, good for general use
- **GGUF**: Best for deployment and inference with CTransformers

Resource Requirements
------------------

+-------------+------------+-------------+------------+
| Method      | Memory     | Speed       | Accuracy   |
+=============+============+=============+============+
| GPTQ        | High       | Slow        | Highest    |
+-------------+------------+-------------+------------+
| AWQ         | Medium     | Fast        | High       |
+-------------+------------+-------------+------------+
| GGUF        | Low        | Very Fast   | Good       |
+-------------+------------+-------------+------------+

Common Parameters
---------------

All quantizers support these common parameters:

- **bits**: Number of quantization bits (2-8)
- **group_size**: Size of quantization groups
- **calibration_data**: Data used for computing statistics

Example Workflow
--------------

Here's a complete example of quantizing a model:

.. code-block:: python

    import torch
    from quantllm import Model, ModelConfig
    from quantllm.quant import AWQQuantizer
    
    # 1. Load model
    model_config = ModelConfig(model_name="facebook/opt-350m")
    model = Model(model_config).get_model()
    
    # 2. Prepare calibration data
    calibration_data = prepare_calibration_data()  # Your calibration data
    
    # 3. Initialize quantizer
    quantizer = AWQQuantizer(
        model=model,
        bits=4,
        group_size=128
    )
    
    # 4. Quantize model
    quantized_model = quantizer.quantize(
        calibration_data=calibration_data,
        calibration_steps=100
    )
    
    # 5. Use the quantized model
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    outputs = quantized_model(**inputs)

For more detailed examples, see the `examples/quantization_examples.py` file in the repository.
