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
        use_packed=True,     # Enable weight packing
        cpu_offload=False,   # Offload to CPU
    )
    
    # Quantize model
    quantized_model = quantizer.quantize() # Calibration data can be optionally passed here
    
    # Export to GGUF format
    quantizer.convert_to_gguf("model-q4.gguf")

Choosing the Right Method
------------------------

- **GPTQ**: Best for highest accuracy with slightly slower quantization. The GPTQ method in QuantLLM involves computing Hessian matrix information. This information is primarily used for activation-based weight reordering when `actorder=True`. Users should note that the detailed iterative weight updates using the full Hessian inverse, as found in some canonical GPTQ literature, may not be fully implemented in the current layer quantization step. The system logs warnings if the Hessian is computed but not fully utilized in this manner.
- **AWQ**: Best balance of speed and accuracy, good for general use
- **GGUF**: Best for deployment and inference with CTransformers. Key parameters include:
    - `cpu_offload: bool = False`: If True, attempts to offload parts of the computation and model data to CPU memory, reducing GPU memory usage at the cost of speed. Defaults to False.

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
- **group_size**: Size of quantization groups (behavior can vary; e.g., -1 for per-tensor in AWQ, specific positive values for GPTQ/GGUF grouping)
- **calibration_data**: Data used for computing statistics (optional for some GGUF modes, but recommended for others)
- **device**: Specifies the primary computation device ('cpu' or 'cuda') for the quantizer.

Specific parameters for GPTQ:
- **actorder**: Enables activation ordering, potentially improving accuracy.
- **use_triton**: Enables the use of Triton kernels. Note: While this flag is present, custom Triton kernels specifically for accelerating GPTQ's core quantization algorithm (like Hessian computation or iterative weight updates) are not currently integrated into `GPTQQuantizer`. General model optimization kernels from `quantllm.quant.kernels` might be applicable separately.

Specific parameters for AWQ:
- **zero_point**: Enables zero-point computation for activations.
- **version**: Specifies the AWQ algorithm version.

Specific parameters for GGUF:
- **use_packed**: Enables weight packing for smaller model size.
- **cpu_offload**: If True, offloads parts of computation/model to CPU, reducing GPU memory. Defaults to False.

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

For a detailed example, refer to the 'Example Workflow' section presented earlier in this document.
