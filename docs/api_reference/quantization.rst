# QuantLLM: Advanced Model Quantization
===================================

💫 Introduction
------------
QuantLLM is a powerful library that provides state-of-the-art quantization methods to compress large language models while maintaining their performance. Supporting multiple quantization methods (AWQ, GPTQ, GGUF), it enables efficient model deployment in production environments.

🚀 Getting Started
---------------
QuantLLM offers multiple quantization methods, each optimized for different use cases. The high-level `QuantLLM` API provides a simple interface to quantize models while the low-level API gives you fine-grained control over the quantization process.

Key Features:
- Multiple quantization methods (AWQ, GPTQ, GGUF)
- Memory-efficient processing
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

    # 1. Model and Method Selection
    model_name = "facebook/opt-125m"  # Any Hugging Face model
    method = "awq"  # Choose: 'awq', 'gptq', or 'gguf'

    # 2. Configure Quantization
    quant_config = {
        "bits": 4,                # Quantization bits (2-8)
        "group_size": 128,        # Size of quantization groups
        "zero_point": True,       # Zero-point quantization (AWQ)
        "version": "v2",          # AWQ algorithm version
        "scale_dtype": "fp32",    # Scale factor data type
        "batch_size": 4          # Processing batch size
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
        print("Starting quantization process...")
        start_time = time.time()
        
        # Perform quantization
        quantized_model, tokenizer = QuantLLM.quantize_from_pretrained(
            model_name=model_name,
            method=method,
            quant_config_dict=quant_config,
            calibration_data=inputs["input_ids"],
            calibration_steps=50,
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
        
        # 6. Save Quantized Model (Optional)
        save_path = "./quantized_model"
        quantized_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
        
    except Exception as e:
        print(f"Error during quantization: {str(e)}")
        raise

    # Define quantization configuration
    quant_config = {
        "bits": 4,
        "group_size": 128,
        "zero_point": True, # AWQ specific
        "awq_version": "v2"   # AWQ specific (maps to 'version')
    }

    # Prepare dummy calibration data (replace with your actual data)
    # For demonstration, creating random data.
    # The shape and content should be representative of your model's input.
    # Tokenizer is usually needed to prepare real calibration data.
    # For this example, let's assume calibration data is a tensor.
    # If the model needs input_ids, it should be shaped like (num_samples, seq_len)
    # If the model's first layer takes features directly, it might be (num_samples, feature_dim)
    # The factory passes this data to the specific quantizer.
    # BaseQuantizer's prepare_calibration_data expects a torch.Tensor.
    
    # For opt-125m, tokenizer.model_max_length is 2048, hidden_size is 768.
    # A simple approach for dummy calibration data:
    num_calibration_samples = 10
    sequence_length = 32 # A shorter sequence length for dummy data
    # Assuming calibration data is a tensor of input features for simplicity here.
    # In a real scenario, this would be tokenized input_ids.
    # The DummyModel in tests uses calibration_data as direct input if last dim matches.
    # Actual models need tokenized input_ids. The factory itself doesn't tokenize.
    # The user must provide calibration_data in the format expected by the model.
    # For now, we'll create a simple tensor.
    # For a real LLM, this should be tokenized sequences.
    # Let's create dummy input_ids as calibration data
    dummy_input_ids = torch.randint(0, 30000, (num_calibration_samples, sequence_length))


    try:
        quantized_model, tokenizer = QuantizerFactory.quantize_from_pretrained(
            model_name_or_path=model_name,
            method=method,
            quant_config_dict=quant_config,
            calibration_data=dummy_input_ids, # Pass input_ids directly
            calibration_steps=50, # For AWQ
            device="cpu" # Specify 'cuda' for GPU
        )
        print(f"Model {model_name} quantized with {method} successfully.")
        print(f"Quantized model: {quantized_model}")
        print(f"Tokenizer: {tokenizer}")

        # You can now use the quantized_model and tokenizer for inference
        # For example:
        # if tokenizer:
        #     inputs = tokenizer("Hello, world!", return_tensors="pt").to(quantized_model.device)
        #     with torch.no_grad():
        #         outputs = quantized_model(**inputs)
        #     print("Inference output:", outputs)

    except Exception as e:
        print(f"An error occurred during quantization: {e}")

Main Parameters of `quantize_from_pretrained`
---------------------------------------------

-   ``model_name_or_path (str)``: Hugging Face model ID (e.g., "facebook/opt-125m") or a local path to a pretrained model.
-   ``method (str)``: The quantization method to use. Supported values are ``'awq'``, ``'gptq'``, or ``'gguf'``.
-   ``quant_config_dict (Optional[Dict[str, Any]])``: A dictionary containing parameters for the chosen quantization method.
    -   **Common Keys (can be used for most methods, defaults may apply):**
        -   `bits (int)`: Number of bits for quantization (e.g., 4, 8). Default: 4.
        -   `group_size (int)`: Size of quantization groups. Default: 128.
        -   `batch_size (int)`: Batch size used internally by the quantizer during its initialization/calibration steps. Default: 4.
    -   **AWQ Specific Keys:**
        -   `zero_point (bool)`: Enable/disable zero-point for activations. Default: True.
        -   `awq_version (str)`: AWQ algorithm version (e.g., "v1", "v2"). Default: "v2". (Maps to `version` in `AWQQuantizer`).
    -   **GPTQ Specific Keys:**
        -   `actorder (bool)`: Enable activation-order quantization. Default: True.
        -   `percdamp (float)`: Dampening percentage for Hessian update. Default: 0.01.
        -   `sym (bool)`: Use symmetric quantization for weights. Default: True.
    -   **GGUF Specific Keys:**
        -   `use_packed (bool)`: Enable weight packing for GGUF. Default: True.
        -   `cpu_offload (bool)`: Offload quantized layers to CPU. Default: False.
        -   `desc_act (bool)`: Describe activations in GGUF metadata. Default: False.
        -   `desc_ten (bool)`: Describe tensors in GGUF metadata. Default: False.
        -   `legacy_format (bool)`: Use legacy GGUF format. Default: False.
    Refer to the docstring of `QuantizerFactory.quantize_from_pretrained` and individual quantizer classes for more details on all available parameters.
-   ``calibration_data (Optional[Any])``: Data required for quantization, typically a `torch.Tensor`. The format depends on the model's input requirements (e.g., tokenized `input_ids`). This data is passed to the underlying quantizer.
-   ``calibration_steps (Optional[int])``: Number of calibration steps. This is particularly relevant for methods like AWQ that use it in their `quantize()` method. Default: 100.
-   ``device (Optional[str])``: The device to run the quantization on (e.g., "cpu", "cuda", "cuda:0"). If `None`, the default device selection logic within the quantizers (usually prioritizing CUDA if available) will be used.

The method returns a tuple containing the quantized ``PreTrainedModel`` and its associated tokenizer (if loadable).

For a full example demonstrating quantization and pushing to the Hugging Face Hub, see the script in ``examples/01_quantize_and_push_to_hub.py``.

Advanced: Direct Quantizer Usage
================================

While `QuantizerFactory` is recommended for ease of use, you can also use the individual quantizer classes directly for more fine-grained control or custom workflows.

Common Parameters for Direct Initialization
-------------------------------------------

All quantizers share a common set of parameters in their `__init__` method, inherited from `BaseQuantizer`:

-   ``model_or_model_name_or_path (Union[str, PreTrainedModel])``: A Hugging Face model ID, a local path to a model, or an already loaded `PreTrainedModel` instance.
-   ``bits (int)``: Number of quantization bits (e.g., 2-8).
-   ``device (Optional[Union[str, torch.device]])``: Specifies the primary computation device ('cpu' or 'cuda') for the quantizer and the prepared model.

Individual Quantizer Details
----------------------------

Below are details specific to each quantization method when used directly.

### 1. GPTQ (`GPTQQuantizer`)

GPTQ offers Hessian-based quantization with activation ordering for high accuracy.

.. automodule:: quantllm.quant.gptq
    :noindex:

.. autoclass:: quantllm.quant.gptq.GPTQQuantizer
    :members: __init__, quantize
    :show-inheritance:
    :inherited-members:
    :undoc-members:

**Key `__init__` Parameters for `GPTQQuantizer`:**
- ``group_size (int)``: Size of quantization groups.
- ``actorder (bool)``: Enables activation ordering.
- ``sym (bool)``: Use symmetric quantization for weights.
- ``percdamp (float)``: Dampening for Hessian update.
- ``use_triton (bool)``: Note: Custom GPTQ Triton kernels are not yet fully integrated for core quantization steps.

**Usage Example (Direct):**

.. code-block:: python

    from quantllm.quant import GPTQQuantizer
    # Assuming 'model' is a loaded PreTrainedModel instance
    # and 'calibration_data' is prepared
    
    quantizer = GPTQQuantizer(
        model_or_model_name_or_path=model, # Can also be model name/path
        bits=4,
        group_size=128,
        actorder=True
    )
    quantized_model = quantizer.quantize(calibration_data=calibration_data)

### 2. AWQ (`AWQQuantizer`)

AWQ adapts quantization based on activation patterns.

.. automodule:: quantllm.quant.awq
    :noindex:

.. autoclass:: quantllm.quant.awq.AWQQuantizer
    :members: __init__, quantize
    :show-inheritance:
    :inherited-members:
    :undoc-members:

**Key `__init__` Parameters for `AWQQuantizer`:**
- ``group_size (int)``: Group size for quantization.
- ``zero_point (bool)``: Enable zero-point computation for activations.
- ``version (str)``: AWQ algorithm version.

**Usage Example (Direct):**

.. code-block:: python

    from quantllm.quant import AWQQuantizer
    # Assuming 'model' is a loaded PreTrainedModel instance
    # and 'calibration_data' is prepared
    
    quantizer = AWQQuantizer(
        model_or_model_name_or_path=model,
        bits=4,
        group_size=128,
        zero_point=True
    )
    quantized_model = quantizer.quantize(
        calibration_data=calibration_data,
        calibration_steps=100 # AWQ's quantize method takes this
    )

### 3. GGUF (`GGUFQuantizer`)

GGUF provides an efficient format with CTransformers integration. It can also offload quantized layers to CPU.

.. automodule:: quantllm.quant.gguf
    :noindex:

.. autoclass:: quantllm.quant.gguf.GGUFQuantizer
    :members: __init__, quantize, convert_to_gguf
    :show-inheritance:
    :inherited-members:
    :undoc-members:

**Key `__init__` Parameters for `GGUFQuantizer`:**
- ``group_size (int)``: Group size.
- ``use_packed (bool)``: Enable weight packing.
- ``cpu_offload (bool)``: If True, quantized layers are placed on CPU.
- ``desc_act (bool)``, ``desc_ten (bool)``, ``legacy_format (bool)``: GGUF format-specific flags.

**Usage Example (Direct):**

.. code-block:: python

    from quantllm.quant import GGUFQuantizer
    # Assuming 'model' is a loaded PreTrainedModel instance
    
    quantizer = GGUFQuantizer(
        model_or_model_name_or_path=model,
        bits=4,
        group_size=32,
        use_packed=True,
        cpu_offload=False 
    )
    # Calibration data is optional for GGUF's quantize method but can be beneficial
    quantized_model = quantizer.quantize(calibration_data=calibration_data) 
    
    # Export to GGUF format
    quantizer.convert_to_gguf("model-q4.gguf")

Choosing the Right Method
------------------------

- **GPTQ**: Best for highest accuracy with slightly slower quantization. The GPTQ method in QuantLLM involves computing Hessian matrix information. This information is primarily used for activation-based weight reordering when `actorder=True`. Users should note that the detailed iterative weight updates using the full Hessian inverse, as found in some canonical GPTQ literature, may not be fully implemented in the current layer quantization step. The system logs warnings if the Hessian is computed but not fully utilized in this manner.
- **AWQ**: Best balance of speed and accuracy, good for general use.
- **GGUF**: Best for deployment and inference with CTransformers.

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

For a detailed example of direct quantizer usage, you can adapt the `QuantizerFactory` example by instantiating the chosen quantizer directly and calling its methods.
