Installation Guide
=================

Requirements
-----------

QuantLLM requires Python 3.10 or later. The following are the core dependencies:

* PyTorch >= 2.0.0
* Transformers >= 4.30.0
* CUDA Toolkit (optional, but recommended for GPU support)

Installation Methods
------------------

1. From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

Basic installation:

.. code-block:: bash

    pip install quantllm

With GGUF support (recommended for deployment):

.. code-block:: bash

    pip install quantllm[gguf]

With development tools:

.. code-block:: bash

    pip install quantllm[dev]

2. From Source
~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/codewithdark-git/DiffusionLM.git
    cd DiffusionLM
    pip install -e .

For development installation:

.. code-block:: bash

    pip install -e .[dev,gguf]

Hardware Requirements
------------------

Minimum Requirements:
~~~~~~~~~~~~~~~~~~

* CPU: 4+ cores
* RAM: 16GB+
* Storage: 10GB+ free space
* Python: 3.10+

Recommended for Large Models:
~~~~~~~~~~~~~~~~~~~~~~~~~

* CPU: 8+ cores
* RAM: 32GB+
* GPU: NVIDIA GPU with 8GB+ VRAM
* CUDA: 11.7 or later
* Storage: 20GB+ free space

GGUF Support
----------

GGUF (GGML Universal Format) support requires additional dependencies:

* llama-cpp-python >= 0.2.0
* ctransformers >= 0.2.0 (optional)

These are automatically installed with:

.. code-block:: bash

    pip install quantllm[gguf]

Verify Installation
----------------

You can verify your installation by running:

.. code-block:: python

    import quantllm
    from quantllm.quant import GGUFQuantizer
    
    # Check GGUF support
    print(f"GGUF Support: {GGUFQuantizer.CT_AVAILABLE}")
    
    # Check CUDA availability
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

Common Issues
-----------

1. CUDA Compatibility
~~~~~~~~~~~~~~~~~~

If you encounter CUDA errors:

.. code-block:: bash

    # Install PyTorch with specific CUDA version
    pip install torch --index-url https://download.pytorch.org/whl/cu118

2. Memory Issues
~~~~~~~~~~~~~

For large models, enable memory optimization:

.. code-block:: python

    quantizer = GGUFQuantizer(
        model_name="large-model",
        cpu_offload=True,
        chunk_size=500,
        gradient_checkpointing=True
    )

3. GGUF Conversion Issues
~~~~~~~~~~~~~~~~~~~~~~

If GGUF conversion fails:

1. Ensure llama-cpp-python is installed:
   
   .. code-block:: bash

       pip install llama-cpp-python --upgrade

2. Check system compatibility:
   
   .. code-block:: bash

       python -c "from ctransformers import AutoModelForCausalLM; print('GGUF support available')"

Next Steps
---------

* Read the :doc:`getting_started` guide
* Check out :doc:`tutorials/index`
* See :doc:`advanced_usage/index` for advanced features
