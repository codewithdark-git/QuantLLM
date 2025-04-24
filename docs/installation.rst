Installation Guide
=================

Requirements
-----------

Before installing QuantLLM, ensure your system meets these requirements:

* Python >= 3.8
* PyTorch >= 2.0
* CUDA >= 11.7 (for GPU support)
* 16GB RAM (minimum)
* 8GB VRAM (recommended for GPU training)

Basic Installation
----------------

You can install QuantLLM using pip:

.. code-block:: bash

    pip install quantllm

For development installation:

.. code-block:: bash

    git clone https://github.com/codewithdark-git/QuantLLM.git
    cd QuantLLM
    pip install -e ".[dev]"

GPU Support
----------

For GPU acceleration, install with CUDA support:

.. code-block:: bash

    pip install quantllm[gpu]

This will install additional dependencies like:

* bitsandbytes
* accelerate
* Flash Attention 2 (where supported)

Apple Silicon (M1/M2)
--------------------

For Apple Silicon Macs:

.. code-block:: bash

    pip install quantllm[mps]

CPU-Only
--------

For CPU-only installations:

.. code-block:: bash

    pip install quantllm[cpu]

Optional Dependencies
-------------------

Weights & Biases integration:

.. code-block:: bash

    pip install quantllm[wandb]

Full installation with all features:

.. code-block:: bash

    pip install quantllm[all]
