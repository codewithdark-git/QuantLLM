Model API
=========

Model
-----

.. automodule:: quantllm.model.model
   :members:
   :undoc-members:
   :show-inheritance:

Model Configuration
-----------------

.. automodule:: quantllm.model.lora_config
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
------------

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from quantllm import Model, ModelConfig

    # Configure model
    config = ModelConfig(
        model_name="facebook/opt-125m",
        load_in_4bit=True
    )

    # Load model
    model = Model(config)
    model_instance = model.get_model()

With LoRA
~~~~~~~~

.. code-block:: python

    config = ModelConfig(
        model_name="facebook/opt-125m",
        load_in_4bit=True,
        use_lora=True
    )
    model = Model(config)

CPU Offloading
~~~~~~~~~~~~

.. code-block:: python

    config = ModelConfig(
        model_name="facebook/opt-125m",
        cpu_offload=True
    )
    model = Model(config)

Advanced Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = ModelConfig(
        model_name="facebook/opt-125m",
        load_in_4bit=True,
        use_lora=True,
        gradient_checkpointing=True,
        bf16=True,
        trust_remote_code=True
    )
    model = Model(config)