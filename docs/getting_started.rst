Getting Started
===============

Quick Start
----------

This guide will help you get started with QuantLLM quickly. Here's a simple example of fine-tuning a language model:

.. code-block:: python

    from quantllm import (
        Model, ModelConfig, 
        LoadDataset, DatasetConfig,
        FineTuningTrainer, TrainingConfig
    )

    # 1. Load and configure model
    model_config = ModelConfig(
        model_name="facebook/opt-125m",
        load_in_4bit=True  # Enable 4-bit quantization
    )
    model = Model(model_config).get_model()

    # 2. Load and prepare dataset
    dataset = LoadDataset().load_hf_dataset("imdb")
    
    # 3. Configure training
    training_config = TrainingConfig(
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=8
    )

    # 4. Train model
    trainer = FineTuningTrainer(
        model=model,
        training_config=training_config
    )
    trainer.train()

Core Features
------------

* **Efficient Quantization**: 4-bit and 8-bit quantization support
* **Hardware Optimization**: Automatic hardware detection and optimization
* **LoRA Integration**: Parameter-efficient fine-tuning
* **Progress Tracking**: Rich logging and visualization
* **Easy Deployment**: Simple export and deployment options

Key Concepts
-----------

Model Configuration
~~~~~~~~~~~~~~~~~

The ModelConfig class helps configure model loading:

.. code-block:: python

    config = ModelConfig(
        model_name="facebook/opt-125m",
        load_in_4bit=True,    # Enable 4-bit quantization
        use_lora=True,        # Enable LoRA
        cpu_offload=True      # Enable CPU offloading
    )

Dataset Handling
~~~~~~~~~~~~~~

Load and preprocess datasets easily:

.. code-block:: python

    dataset_config = DatasetConfig(
        dataset_name="imdb",
        text_column="text",
        label_column="label",
        max_length=512
    )

Training Configuration
~~~~~~~~~~~~~~~~~~~

Configure training parameters:

.. code-block:: python

    training_config = TrainingConfig(
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=4
    )

Progress Tracking
~~~~~~~~~~~~~~

Monitor training progress:

.. code-block:: python

    from quantllm import TrainingLogger

    logger = TrainingLogger()
    trainer = FineTuningTrainer(
        model=model,
        logger=logger
    )

Next Steps
---------

* Check out our :doc:`tutorials/index` for detailed examples
* Read the :doc:`api_reference/index` for complete API documentation
* See :doc:`advanced_usage/index` for advanced features
* Visit :doc:`deployment` for deployment options