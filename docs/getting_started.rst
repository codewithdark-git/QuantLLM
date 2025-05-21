Getting Started
===============

Quick Start
----------

QuantLLM is designed to make working with large language models more accessible and efficient. Here's a complete example showcasing its key features:

.. code-block:: python

    from quantllm import (
        Model, ModelConfig, 
        LoadDataset, DatasetConfig,
        FineTuningTrainer, TrainingConfig,
        TrainingLogger
    )

    # Initialize logger for rich progress tracking
    logger = TrainingLogger()  # This will display the ASCII art logo!

    # 1. Load and configure model with best practices
    model_config = ModelConfig(
        model_name="facebook/opt-125m",
        load_in_4bit=True,     # Enable memory-efficient 4-bit quantization
        use_lora=True,         # Enable parameter-efficient fine-tuning
        gradient_checkpointing=True  # Reduce memory usage during training
    )
    model = Model(model_config).get_model()

    # 2. Load and prepare dataset with automatic preprocessing
    dataset = LoadDataset().load_hf_dataset("imdb")
    dataset_config = DatasetConfig(
        text_column="text",
        label_column="label",
        max_length=512
    )
    
    # 3. Configure training with optimized defaults
    training_config = TrainingConfig(
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=4,  # For larger effective batch sizes
        warmup_ratio=0.1,              # Gradual learning rate warmup
        evaluation_strategy="steps",    # Regular evaluation during training
        eval_steps=100
    )

    # 4. Initialize trainer with progress tracking
    trainer = FineTuningTrainer(
        model=model,
        training_config=training_config,
        logger=logger  # Enable rich progress tracking
    )
    
    # 5. Start training with automatic hardware optimization
    trainer.train()

Core Features
------------

* **Advanced Quantization**
    * 4-bit and 8-bit quantization for up to 75% memory reduction
    * Automatic format selection based on your hardware
    * Zero-shot quantization with minimal accuracy loss

* **Efficient Fine-tuning**
    * LoRA support for parameter-efficient training
    * Gradient checkpointing for reduced memory usage
    * Automatic mixed precision training

* **Hardware Optimization**
    * Automatic hardware detection (CUDA, MPS, CPU)
    * Optimal settings for your specific GPU
    * CPU offloading for large models

* **Rich Progress Tracking**
    * Beautiful terminal-based progress display
    * Detailed training metrics and logs
    * Integration with WandB and TensorBoard

* **Production Ready**
    * Simple export to ONNX and TorchScript
    * Quantized model deployment
    * GPU and CPU inference optimization

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