Quick Start Tutorial
==================

This tutorial will walk you through fine-tuning a small language model on the IMDB dataset using QuantLLM.

Setup
-----

First, install QuantLLM:

.. code-block:: bash

    pip install quantllm[gpu]  # For GPU support
    # or
    pip install quantllm[cpu]  # For CPU-only

Basic Example
------------

Here's a complete example that demonstrates the core features of QuantLLM:

.. code-block:: python

    from quantllm import (
        Model, ModelConfig,
        LoadDataset, DatasetConfig,
        DatasetPreprocessor, DatasetSplitter,
        FineTuningTrainer, TrainingConfig,
        TrainingLogger
    )

    # Initialize logger
    logger = TrainingLogger()

    # 1. Configure and load model
    logger.log_info("Loading model...")
    model_config = ModelConfig(
        model_name="facebook/opt-125m",  # Small model for demonstration
        load_in_4bit=True,               # Enable 4-bit quantization
        use_lora=True                    # Enable LoRA for efficient fine-tuning
    )
    model = Model(model_config)

    # 2. Load and prepare dataset
    logger.log_info("Preparing dataset...")
    dataset = LoadDataset().load_hf_dataset("imdb")
    
    # Split dataset
    splitter = DatasetSplitter()
    train_dataset, val_dataset, test_dataset = splitter.train_val_test_split(
        dataset["train"],  # Use train split from IMDB
        train_size=0.8,
        val_size=0.1,
        test_size=0.1
    )

    # Preprocess datasets
    tokenizer = model.get_tokenizer()
    preprocessor = DatasetPreprocessor(tokenizer)
    train_processed, val_processed, test_processed = preprocessor.tokenize_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        max_length=512,
        text_column="text"
    )

    # 3. Configure training
    training_config = TrainingConfig(
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=50
    )

    # 4. Initialize trainer
    trainer = FineTuningTrainer(
        model=model.get_model(),
        training_config=training_config,
        train_dataloader=train_processed,
        eval_dataloader=val_processed,
        logger=logger
    )

    # 5. Train model
    trainer.train()

    # 6. Evaluate on test set
    logger.log_info("Evaluating model...")
    test_metrics = trainer.evaluate(test_processed)
    logger.log_info(f"Test metrics: {test_metrics}")

Step-by-Step Explanation
----------------------

1. Model Configuration
~~~~~~~~~~~~~~~~~~

The ModelConfig class sets up model loading options:

- ``model_name``: Which model to load from HuggingFace
- ``load_in_4bit``: Enable 4-bit quantization for memory efficiency
- ``use_lora``: Enable LoRA for parameter-efficient fine-tuning

2. Dataset Preparation
~~~~~~~~~~~~~~~~~~~

We use three main classes for dataset handling:

- ``LoadDataset``: Loads datasets from HuggingFace or local files
- ``DatasetSplitter``: Creates train/validation/test splits
- ``DatasetPreprocessor``: Handles tokenization and preprocessing

3. Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

TrainingConfig controls the training process:

- ``learning_rate``: How fast the model learns
- ``num_epochs``: How many times to process the dataset
- ``batch_size``: Samples processed at once
- ``gradient_accumulation_steps``: Accumulate gradients for larger effective batch size

4. Training
~~~~~~~~~

The FineTuningTrainer handles the training loop:

- Manages model updates
- Tracks progress
- Handles checkpointing
- Provides evaluation

Monitoring Progress
-----------------

The TrainingLogger provides rich progress information:

.. code-block:: python

    logger = TrainingLogger()
    logger.log_info("Starting training...")  # Basic logging
    logger.log_metrics({"loss": 0.5})        # Track metrics
    logger.log_success("Training complete!")  # Success messages

Next Steps
---------

- Try with different models from HuggingFace
- Experiment with training parameters
- Use your own dataset
- Enable advanced features like gradient checkpointing

Check out other tutorials for more advanced usage:

- :doc:`text_classification` for detailed text classification
- :doc:`custom_dataset` for using your own data
- :doc:`distributed_training` for multi-GPU training