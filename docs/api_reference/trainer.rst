Trainer API
==========

QuantLLM provides a comprehensive training API with built-in support for quantization, 
efficient fine-tuning, and progress tracking.

Fine-Tuning Trainer
-----------------

.. automodule:: quantllm.trainer.trainer
   :members:
   :undoc-members:
   :show-inheritance:

Model Evaluator
-------------

.. automodule:: quantllm.trainer.evaluator
   :members:
   :undoc-members:
   :show-inheritance:

Training Logger
-------------

.. automodule:: quantllm.trainer.logger
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-----------

Complete Training Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from quantllm import (
        Model, ModelConfig, 
        FineTuningTrainer, TrainingConfig,
        TrainingLogger, CheckpointManager
    )

    # Initialize logger for beautiful progress display
    logger = TrainingLogger()

    # Configure model with advanced optimizations
    config = ModelConfig(
        model_name="facebook/opt-125m",
        load_in_4bit=True,         # Memory efficient!
        use_lora=True,             # Parameter efficient!
        gradient_checkpointing=True # Training efficient!
    )

    # Initialize training with rich features
    training_config = TrainingConfig(
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=4,
        # Advanced features
        warmup_ratio=0.1,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        logging_steps=10,
        # Mixed precision training
        fp16=True,
        # Multi-GPU support
        ddp_find_unused_parameters=False
    )

    # Setup checkpointing
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="./checkpoints",
        save_total_limit=3
    )

    # Initialize and train
    trainer = FineTuningTrainer(
        model=model,
        training_config=training_config,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        logger=logger,
        checkpoint_manager=checkpoint_manager
    )
    
    # Start training with full monitoring
    trainer.train()

Basic Training
~~~~~~~~~~~~

.. code-block:: python

    from quantllm import FineTuningTrainer, TrainingConfig

    config = TrainingConfig(
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=4
    )

    trainer = FineTuningTrainer(
        model=model,
        training_config=config,
        train_dataloader=train_loader,
        eval_dataloader=val_loader
    )
    trainer.train()

With Progress Tracking
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from quantllm import FineTuningTrainer, TrainingLogger

    logger = TrainingLogger()
    trainer = FineTuningTrainer(
        model=model,
        training_config=config,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        logger=logger
    )
    trainer.train()

Model Evaluation
~~~~~~~~~~~~~

.. code-block:: python

    from quantllm import ModelEvaluator

    evaluator = ModelEvaluator(
        model=model,
        eval_dataloader=test_loader
    )
    metrics = evaluator.evaluate()

Checkpoint Management
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from quantllm import CheckpointManager

    checkpoint_manager = CheckpointManager(
        checkpoint_dir="./checkpoints",
        save_total_limit=3
    )

    trainer = FineTuningTrainer(
        model=model,
        checkpoint_manager=checkpoint_manager,
        ...
    )

Custom Training Loop
~~~~~~~~~~~~~~~~~

.. code-block:: python

    class CustomTrainer(FineTuningTrainer):
        def training_step(self, batch):
            # Custom training logic
            pass

        def validation_step(self, batch):
            # Custom validation logic
            pass

    trainer = CustomTrainer(model=model, ...)