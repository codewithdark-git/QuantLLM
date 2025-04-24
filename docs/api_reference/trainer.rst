Trainer API
==========

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