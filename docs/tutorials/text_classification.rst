Text Classification Tutorial
========================

This tutorial demonstrates how to fine-tune a model for text classification tasks using QuantLLM.

Task Overview
-----------

We'll fine-tune a model to classify movie reviews as positive or negative using the IMDB dataset.

Prerequisites
-----------

- QuantLLM installed with GPU support
- Basic understanding of transformers
- At least 8GB GPU VRAM (or 16GB RAM for CPU)

Complete Implementation
--------------------

.. code-block:: python

    import torch
    from quantllm import (
        Model, ModelConfig,
        LoadDataset, DatasetConfig,
        DatasetPreprocessor, DatasetSplitter,
        FineTuningTrainer, TrainingConfig,
        TrainingLogger, CheckpointManager
    )

    class TextClassificationTrainer:
        def __init__(self):
            self.logger = TrainingLogger()
            self.setup_model()
            self.setup_data()
            self.setup_training()

        def setup_model(self):
            """Configure and load the model."""
            self.logger.log_info("Setting up model...")
            
            # Use smaller model with LoRA for efficiency
            self.model_config = ModelConfig(
                model_name="facebook/opt-350m",
                load_in_4bit=True,
                use_lora=True,
                gradient_checkpointing=True
            )
            
            self.model = Model(self.model_config)
            self.tokenizer = self.model.get_tokenizer()

        def setup_data(self):
            """Load and prepare the IMDB dataset."""
            self.logger.log_info("Preparing dataset...")
            
            # Load dataset
            dataset = LoadDataset().load_hf_dataset("imdb")
            
            # Split dataset
            splitter = DatasetSplitter()
            self.train_dataset, self.val_dataset, self.test_dataset = splitter.train_val_test_split(
                dataset["train"],
                train_size=0.8,
                val_size=0.1,
                test_size=0.1
            )
            
            # Preprocess datasets
            preprocessor = DatasetPreprocessor(self.tokenizer)
            self.train_processed, self.val_processed, self.test_processed = preprocessor.tokenize_dataset(
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                test_dataset=self.test_dataset,
                max_length=512,
                text_column="text",
                label_column="label"
            )

        def setup_training(self):
            """Configure training parameters."""
            self.logger.log_info("Configuring training...")
            
            # Set up checkpoint management
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir="./checkpoints",
                save_total_limit=3
            )
            
            # Configure training
            self.training_config = TrainingConfig(
                learning_rate=2e-4,
                num_epochs=3,
                batch_size=8,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                eval_steps=500,
                save_steps=1000,
                logging_steps=50
            )
            
            # Initialize trainer
            self.trainer = FineTuningTrainer(
                model=self.model.get_model(),
                training_config=self.training_config,
                train_dataloader=self.train_processed,
                eval_dataloader=self.val_processed,
                logger=self.logger,
                checkpoint_manager=self.checkpoint_manager
            )

        def train(self):
            """Run the training process."""
            self.logger.log_info("Starting training...")
            self.trainer.train()
            
            # Evaluate on test set
            self.logger.log_info("Evaluating on test set...")
            test_metrics = self.trainer.evaluate(self.test_processed)
            self.logger.log_info(f"Test metrics: {test_metrics}")

        def predict(self, text: str) -> float:
            """Make a prediction on new text."""
            # Preprocess input
            inputs = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model.get_model()(**inputs)
                logits = outputs.logits
                prediction = torch.sigmoid(logits)[0].item()
            
            return prediction

    # Usage example
    def main():
        # Initialize trainer
        classifier = TextClassificationTrainer()
        
        # Train model
        classifier.train()
        
        # Make predictions
        test_text = "This movie was absolutely fantastic! The acting was superb."
        prediction = classifier.predict(test_text)
        print(f"Prediction (positive): {prediction:.2%}")

    if __name__ == "__main__":
        main()

Step-by-Step Explanation
----------------------

1. Model Setup
~~~~~~~~~~~~

We use a medium-sized model with optimizations:

- 4-bit quantization for memory efficiency
- LoRA for parameter-efficient fine-tuning
- Gradient checkpointing for larger batch sizes

2. Dataset Preparation
~~~~~~~~~~~~~~~~~~~

The dataset preparation pipeline:

1. Load IMDB dataset
2. Split into train/val/test
3. Preprocess and tokenize
4. Create dataloaders

3. Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

Key training parameters:

- Learning rate: 2e-4
- Batch size: 8
- Gradient accumulation: 4 steps
- Evaluation every 500 steps
- Checkpoints every 1000 steps

4. Training Process
~~~~~~~~~~~~~~~~

The training process includes:

- Automatic hardware optimization
- Progress tracking
- Regular evaluation
- Checkpoint saving

Making Predictions
----------------

Use the trained model for predictions:

.. code-block:: python

    classifier = TextClassificationTrainer()
    classifier.train()

    # Single prediction
    text = "This movie was fantastic!"
    prediction = classifier.predict(text)
    print(f"Positive probability: {prediction:.2%}")

    # Batch predictions
    texts = ["Great movie!", "Terrible acting", "Mixed feelings"]
    predictions = [classifier.predict(text) for text in texts]

Tips for Better Results
--------------------

1. Data Quality
~~~~~~~~~~~~~

- Clean your input texts
- Balance your dataset
- Use appropriate text length

2. Model Selection
~~~~~~~~~~~~~~~

- Start with smaller models
- Use LoRA for efficiency
- Enable quantization

3. Training Parameters
~~~~~~~~~~~~~~~~~~

- Adjust learning rate
- Increase epochs for better results
- Use gradient accumulation

4. Hardware Utilization
~~~~~~~~~~~~~~~~~~~~

- Enable GPU acceleration
- Use gradient checkpointing
- Monitor memory usage

Next Steps
---------

- Try different model architectures
- Experiment with LoRA parameters
- Add custom evaluation metrics
- Implement cross-validation
- Deploy your model

See Also
-------

- :doc:`custom_dataset` for using your own data
- :doc:`deployment` for model deployment
- :doc:`advanced_usage/index` for advanced features