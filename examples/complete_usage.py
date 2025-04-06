from quantllm import (
    Model,
    LoadDataset,
    DatasetPreprocessor,
    DatasetSplitter,
    FineTuningTrainer,
    ModelEvaluator,
    HubManager,
    CheckpointManager,
    DataLoader
)
import os
# Initialize logger
from quantllm.trainer import TrainingLogger
from quantllm.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)

def main():
    logger = TrainingLogger()
    
    try:
        # 1. Initialize hub manager first
        logger.log_info("Initializing hub and checkpoint managers")
        hub_manager = HubManager(
            model_id="your-username/llama-2-imdb",
            token=os.getenv("HF_TOKEN")
        )

        # 2. Model Configuration and Loading
        logger.log_info("Setting up model configuration")
        model_config = ModelConfig(
            model_name="meta-llama/Llama-3.2-3B",
            load_in_4bit=True,
            use_lora=True
        )

        model_loader = Model(model_config)
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()

        # 3. Dataset Configuration and Loading
        logger.log_info("Setting up dataset configuration")
        dataset_config = DatasetConfig(
            dataset_name_or_path="imdb",
            dataset_type="huggingface",
            text_column="text",
            label_column="label",
            max_length=512,
            train_size=0.8,
            val_size=0.1,
            test_size=0.1
        )

        # Load and prepare dataset
        logger.log_info("Loading and preparing dataset")
        dataset_loader = LoadDataset(logger)
        dataset = dataset_loader.load_hf_dataset(dataset_config)

        # Split dataset
        dataset_splitter = DatasetSplitter(logger)
        train_dataset, val_dataset, test_dataset = dataset_splitter.train_val_test_split(
            dataset,
            train_size=dataset_config.train_size,
            val_size=dataset_config.val_size,
            test_size=dataset_config.test_size
        )

        # 4. Dataset Preprocessing
        logger.log_info("Preprocessing datasets")
        preprocessor = DatasetPreprocessor(tokenizer, logger)
        train_tokenizer, val_tokenizer, test_tokenizer = preprocessor.tokenize_dataset(
            train_dataset, val_dataset, test_dataset,
            max_length=dataset_config.max_length,
            text_column=dataset_config.text_column,
            label_column=dataset_config.label_column
        )

        # Create data loaders using QuantLLMDataLoader
        logger.log_info("Creating data loaders")
        dataloaders = DataLoader.from_datasets(
            train_dataset=train_tokenizer,
            val_dataset=val_tokenizer,
            test_dataset=test_tokenizer,
            batch_size=4,
            num_workers=4
        )

        # 5. Training Configuration
        logger.log_info("Setting up training configuration")
        training_config = TrainingConfig(
            learning_rate=2e-4,
            num_epochs=3,
            batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=50,
            eval_steps=200,
            save_steps=500,
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )

        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(
            output_dir="./checkpoints",
            save_total_limit=3
        )

        # 6. Initialize Trainer
        logger.log_info("Initializing trainer")
        trainer = FineTuningTrainer(
            model=model,
            training_config=training_config,
            train_dataloader=dataloaders["train"],
            eval_dataloader=dataloaders["val"],
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            hub_manager=hub_manager,
            use_wandb=True,
            wandb_config={
                "project": "quantllm-imdb",
                "name": "llama-2-imdb-finetuning"
            }
        )

        # 7. Train the model
        logger.log_info("Starting training")
        trainer.train()

        # 8. Evaluate on test set
        logger.log_info("Evaluating on test set")
        evaluator = ModelEvaluator(
            model=model,
            eval_dataloader=dataloaders["test"],
            metrics=[
                lambda preds, labels, _: (preds.argmax(dim=-1) == labels).float().mean().item()  # Accuracy
            ],
            logger=logger
        )

        test_metrics = evaluator.evaluate()
        logger.log_info(f"Test metrics: {test_metrics}")

        # 9. Save final model
        logger.log_info("Saving final model")
        trainer.save_model("./final_model")

        # 10. Push to Hub if logged in
        if hub_manager.is_logged_in():
            logger.log_info("Pushing model to Hugging Face Hub")
            hub_manager.push_model(
                model,
                commit_message=f"Final model with test accuracy: {test_metrics.get('accuracy', 0):.4f}"
            )
        else:
            logger.log_warning("Not logged in to Hugging Face Hub. Skipping model push.")

        logger.log_info("Training and evaluation completed successfully!")

    except Exception as e:
        logger.log_error(f"Error in training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 