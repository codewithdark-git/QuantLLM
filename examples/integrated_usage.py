from quantllm import (
    ModelLoader,
    LoraConfigManager,
    FineTuningTrainer,
    DatasetLoader,
    DatasetPreprocessor,
    DatasetSplitter,
    HubManager,
    CheckpointManager
)

def main():
    # Initialize logger
    from quantllm.finetune import TrainingLogger
    logger = TrainingLogger()
    
    try:
        # Load dataset
        logger.log_info("Loading and preparing dataset")
        dataset_loader = DatasetLoader(logger)
        dataset = dataset_loader.load_hf_dataset("imdb", split="train")
        
        # Split dataset
        dataset_splitter = DatasetSplitter(logger)
        train_dataset, val_dataset, test_dataset = dataset_splitter.train_val_test_split(
            dataset,
            val_size=0.1,
            test_size=0.1
        )
        
        # Load model
        logger.log_info("Loading model")
        model_loader = ModelLoader(
            model_name="meta-llama/Llama-2-7b-hf",
            quantization="4bit",
            use_lora=True
        )
        model = model_loader.get_model()
        tokenizer = model_loader.get_tokenizer()
        
        # Preprocess datasets
        logger.log_info("Preprocessing datasets")
        preprocessor = DatasetPreprocessor(tokenizer, logger)
        train_dataset = preprocessor.tokenize_dataset(train_dataset)
        val_dataset = preprocessor.tokenize_dataset(val_dataset)
        test_dataset = preprocessor.tokenize_dataset(test_dataset)
        
        # Initialize trainer
        logger.log_info("Initializing trainer")
        trainer = FineTuningTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir="./results",
            logging_dir="./logs"
        )
        
        # Train
        logger.log_info("Starting training")
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            num_train_epochs=3,
            per_device_train_batch_size=4
        )
        
        # Save checkpoint
        logger.log_info("Saving checkpoint")
        checkpoint_manager = CheckpointManager()
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            epoch=3
        )
        
        # Push to Hub
        logger.log_info("Pushing to HuggingFace Hub")
        hub_manager = HubManager(
            model_id="your-username/llama-2-4bit",
            token="your-hf-token"
        )
        hub_manager.push_model(model, tokenizer)
        hub_manager.push_checkpoint(checkpoint_path)
        
        logger.log_info("Training completed successfully!")
        
    except Exception as e:
        logger.log_error(f"Error in training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 