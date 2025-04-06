from argparse import Namespace
from ..models import ModelLoader, QuantConfig
from ..data import DatasetLoader
from ..training import FineTuningTrainer, ModelEvaluator
from ..config import ModelConfig, TrainingConfig, DatasetConfig
from ..runtime import DeviceManager
from ..utils.monitoring import TrainingLogger

def train(args: Namespace):
    """Execute model training command."""
    logger = TrainingLogger()
    device_manager = DeviceManager()
    
    try:
        # Load configurations
        model_config = ModelConfig(
            model_name=args.model,
            load_in_4bit=args.quantization == "4bit",
            load_in_8bit=args.quantization == "8bit",
            use_lora=args.use_lora
        )
        
        dataset_config = DatasetConfig(dataset_name=args.dataset)
        training_config = TrainingConfig(output_dir=args.output_dir)
        
        # Initialize components
        model_loader = ModelLoader(model_config)
        dataset_loader = DatasetLoader(dataset_config)
        
        # Load model and dataset
        model, tokenizer = model_loader.load()
        dataset = dataset_loader.load()
        
        # Initialize trainer
        trainer = FineTuningTrainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=training_config,
            device_manager=device_manager
        )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def evaluate(args: Namespace):
    """Execute model evaluation command."""
    logger = TrainingLogger()
    device_manager = DeviceManager()
    
    try:
        # Load model and dataset
        model_config = ModelConfig(model_name=args.model)
        dataset_config = DatasetConfig(dataset_name=args.dataset)
        
        model_loader = ModelLoader(model_config)
        dataset_loader = DatasetLoader(dataset_config)
        
        model, tokenizer = model_loader.load()
        dataset = dataset_loader.load()
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            device_manager=device_manager
        )
        
        # Run evaluation
        results = evaluator.evaluate(dataset)
        
        # Save results if output file specified
        if args.output_file:
            evaluator.save_results(results, args.output_file)
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def quantize(args: Namespace):
    """Execute model quantization command."""
    logger = TrainingLogger()
    
    try:
        # Configure quantization
        model_config = ModelConfig(
            model_name=args.model,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8
        )
        
        # Load and quantize model
        model_loader = ModelLoader(model_config)
        model, tokenizer = model_loader.load()
        
        # Save quantized model
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
    except Exception as e:
        logger.error(f"Quantization failed: {str(e)}")
        raise

def serve(args: Namespace):
    """Execute model serving command."""
    logger = TrainingLogger()
    device_manager = DeviceManager()
    
    try:
        from ..serving import ModelServer
        
        # Load model
        model_config = ModelConfig(model_name=args.model)
        model_loader = ModelLoader(model_config)
        model, tokenizer = model_loader.load()
        
        # Initialize and start server
        server = ModelServer(
            model=model,
            tokenizer=tokenizer,
            device_manager=device_manager,
            host=args.host,
            port=args.port
        )
        
        server.start()
        
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise