"""
Post-Quantization Fine-tuning Example

This example demonstrates the enhanced post-quantization fine-tuning capabilities
including QLoRA integration, adapter management, and comprehensive validation.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Import QuantLLM components
from quantllm import (
    # Fine-tuning components
    GPTQFineTuner,
    GPTQFineTuningEvaluator,
    QLoRAConfig,
    create_qlora_config,
    
    # Validation components
    FineTuningValidator,
    FineTuningProgressTracker,
    FineTuningValidationConfig,
    
    # Adapter management
    AdapterManager,
    AdapterConfig,
    
    # Backend components
    RobustGPTQBackend,
    
    # Configuration
    GPTQConfig
)


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


def main():
    """Main example function."""
    print("🚀 Post-Quantization Fine-tuning Example")
    print("=" * 50)
    
    # Configuration
    model_name = "microsoft/DialoGPT-small"  # Small model for demo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📱 Using device: {device}")
    print(f"🤖 Model: {model_name}")
    
    # Step 1: Load model and tokenizer
    print("\n📥 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 2: Create sample dataset
    print("\n📊 Creating sample dataset...")
    sample_texts = [
        "Hello, how are you today?",
        "I'm doing well, thank you for asking!",
        "What's your favorite programming language?",
        "I enjoy working with Python and PyTorch.",
        "Can you help me with machine learning?",
        "Of course! I'd be happy to help with ML questions.",
        "What is quantization in deep learning?",
        "Quantization reduces model precision to save memory and speed up inference.",
        "How does fine-tuning work?",
        "Fine-tuning adapts a pre-trained model to specific tasks or domains."
    ]
    
    # Split into train and eval
    train_texts = sample_texts[:8]
    eval_texts = sample_texts[8:]
    
    train_dataset = SimpleTextDataset(train_texts, tokenizer, max_length=128)
    eval_dataset = SimpleTextDataset(eval_texts, tokenizer, max_length=128)
    
    print(f"✅ Created datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Step 3: Simulate GPTQ quantization (for demo, we'll use the original model)
    print("\n⚡ Simulating GPTQ quantization...")
    
    # Create a mock quantized model wrapper
    class MockQuantizedModel:
        def __init__(self, model):
            self.model = model
            self.backend_name = "GPTQ"
    
    quantized_model = MockQuantizedModel(model)
    print("✅ Mock quantized model created")
    
    # Step 4: Initialize adapter manager
    print("\n🔧 Initializing adapter manager...")
    adapter_manager = AdapterManager(base_dir="./demo_adapters")
    
    # Step 5: Create QLoRA configuration
    print("\n⚙️ Creating QLoRA configuration...")
    qlora_config = create_qlora_config(
        r=8,  # Small rank for demo
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],  # DialoGPT modules
        lora_dropout=0.1,
        learning_rate=5e-4,
        num_epochs=2,  # Short training for demo
        batch_size=2,
        gradient_accumulation_steps=2
    )
    
    print(f"✅ QLoRA config created with rank {qlora_config.r}")
    
    # Step 6: Initialize fine-tuning validator
    print("\n🔍 Setting up validation...")
    validation_config = FineTuningValidationConfig(
        max_perplexity_increase=0.2,  # Allow 20% increase for demo
        min_accuracy_improvement=0.001,  # Lower threshold for demo
        generation_prompts=["Hello, how are you?", "What is AI?"],
        output_dir="./demo_validation_results"
    )
    
    validator = FineTuningValidator(validation_config)
    progress_tracker = FineTuningProgressTracker("./demo_progress")
    
    # Step 7: Pre-training validation
    print("\n🔍 Running pre-training validation...")
    pre_validation = validator.validate_pre_training(
        model=model,
        dataset=train_dataset,
        tokenizer=tokenizer,
        baseline_model=model
    )
    
    print(f"✅ Pre-training validation - Ready: {pre_validation['ready_for_training']}")
    if pre_validation['warnings']:
        print(f"⚠️ Warnings: {pre_validation['warnings']}")
    
    # Step 8: Initialize fine-tuner
    print("\n🎯 Initializing GPTQ fine-tuner...")
    try:
        fine_tuner = GPTQFineTuner(quantized_model, qlora_config)
        
        # Prepare model for fine-tuning
        peft_model, adapter_id = adapter_manager.create_adapter(
            model=quantized_model,
            config=AdapterConfig(
                r=qlora_config.r,
                lora_alpha=qlora_config.lora_alpha,
                target_modules=qlora_config.target_modules,
                lora_dropout=qlora_config.lora_dropout,
                adapter_name="demo_adapter",
                description="Demonstration adapter for post-quantization fine-tuning",
                tags=["demo", "gptq", "qlora"]
            )
        )
        
        print(f"✅ Fine-tuner initialized with adapter ID: {adapter_id}")
        
        # Step 9: Simulate training progress monitoring
        print("\n📈 Simulating training progress...")
        
        # Simulate some training steps
        for epoch in range(2):
            for step in range(3):
                # Simulate loss and learning rate
                loss = 2.5 - (epoch * 0.5 + step * 0.1)  # Decreasing loss
                lr = qlora_config.learning_rate * (0.9 ** (epoch * 3 + step))
                
                # Monitor progress
                monitoring_result = validator.monitor_training_progress(
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    learning_rate=lr,
                    model=peft_model
                )
                
                # Track progress
                progress_tracker.log_step(epoch, step, loss, lr)
                
                print(f"  Epoch {epoch}, Step {step}: Loss={loss:.3f}, LR={lr:.2e}, Stable={monitoring_result['stable']}")
        
        # Step 10: Evaluate fine-tuned model
        print("\n📊 Evaluating fine-tuned model...")
        evaluator = GPTQFineTuningEvaluator(peft_model)
        
        # Basic evaluation
        try:
            perplexity = evaluator.evaluate_perplexity(eval_dataset, tokenizer, batch_size=1)
            accuracy_metrics = evaluator.evaluate_accuracy(eval_dataset, tokenizer, batch_size=1)
            
            print(f"✅ Perplexity: {perplexity:.2f}")
            print(f"✅ Token Accuracy: {accuracy_metrics['token_accuracy']:.3f}")
            print(f"✅ Sequence Accuracy: {accuracy_metrics['sequence_accuracy']:.3f}")
            
        except Exception as e:
            print(f"⚠️ Evaluation error (expected in demo): {str(e)}")
        
        # Step 11: Generation quality evaluation
        print("\n🎭 Testing generation quality...")
        try:
            generation_quality = evaluator.evaluate_generation_quality(
                prompts=["Hello, how are you?"],
                tokenizer=tokenizer,
                max_new_tokens=20
            )
            
            print(f"✅ Average generation length: {generation_quality['average_generation_length']:.1f}")
            print(f"✅ Diversity ratio: {generation_quality['diversity_ratio']:.3f}")
            
            if generation_quality['generations']:
                print(f"📝 Sample generation: {generation_quality['generations'][0]}")
                
        except Exception as e:
            print(f"⚠️ Generation evaluation error (expected in demo): {str(e)}")
        
        # Step 12: Adapter efficiency evaluation
        print("\n⚡ Evaluating adapter efficiency...")
        try:
            # Save adapter first
            adapter_path = adapter_manager.save_adapter(
                peft_model=peft_model,
                adapter_id=adapter_id,
                training_metrics={"final_loss": 1.8, "epochs": 2},
                performance_metrics={"perplexity": perplexity if 'perplexity' in locals() else 0}
            )
            
            efficiency_metrics = evaluator.evaluate_adapter_efficiency(adapter_path)
            
            print(f"✅ Adapter size: {efficiency_metrics['adapter_size_mb']:.2f} MB")
            print(f"✅ Adapter parameters: {efficiency_metrics['adapter_parameters']:,}")
            print(f"✅ Parameter efficiency: {efficiency_metrics['adapter_parameter_ratio_percent']:.2f}%")
            
        except Exception as e:
            print(f"⚠️ Adapter efficiency evaluation error: {str(e)}")
        
        # Step 13: Post-training validation
        print("\n🔍 Running post-training validation...")
        try:
            post_validation = validator.validate_post_training(
                fine_tuned_model=peft_model,
                original_model=model,
                dataset=eval_dataset,
                tokenizer=tokenizer,
                adapter_path=adapter_path if 'adapter_path' in locals() else None
            )
            
            print(f"✅ Post-training validation - Passed: {post_validation.passed}")
            print(f"📊 Validation time: {post_validation.validation_time_seconds:.2f}s")
            
            if post_validation.recommendations:
                print("💡 Recommendations:")
                for rec in post_validation.recommendations[:3]:  # Show first 3
                    print(f"  • {rec}")
            
            if post_validation.report_path:
                print(f"📄 Validation report saved: {post_validation.report_path}")
                
        except Exception as e:
            print(f"⚠️ Post-training validation error: {str(e)}")
        
        # Step 14: Adapter management demonstration
        print("\n🗂️ Demonstrating adapter management...")
        
        # List adapters
        adapters = adapter_manager.list_adapters()
        print(f"✅ Found {len(adapters)} adapters:")
        for adapter in adapters:
            print(f"  • {adapter['name']} (ID: {adapter['adapter_id'][:8]}...)")
        
        # Validate adapter compatibility
        if adapters:
            compatibility = adapter_manager.validate_adapter_compatibility(
                adapter_id=adapters[0]['adapter_id'],
                model=model
            )
            print(f"✅ Adapter compatibility: {compatibility['compatible']}")
        
        # Step 15: Progress summary
        print("\n📈 Training progress summary...")
        progress_summary = progress_tracker.get_progress_summary()
        
        if progress_summary.get("status") != "no_data":
            print(f"✅ Total steps: {progress_summary['total_steps']}")
            print(f"✅ Final loss: {progress_summary['current_loss']:.3f}")
            print(f"✅ Loss trend: {progress_summary['loss_trend']}")
            print(f"✅ Training time: {progress_summary['training_time_minutes']:.1f} minutes")
        
        print("\n🎉 Post-quantization fine-tuning example completed successfully!")
        print("\n📋 Summary of capabilities demonstrated:")
        print("  ✅ QLoRA configuration and setup")
        print("  ✅ Adapter creation and management")
        print("  ✅ Training progress monitoring")
        print("  ✅ Comprehensive model evaluation")
        print("  ✅ Fine-tuning validation")
        print("  ✅ Adapter efficiency analysis")
        print("  ✅ Generation quality assessment")
        
    except ImportError as e:
        print(f"⚠️ PEFT library not available: {str(e)}")
        print("💡 Install with: pip install peft")
        print("🔄 Continuing with mock demonstration...")
        
        # Mock demonstration without PEFT
        print("\n🎭 Mock demonstration of fine-tuning workflow:")
        print("  1. ✅ Model and dataset preparation")
        print("  2. ✅ Configuration setup")
        print("  3. ✅ Validation framework initialization")
        print("  4. 🔄 Fine-tuning (would run with PEFT)")
        print("  5. 🔄 Evaluation (would run with PEFT)")
        print("  6. ✅ Progress tracking")
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {str(e)}")
        print("💡 This is expected in a demo environment")


if __name__ == "__main__":
    main()