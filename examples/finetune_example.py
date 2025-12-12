"""
QuantLLM v2.0 - Fine-tuning Example

Demonstrates one-line fine-tuning with automatic configuration.
"""

from quantllm import turbo
from quantllm.core import auto_configure_training, LoRAAutoConfig


def main():
    # ============================================
    # LOAD MODEL
    # ============================================
    print("Loading model...")
    model = turbo("microsoft/phi-2", bits=4)
    
    # ============================================
    # SIMPLE FINE-TUNING (One Line!)
    # ============================================
    print("\n--- Simple Fine-tuning ---")
    
    # Just provide your data file - everything else is automatic
    # Supports: .json, .jsonl, .csv, or HuggingFace dataset names
    
    # Example data format (instruction/output):
    sample_data = [
        {"instruction": "What is Python?", "output": "Python is a programming language."},
        {"instruction": "What is AI?", "output": "AI stands for Artificial Intelligence."},
        # Add more examples...
    ]
    
    # Uncomment to actually train:
    # model.finetune(sample_data, epochs=3)
    
    # Or from file:
    # model.finetune("my_data.json", epochs=3)
    
    # ============================================
    # ADVANCED: Custom Configuration
    # ============================================
    print("\n--- Advanced Configuration ---")
    
    # Get auto-detected LoRA config for this model
    lora_config = LoRAAutoConfig.get_config(model.model)
    print(f"Auto-detected LoRA config:")
    print(f"  - Rank (r): {lora_config['r']}")
    print(f"  - Alpha: {lora_config['lora_alpha']}")
    print(f"  - Target modules: {lora_config['target_modules']}")
    
    # Fine-tune with custom settings
    # model.finetune(
    #     sample_data,
    #     epochs=5,
    #     learning_rate=1e-4,
    #     lora_r=32,
    #     lora_alpha=64,
    #     batch_size=4,
    # )
    
    # ============================================
    # SAVE FINE-TUNED MODEL
    # ============================================
    print("\n--- Save Model ---")
    
    # After fine-tuning, export:
    # model.export("safetensors", "./fine-tuned-model/")
    # model.export("gguf", "fine-tuned-model-q4.gguf")
    
    print("Done! Uncomment the training lines to actually fine-tune.")


if __name__ == "__main__":
    main()
