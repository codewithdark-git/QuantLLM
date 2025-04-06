from quantllm import QuantizedLLM

def main():
    # Initialize model
    model = QuantizedLLM(
        model_name="meta-llama/Llama-2-7b-hf",
        quantization="4bit",
        use_lora=True,
        push_to_hub=False  # Set to True if you want to push to Hub
    )
    
    # Load dataset
    print("Loading dataset...")
    model.load_dataset("imdb", split="train[:1000]")  # Using a small subset for demo
    
    # Fine-tune
    print("Starting fine-tuning...")
    model.finetune(
        epochs=1,
        batch_size=4,
        learning_rate=2e-4
    )
    
    # Save checkpoint
    print("Saving checkpoint...")
    model.save_checkpoint("checkpoints/demo_checkpoint")
    
    print("Done!")

if __name__ == "__main__":
    main() 