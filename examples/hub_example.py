"""
QuantLLM v2.0 - Hub Integration Example

Demonstrates using QuantLLMHubManager for model lifecycle management.
Requires: pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git
"""

import os
from quantllm import turbo
from quantllm.hub import QuantLLMHubManager, create_hub_manager, is_hf_lifecycle_available


def basic_hub_usage():
    """Basic Hub integration example."""
    
    print("=== QuantLLM Hub Integration ===\n")
    print(f"hf_lifecycle available: {is_hf_lifecycle_available()}")
    
    if not is_hf_lifecycle_available():
        print("\nInstall hf_lifecycle for full functionality:")
        print("pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git")
        return
    
    # ============================================
    # INITIALIZE HUB MANAGER
    # ============================================
    manager = QuantLLMHubManager(
        repo_id="username/my-quantllm-model",
        local_dir="./outputs",
        checkpoint_dir="./checkpoints",
        hf_token=os.environ.get("HF_TOKEN"),  # Or pass directly
        auto_push=False,  # Set True to auto-push checkpoints
    )
    
    # ============================================
    # TRACK HYPERPARAMETERS
    # ============================================
    manager.track_hyperparameters({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "model": "microsoft/phi-2",
        "quantization": "4-bit",
    })
    
    # ============================================
    # LOAD MODEL
    # ============================================
    model = turbo("microsoft/phi-2", bits=4)
    
    # ============================================
    # TRAINING LOOP (simulated)
    # ============================================
    epochs = 3
    for epoch in range(epochs):
        # Simulate training
        train_loss = 1.0 / (epoch + 1)
        val_loss = 1.1 / (epoch + 1)
        
        # Log metrics
        manager.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch,
        }, step=epoch)
        
        # Save checkpoint
        manager.save_checkpoint(
            model=model.model,
            optimizer=None,  # Add your optimizer here
            epoch=epoch,
            metrics={"val_loss": val_loss},
            push=False,  # Override auto_push if needed
        )
        
        print(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # ============================================
    # SAVE FINAL MODEL
    # ============================================
    manager.save_final_model(
        model=model.model,
        format="safetensors",
        tokenizer=model.tokenizer,
    )
    
    # ============================================
    # PUSH TO HUB
    # ============================================
    # Uncomment to actually push:
    # manager.push(
    #     push_checkpoints=True,
    #     push_metadata=True,
    #     push_final_model=True,
    # )
    
    # ============================================
    # CLEANUP OLD CHECKPOINTS
    # ============================================
    # manager.cleanup_checkpoints(keep_last=3)
    
    print("\nDone! Uncomment push() to upload to HuggingFace Hub.")


def load_checkpoint_example():
    """Example of loading checkpoints."""
    
    manager = create_hub_manager(
        repo_id="username/my-model",
        hf_token=os.environ.get("HF_TOKEN"),
    )
    
    # List available checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"Available checkpoints: {len(checkpoints)}")
    
    # Load specific checkpoint
    # checkpoint_data = manager.load_checkpoint(
    #     name="checkpoint_epoch_5",
    #     model=model,
    #     optimizer=optimizer,
    # )
    
    # Load latest checkpoint
    # checkpoint_data = manager.load_latest_checkpoint(
    #     model=model,
    #     optimizer=optimizer,
    # )


def register_custom_model_example():
    """Example of registering a custom model."""
    
    manager = create_hub_manager(
        repo_id="username/my-custom-model",
        hf_token=os.environ.get("HF_TOKEN"),
    )
    
    model = turbo("microsoft/phi-2", bits=4)
    
    # Register custom model
    # manager.register_custom_model(
    #     model=model.model,
    #     config=model.config,
    #     model_name="My Custom QuantLLM Model",
    #     description="Fine-tuned and quantized model using QuantLLM",
    #     push_to_hub=True,
    #     commit_message="Initial upload",
    # )


if __name__ == "__main__":
    basic_hub_usage()