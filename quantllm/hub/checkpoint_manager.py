import os
import shutil
from typing import Optional, Dict, Any
from datetime import datetime
from ..trainer.logger import TrainingLogger

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "./checkpoints", save_total_limit: int = None):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir (str): Directory to store checkpoints
            save_total_limit (int, optional): Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_total_limit = save_total_limit
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger = TrainingLogger()
        
    def save_checkpoint(
        self,
        model,
        tokenizer,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Save model checkpoint.
        
        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            epoch (int): Current epoch
            metrics (Dict[str, Any], optional): Training metrics
            **kwargs: Additional arguments for save
        """
        try:
            # Create checkpoint directory
            checkpoint_name = f"checkpoint_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model
            model.save_pretrained(checkpoint_path, **kwargs)
            
            # Save tokenizer
            tokenizer.save_pretrained(checkpoint_path, **kwargs)
            
            # Save metrics if provided
            if metrics:
                metrics_path = os.path.join(checkpoint_path, "metrics.json")
                with open(metrics_path, 'w') as f:
                    import json
                    json.dump(metrics, f, indent=4)
            
            # Handle save_total_limit
            if self.save_total_limit is not None:
                checkpoints = self.list_checkpoints()
                if len(checkpoints) > self.save_total_limit:
                    # Sort by creation time and remove oldest
                    checkpoints.sort(key=lambda x: os.path.getctime(x))
                    for checkpoint in checkpoints[:-self.save_total_limit]:
                        self.delete_checkpoint(checkpoint)
                    
            self.logger.log_info(f"Checkpoint saved to {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            self.logger.log_error(f"Error saving checkpoint: {str(e)}")
            raise
            
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model_class,
        tokenizer_class,
        **kwargs
    ):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint
            model_class: Model class to load
            tokenizer_class: Tokenizer class to load
            **kwargs: Additional arguments for load
            
        Returns:
            Tuple: Loaded model and tokenizer
        """
        try:
            # Load model
            model = model_class.from_pretrained(checkpoint_path, **kwargs)
            
            # Load tokenizer
            tokenizer = tokenizer_class.from_pretrained(checkpoint_path, **kwargs)
            
            self.logger.log_info(f"Checkpoint loaded from {checkpoint_path}")
            return model, tokenizer
            
        except Exception as e:
            self.logger.log_error(f"Error loading checkpoint: {str(e)}")
            raise
            
    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = []
        for item in os.listdir(self.checkpoint_dir):
            if item.startswith("checkpoint_epoch_"):
                checkpoints.append(os.path.join(self.checkpoint_dir, item))
        return sorted(checkpoints)
        
    def delete_checkpoint(self, checkpoint_path: str):
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint
        """
        try:
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
                self.logger.log_info(f"Deleted checkpoint: {checkpoint_path}")
            else:
                self.logger.log_warning(f"Checkpoint not found: {checkpoint_path}")
                
        except Exception as e:
            self.logger.log_error(f"Error deleting checkpoint: {str(e)}")
            raise