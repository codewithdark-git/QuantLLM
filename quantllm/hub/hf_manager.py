"""
HFManager Integration for QuantLLM v2.0

Enhanced hub integration using hf_lifecycle for comprehensive
model lifecycle management on HuggingFace Hub.

Install: pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git
"""

import os
import importlib.util
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

# Try to import hf_lifecycle with multiple approaches
HF_LIFECYCLE_AVAILABLE = False
BaseHFManager = None

def _try_import_hf_lifecycle():
    """Try multiple approaches to import hf_lifecycle."""
    global HF_LIFECYCLE_AVAILABLE, BaseHFManager
    
    # Approach 1: Direct import
    try:
        from hf_lifecycle import HFManager as _HFManager
        BaseHFManager = _HFManager
        HF_LIFECYCLE_AVAILABLE = True
        return True
    except ImportError:
        pass
    
    # Approach 2: Check if module spec exists
    try:
        spec = importlib.util.find_spec("hf_lifecycle")
        if spec is not None:
            hf_lifecycle = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hf_lifecycle)
            BaseHFManager = hf_lifecycle.HFManager
            HF_LIFECYCLE_AVAILABLE = True
            return True
    except Exception:
        pass
    
    return False

# Run import check
_try_import_hf_lifecycle()

from ..utils.logger import logger


def is_hf_lifecycle_available() -> bool:
    """Check if hf_lifecycle is installed."""
    # Re-check in case it was installed after initial import
    if not HF_LIFECYCLE_AVAILABLE:
        _try_import_hf_lifecycle()
    return HF_LIFECYCLE_AVAILABLE


class QuantLLMHubManager:
    """
    Enhanced Hub Manager for QuantLLM using hf_lifecycle.
    
    Provides comprehensive model lifecycle management:
    - Checkpoint saving/loading with auto-push
    - Hyperparameter tracking
    - Metric logging
    - Final model export
    
    Example:
        >>> manager = QuantLLMHubManager(
        ...     repo_id="username/my-model",
        ...     local_dir="./outputs",
        ...     hf_token="your_token"
        ... )
        >>> 
        >>> # Track hyperparameters
        >>> manager.track_hyperparameters({"lr": 0.001, "epochs": 10})
        >>> 
        >>> # Log metrics during training
        >>> manager.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)
        >>> 
        >>> # Save checkpoint
        >>> manager.save_checkpoint(model, optimizer, epoch=1)
        >>> 
        >>> # Push everything to Hub
        >>> manager.push(push_checkpoints=True, push_final_model=True)
    """
    
    def __init__(
        self,
        repo_id: str,
        local_dir: str = "./outputs",
        checkpoint_dir: str = "./checkpoints",
        hf_token: Optional[str] = None,
        auto_push: bool = False,
    ):
        """
        Initialize the QuantLLM Hub Manager.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., "username/model-name")
            local_dir: Local directory for outputs
            checkpoint_dir: Directory for checkpoints
            hf_token: HuggingFace API token
            auto_push: Automatically push checkpoints when saved
        """
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.checkpoint_dir = checkpoint_dir
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.auto_push = auto_push
        
        # Initialize hf_lifecycle manager if available
        if HF_LIFECYCLE_AVAILABLE:
            self._manager = BaseHFManager(
                repo_id=repo_id,
                local_dir=local_dir,
                checkpoint_dir=checkpoint_dir,
                hf_token=hf_token,
                auto_push=auto_push,
            )
            logger.log_info(f"Initialized HFManager for repo: {repo_id}")
        else:
            self._manager = None
            logger.log_warning(
                "hf_lifecycle not installed. Install with: "
                "pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git"
            )
        
        # Create directories
        os.makedirs(local_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _require_manager(self):
        """Raise error if hf_lifecycle not available."""
        if self._manager is None:
            raise ImportError(
                "hf_lifecycle is required for this feature. "
                "Install with: pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git"
            )
    
    def track_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Track hyperparameters for the run.
        
        Args:
            params: Dictionary of hyperparameters
        """
        self._require_manager()
        self._manager.track_hyperparameters(params)
        logger.log_info(f"Tracked {len(params)} hyperparameters")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step or epoch number
        """
        self._require_manager()
        self._manager.log_metrics(metrics, step=step)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Any] = None,
        push: Optional[bool] = None,
        scheduler: Optional[Any] = None,
    ) -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optional optimizer state
            epoch: Current epoch number
            metrics: Optional metrics dictionary
            config: Optional config object
            push: Override auto_push for this checkpoint
            scheduler: Optional learning rate scheduler
            
        Returns:
            Path to saved checkpoint
        """
        self._require_manager()
        
        return self._manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
            push=push if push is not None else self.auto_push,
        )
    
    def load_checkpoint(
        self,
        name: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load a specific checkpoint.
        
        Args:
            name: Checkpoint name to load
            model: Model to load state into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            map_location: Device mapping for loading
            
        Returns:
            Dictionary with loaded checkpoint data
        """
        self._require_manager()
        
        return self._manager.load_checkpoint(
            name=name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
        )
    
    def load_latest_checkpoint(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load the most recent checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            map_location: Device mapping for loading
            
        Returns:
            Dictionary with loaded checkpoint data
        """
        self._require_manager()
        
        return self._manager.load_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
        )
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint info dictionaries
        """
        self._require_manager()
        return self._manager.list_checkpoints()
    
    def save_final_model(
        self,
        model: nn.Module,
        format: str = "safetensors",
        config: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """
        Save the final trained model.
        
        Args:
            model: Model to save
            format: Save format ("safetensors" or "pt")
            config: Optional config to save
            tokenizer: Optional tokenizer to save
            
        Returns:
            Path to saved model
        """
        self._require_manager()
        
        result = self._manager.save_final_model(
            model=model,
            format=format,
            config=config,
        )
        
        # Also save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(self.local_dir)
        
        return result
    
    def push(
        self,
        push_checkpoints: bool = True,
        push_metadata: bool = True,
        push_final_model: bool = True,
    ) -> None:
        """
        Push everything to HuggingFace Hub.
        
        Args:
            push_checkpoints: Whether to push checkpoints
            push_metadata: Whether to push metadata/metrics
            push_final_model: Whether to push final model
        """
        self._require_manager()
        
        self._manager.push(
            push_checkpoints=push_checkpoints,
            push_metadata=push_metadata,
            push_final_model=push_final_model,
        )
        logger.log_success(f"Pushed to Hub: {self.repo_id}")
    
    def register_custom_model(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[Any] = None,
        repo_id: Optional[str] = None,
        model_name: Optional[str] = None,
        description: str = "",
        push_to_hub: bool = False,
        commit_message: str = "Upload custom model",
    ) -> None:
        """
        Register a custom model on HuggingFace Hub.
        
        Args:
            model: Model to register
            config: Model configuration
            repo_id: Override repo ID
            model_name: Display name for the model
            description: Model description
            push_to_hub: Whether to push immediately
            commit_message: Commit message for push
        """
        self._require_manager()
        
        self._manager.register_custom_model(
            model=model,
            config=config,
            repo_id=repo_id,
            model_name=model_name,
            description=description,
            push_to_hub=push_to_hub,
            commit_message=commit_message,
        )
    
    def cleanup_checkpoints(
        self,
        keep_last: int = 3,
    ) -> None:
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        self._require_manager()
        self._manager.cleanup_checkpoints()
        logger.log_info(f"Cleaned up checkpoints, keeping last {keep_last}")


# Convenience function
def create_hub_manager(
    repo_id: str,
    hf_token: Optional[str] = None,
    auto_push: bool = False,
    **kwargs,
) -> QuantLLMHubManager:
    """
    Create a QuantLLM Hub Manager.
    
    Args:
        repo_id: HuggingFace repo ID
        hf_token: HuggingFace token (or set HF_TOKEN env var)
        auto_push: Auto-push checkpoints
        **kwargs: Additional arguments
        
    Returns:
        Configured QuantLLMHubManager
    """
    return QuantLLMHubManager(
        repo_id=repo_id,
        hf_token=hf_token,
        auto_push=auto_push,
        **kwargs,
    )
