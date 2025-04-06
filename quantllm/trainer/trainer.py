import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
from datetime import datetime
import os
from ..config.training_config import TrainingConfig
from ..trainer.logger import TrainingLogger
from ..hub.checkpoint_manager import CheckpointManager
from ..hub.hub_manager import HubManager

class FineTuningTrainer:
    def __init__(
        self,
        model: nn.Module,
        training_config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        logger: Optional[TrainingLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        hub_manager: Optional[HubManager] = None,
        device: Optional[str] = None,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the trainer with PyTorch-based training loop.
        
        Args:
            model (nn.Module): The model to train
            training_config (TrainingConfig): Training configuration
            train_dataloader (DataLoader): Training data loader
            eval_dataloader (DataLoader, optional): Evaluation data loader
            logger (TrainingLogger, optional): Logger instance
            checkpoint_manager (CheckpointManager, optional): Checkpoint manager
            hub_manager (HubManager, optional): Hub manager for model pushing
            device (str, optional): Device to train on
            use_wandb (bool): Whether to use Weights & Biases
            wandb_config (Dict[str, Any], optional): Weights & Biases configuration
        """
        self.model = model
        self.config = training_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.logger = logger or TrainingLogger()
        self.checkpoint_manager = checkpoint_manager
        self.hub_manager = hub_manager
        self.use_wandb = use_wandb
        self.wandb_config = wandb_config or {}
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.device == "cuda" else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Setup Weights & Biases
        if self.use_wandb:
            self._setup_wandb()
            
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        # Get parameters to optimize (excluding frozen parameters)
        params_to_optimize = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        
        # Create optimizer
        if self.config.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                params_to_optimize,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
            
        return optimizer
        
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        if self.config.scheduler.lower() == "linear":
            def lr_lambda(current_step: int) -> float:
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))
                return max(
                    0.0,
                    float(self.config.num_epochs * len(self.train_dataloader) - current_step) /
                    float(max(1, self.config.num_epochs * len(self.train_dataloader) - self.config.warmup_steps))
                )
                
            scheduler = LambdaLR(self.optimizer, lr_lambda)
        elif self.config.scheduler.lower() == "plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=3,
                verbose=True
            )
        else:
            scheduler = None
            
        return scheduler
        
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        if not wandb.api.api_key:
            self.logger.log_warning("Weights & Biases API key not found. Disabling W&B logging.")
            self.use_wandb = False
            return
            
        wandb.init(
            project=self.wandb_config.get("project", "quantllm"),
            name=self.wandb_config.get("name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config=self.config.to_dict()
        )
        
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch of data."""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
                
        # Forward pass
        outputs = self.model(**batch)
        return outputs.loss
        
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step."""
        self.model.train()
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss = self._compute_loss(batch)
                
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Optimizer step with gradient scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self._compute_loss(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
        return loss.item()
        
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set."""
        if self.eval_dataloader is None:
            return {}
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        return {"eval_loss": avg_loss}
        
    def train(self):
        """Train the model."""
        self.logger.log_info("Starting training")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.logger.log_info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training loop
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(self.train_dataloader, desc="Training")
            for batch in progress_bar:
                # Training step
                loss = self._train_step(batch)
                total_loss += loss
                num_batches += 1
                
                # Update learning rate
                if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
                    
                # Log metrics
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    metrics = {
                        "train_loss": avg_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                        "step": self.global_step
                    }
                    
                    self.logger.log_metrics(metrics)
                    if self.use_wandb:
                        wandb.log(metrics)
                        
                # Evaluation and checkpointing
                if self.eval_dataloader is not None and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self._evaluate()
                    self.logger.log_metrics(eval_metrics)
                    if self.use_wandb:
                        wandb.log(eval_metrics)
                        
                    # Update learning rate scheduler if using ReduceLROnPlateau
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(eval_metrics["eval_loss"])
                        
                    # Early stopping and checkpointing
                    if eval_metrics["eval_loss"] < self.best_metric - self.config.early_stopping_threshold:
                        self.best_metric = eval_metrics["eval_loss"]
                        self.patience_counter = 0
                        
                        # Save checkpoint locally
                        if self.checkpoint_manager is not None:
                            self.checkpoint_manager.save_checkpoint(
                                self.model,
                                self.optimizer,
                                self.scheduler,
                                self.global_step,
                                self.epoch,
                                eval_metrics
                            )
                            
                        # Push to hub if configured
                        if self.hub_manager is not None and self.hub_manager.is_logged_in():
                            try:
                                self.hub_manager.push_model(
                                    self.model,
                                    commit_message=f"Checkpoint at step {self.global_step} with eval_loss {eval_metrics['eval_loss']:.4f}"
                                )
                                self.logger.log_info("Model pushed to hub successfully")
                            except Exception as e:
                                self.logger.log_error(f"Failed to push model to hub: {str(e)}")
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config.early_stopping_patience:
                            self.logger.log_info("Early stopping triggered")
                            return
                            
                self.global_step += 1
                
            # End of epoch
            avg_loss = total_loss / num_batches
            self.logger.log_info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
        self.logger.log_info("Training completed")
        if self.use_wandb:
            wandb.finish()
            
    def save_model(self, output_dir: Union[str, Path]):
        """Save the model and training state."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / "model"
        self.model.save_pretrained(model_path)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric
        }
        
        torch.save(training_state, output_dir / "training_state.pt")
        
    def load_model(self, input_dir: Union[str, Path]):
        """Load the model and training state."""
        input_dir = Path(input_dir)
        
        # Load model
        model_path = input_dir / "model"
        self.model = self.model.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Load training state
        training_state = torch.load(input_dir / "training_state.pt")
        self.global_step = training_state["global_step"]
        self.epoch = training_state["epoch"]
        self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
        if self.scheduler and training_state["scheduler_state_dict"]:
            self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
        self.best_metric = training_state["best_metric"] 