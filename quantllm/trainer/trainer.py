import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

# Conditionally import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
        device: Optional[Union[str, torch.device]] = None,
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
            device (str or torch.device, optional): Device to train on
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
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_config = wandb_config or {}
        
        # Handle device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.logger.log_info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.device == torch.device("cuda") else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Setup Weights & Biases if enabled
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
        try:
            if wandb.login(key=self.wandb_token, relogin=True):
                self.logger.log_info("Logged in to Weights & Biases")
                wandb.init(
                    project=self.wandb_config.get("project", "quantllm"),
                    name=self.wandb_config.get("name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                    config=self.config.to_dict()
                )
            else:
                self.logger.log_warning("Failed to log in to Weights & Biases. Continuing without wandb logging.")
                self.use_wandb = False
        except Exception as e:
            self.logger.log_warning(f"Error setting up Weights & Biases: {str(e)}. Continuing without wandb logging.")
            self.use_wandb = False
            
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch of data."""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
                
        # Forward pass
        outputs = self.model(**batch)
        return outputs.loss
        
    def train_step(self, batch, scaler):
        """Single training step."""
        try:
            # Convert batch to dictionary if it's a tuple/list
            if isinstance(batch, (tuple, list)):
                batch = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2] if len(batch) > 2 else None
                }
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Determine if we should use autocast based on device
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if self.config.max_grad_norm is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # CPU or MPS training - no autocast needed
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Standard backward pass
                loss.backward()
                
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            return loss.item()
            
        except Exception as e:
            self.logger.log_error(f"Error in training step: {str(e)}")
            raise

    def train(self):
        """Train the model."""
        try:
            # Log training start
            self.logger.log_training_start(
                model_name=self.model.config.name_or_path,
                dataset_name=self.train_dataloader.dataset.__class__.__name__,
                config=self.config.to_dict()
            )
            
            # Disable model caching when using gradient checkpointing
            if hasattr(self.model.config, 'gradient_checkpointing') and self.model.config.gradient_checkpointing:
                self.model.config.use_cache = False
                self.logger.log_info("Disabled model caching due to gradient checkpointing")
                
            scaler = torch.cuda.amp.GradScaler()
            
            for epoch in range(self.config.num_epochs):
                self.model.train()
                total_loss = 0
                
                # Log epoch start
                self.logger.log_epoch_start(epoch + 1, self.config.num_epochs)
                
                # Training loop
                with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch + 1}/{self.config.num_epochs}") as pbar:
                    for step, batch in enumerate(self.train_dataloader):
                        loss = self.train_step(batch, scaler)
                        total_loss += loss
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({'loss': f'{loss:.4f}'})
                        
                        # Log steps if configured
                        if step > 0 and self.config.logging_steps > 0 and step % self.config.logging_steps == 0:
                            avg_loss = total_loss / (step + 1)
                            self.logger.log_metrics({"loss": avg_loss}, step=step)
                        
                        # Save checkpoint if configured
                        if self.config.save_steps > 0 and (step + 1) % self.config.save_steps == 0:
                            metrics = {"loss": total_loss / (step + 1)}
                            self._save_checkpoint(epoch, step, metrics)
                            
                # Epoch end processing
                avg_loss = total_loss / len(self.train_dataloader)
                epoch_metrics = {"avg_loss": avg_loss}
                self.logger.log_epoch_complete(epoch + 1, epoch_metrics)
                
                # Run evaluation if configured
                if self.config.eval_epochs > 0 and (epoch + 1) % self.config.eval_epochs == 0:
                    eval_metrics = self._evaluate()
                    self.logger.log_evaluation_complete(eval_metrics)
                
                # Save epoch checkpoint if configured
                if self.config.save_epochs > 0 and (epoch + 1) % self.config.save_epochs == 0:
                    metrics = {"epoch": epoch + 1, "avg_loss": avg_loss}
                    self._save_checkpoint(epoch, None, metrics)
            
            # Log final training metrics
            final_metrics = {
                "final_loss": avg_loss,
                "total_epochs": self.config.num_epochs,
                "total_steps": self.global_step
            }
            self.logger.log_training_complete(final_metrics)
            
        except Exception as e:
            self.logger.log_error(f"Training error: {str(e)}")
            raise

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set."""
        if self.eval_dataloader is None:
            return {}
            
        self.logger.log_evaluation_start()
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        metrics = {"eval_loss": avg_loss}
        self.logger.log_evaluation_complete(metrics)
        return metrics

    def _save_checkpoint(self, epoch: int, step: Optional[int] = None, metrics: Optional[Dict[str, float]] = None):
        """Save a checkpoint."""
        if self.checkpoint_manager is None:
            return
            
        checkpoint_metrics = metrics or {}
        checkpoint_metrics.update({
            "epoch": epoch + 1,
            "step": step if step is not None else "end_of_epoch",
            "global_step": self.global_step
        })
        
        path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            tokenizer=None,  # We don't save tokenizer with checkpoints
            epoch=epoch,
            metrics=checkpoint_metrics
        )
        
        self.logger.log_checkpoint_save(path, checkpoint_metrics)

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