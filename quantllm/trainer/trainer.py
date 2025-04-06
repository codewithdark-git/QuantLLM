import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm
from wandb import wandb, login
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
        self.wandb_token = self.wandb_config['API_KEY']
        
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
        if wandb.login(key=self.wandb_token, relogin=True):
            self.logger.log_info("Logged in to Weights & Biases")
        else:
            self.logger.log_error("Failed to log in to Weights & Biases")
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
        
    def train_step(self, batch, scaler):
        """Single training step."""
        try:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with modern autocast
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(**batch)
                loss = outputs.loss

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            if self.config.max_grad_norm is not None:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
            scaler.step(self.optimizer)
            scaler.update()
            
            self.optimizer.zero_grad()
            
            return loss.item()
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            raise

    def train(self):
        """Train the model."""
        try:
            self.logger.log_info("Starting training")
            
            # Disable model caching when using gradient checkpointing
            if hasattr(self.model.config, 'gradient_checkpointing') and self.model.config.gradient_checkpointing:
                self.model.config.use_cache = False
                self.logger.log_info("Disabled model caching due to gradient checkpointing")
                
            scaler = torch.cuda.amp.GradScaler()
            
            for epoch in range(self.config.num_epochs):
                self.model.train()
                total_loss = 0
                
                # Training loop
                with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch + 1}/{self.config.num_epochs}") as pbar:
                    for step, batch in enumerate(self.train_dataloader):
                        loss = self.train_step(batch, scaler)
                        total_loss += loss
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({'loss': f'{loss:.4f}'})
                        
                        if self.config.save_steps > 0 and (step + 1) % self.config.save_steps == 0:
                            self._save_checkpoint(epoch, step)
                            
                # Epoch end processing
                avg_loss = total_loss / len(self.train_dataloader)
                self.logger.log_info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")
                
                if self.config.save_epochs > 0 and (epoch + 1) % self.config.save_epochs == 0:
                    self._save_checkpoint(epoch)
                    
                if self.config.eval_epochs > 0 and (epoch + 1) % self.config.eval_epochs == 0:
                    self._evaluate()
                    
        except Exception as e:
            self.logger.log_error(f"Training error: {str(e)}")
            raise
                    
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