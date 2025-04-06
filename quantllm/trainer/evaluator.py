import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ..trainer.logger import TrainingLogger

class ModelEvaluator:
    def __init__(
        self,
        model: nn.Module,
        eval_dataloader: DataLoader,
        metrics: Optional[List[Callable]] = None,
        logger: Optional[TrainingLogger] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model (nn.Module): The model to evaluate
            eval_dataloader (DataLoader): Evaluation data loader
            metrics (List[Callable], optional): List of metric functions
            logger (TrainingLogger, optional): Logger instance
            device (str, optional): Device to evaluate on
        """
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.metrics = metrics or []
        self.logger = logger or TrainingLogger()
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch of data."""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
                
        # Forward pass
        outputs = self.model(**batch)
        return outputs.loss
        
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics for a batch."""
        metrics_dict = {}
        
        for metric_fn in self.metrics:
            try:
                metric_value = metric_fn(predictions, labels, batch)
                metrics_dict[metric_fn.__name__] = metric_value
            except Exception as e:
                self.logger.log_warning(f"Failed to compute metric {metric_fn.__name__}: {str(e)}")
                
        return metrics_dict
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the evaluation dataset."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Compute loss
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions and labels
                outputs = self.model(**batch)
                predictions = outputs.logits
                labels = batch.get("labels")
                
                # Compute metrics if available
                if labels is not None and self.metrics:
                    batch_metrics = self._compute_metrics(predictions, labels, batch)
                    for metric_name, metric_value in batch_metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(metric_value)
                        
        # Compute average metrics
        results = {"eval_loss": total_loss / num_batches}
        for metric_name, metric_values in all_metrics.items():
            results[metric_name] = np.mean(metric_values)
            
        return results
        
    def evaluate_on_specific_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate the model on a specific batch of data."""
        self.model.eval()
        
        with torch.no_grad():
            # Compute loss
            loss = self._compute_loss(batch)
            
            # Get predictions and labels
            outputs = self.model(**batch)
            predictions = outputs.logits
            labels = batch.get("labels")
            
            # Compute metrics if available
            results = {"eval_loss": loss.item()}
            if labels is not None and self.metrics:
                batch_metrics = self._compute_metrics(predictions, labels, batch)
                results.update(batch_metrics)
                
        return results 