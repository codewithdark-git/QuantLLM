import logging
import os
from datetime import datetime
from typing import Dict, Any
import json

class TrainingLogger:
    def __init__(self, log_dir: str = "./logs"):
        """
        Initialize the training logger.
        
        Args:
            log_dir (str): Directory to store logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_info(self, message: str):
        """Log an info message"""
        self.logger.info(message)
        
    def log_warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)
        
    def log_error(self, message: str):
        """Log an error message"""
        self.logger.error(message)
        
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics"""
        # Save metrics to JSON file
        metrics_file = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Log metrics
        self.logger.info("Metrics:")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value}")
            
    def log_config(self, config: Dict[str, Any]):
        """Log configuration"""
        config_file = os.path.join(self.log_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
            
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}") 