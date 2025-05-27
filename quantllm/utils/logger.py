"""Global logger for QuantLLM project."""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import datetime
from enum import Enum

class LogLevel(Enum):
    INFO = "\033[94m"  # Blue
    SUCCESS = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    RESET = "\033[0m"  # Reset color

class GlobalLogger:
    """Global logger with enhanced functionality for QuantLLM."""
    
    _instance = None
    _welcome_shown = False  # Class-level flag to track if welcome message has been shown
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalLogger, cls).__new__(cls)
            cls._instance._initialize_logger()
            if not cls._welcome_shown:
                cls._instance.log_welcome_message()
                cls._welcome_shown = True
        return cls._instance
    
    def _initialize_logger(self):
        """Initialize the logger with both file and console handlers."""
        self.logger = logging.getLogger('QuantLLM')
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_dir / f"quantllm_{timestamp}.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        self.memory_logs = []
        self.start_time = datetime.datetime.now()

    def log_welcome_message(self):
        """Display QuantLLM welcome message with ASCII art."""
        from importlib.metadata import version
        try:
            __version__ = version("quantllm")
        except:
            __version__ = "1.1.0"
        logo = (
            f"{LogLevel.INFO.value}"
            "╔══════════════════════════════════════════════════════════════════════════════════╗\n"
            "║                                                                                  ║\n"
            "║   ███████╗ ██╗   ██╗  █████╗  ███╗   ██╗ ████████╗ ██╗     ██╗     ███╗   ███╗║\n"
            "║   ██╔═══██╗██║   ██║ ██╔══██╗ ████╗  ██║ ╚══██╔══╝ ██║     ██║     ████╗ ████║║\n"
            "║   ██║   ██║██║   ██║ ███████║ ██╔██╗ ██║    ██║    ██║     ██║     ██╔████╔██║║\n"
            "║   ██║▄▄ ██║██║   ██║ ██╔══██║ ██║╚██╗██║    ██║    ██║     ██║     ██║╚██╔╝██║║\n"
            "║   ╚██████╔╝╚██████╔╝ ██║  ██║ ██║ ╚████║    ██║    ███████╗███████╗██║ ╚═╝ ██║║\n"
            "║    ╚══▀▀═╝  ╚═════╝  ╚═╝  ╚═╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝╚══════╝╚═╝     ╚═╝║\n"
            "║                                                                                  ║\n"
            "╚══════════════════════════════════════════════════════════════════════════════════╝\n"
            f"{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}╭────────────────── Welcome to QuantLLM v{__version__} ───────────────────╮{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}│                                                                         │{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}│  🎯 State-of-the-Art Model Quantization & Efficient Training           │{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}│  🚀 Optimized for Production Deployment                                │{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}│  💻 Supports CUDA, CPU, and Apple Silicon                              │{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}│                                                                         │{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}╰─────────────────────────────────────────────────────────────────────────╯{LogLevel.RESET.value}\n\n"
        )
        print(logo)

    def _format_message(self, level: LogLevel, message: str) -> str:
        """Format log message with timestamp and color."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{level.value}[{timestamp}] {level.name}: {message}{LogLevel.RESET.value}"

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for logging."""
        return ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                         for k, v in metrics.items()])

    def log_info(self, message: str):
        """Log info message."""
        print(self._format_message(LogLevel.INFO, message))
        self.logger.info(message)

    def log_success(self, message: str):
        """Log success message."""
        print(self._format_message(LogLevel.SUCCESS, message))
        self.logger.info(message)

    def log_warning(self, message: str):
        """Log warning message."""
        print(self._format_message(LogLevel.WARNING, message))
        self.logger.warning(message)

    def log_error(self, message: str):
        """Log error message."""
        print(self._format_message(LogLevel.ERROR, message))
        self.logger.error(message)

    def log_debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def log_critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)

    def log_memory(self, operation: str, memory_used: Optional[float] = None, device: Optional[str] = None):
        """Log memory usage information."""
        device_str = f" on {device}" if device else ""
        message = f"Memory usage{device_str} after {operation}"
        if memory_used is not None:
            message += f": {memory_used:.2f} GB"
        self.logger.info(message)
        self.memory_logs.append({
            'timestamp': datetime.datetime.now(),
            'operation': operation,
            'memory_used': memory_used,
            'device': device
        })

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log training metrics."""
        step_str = f" (Step {step})" if step is not None else ""
        message = f"Metrics{step_str}: {self._format_metrics(metrics)}"
        print(self._format_message(LogLevel.INFO, message))
        self.logger.info(message)

    def log_training_start(self, model_name: str, dataset_name: str, config: Dict[str, Any]):
        """Log training start with configuration details."""
        self.start_time = datetime.datetime.now()
        self.log_success(f"Starting training for model: {model_name}")
        self.log_info(f"Dataset: {dataset_name}")
        self.log_info("Configuration:")
        for key, value in config.items():
            self.log_info(f"  {key}: {value}")

    def log_training_complete(self, metrics: Dict[str, Any]):
        """Log training completion with final metrics."""
        duration = datetime.datetime.now() - self.start_time
        self.log_success("Training completed successfully!")
        self.log_info(f"Total training time: {duration}")
        self.log_info("Final metrics:")
        for key, value in metrics.items():
            self.log_info(f"  {key}: {value}")

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.log_info(f"Starting epoch {epoch}/{total_epochs}")

    def log_epoch_complete(self, epoch: int, metrics: Dict[str, Any]):
        """Log epoch completion with metrics."""
        self.log_success(f"Completed epoch {epoch}")
        self.log_metrics(metrics)

    def log_evaluation_start(self, split: str = "validation"):
        """Log evaluation start."""
        self.log_info(f"Starting evaluation on {split} set")

    def log_evaluation_complete(self, metrics: Dict[str, Any], split: str = "validation"):
        """Log evaluation completion with metrics."""
        self.log_success(f"Completed evaluation on {split} set")
        self.log_metrics(metrics)

    def log_checkpoint_save(self, path: str, metrics: Dict[str, Any]):
        """Log checkpoint saving."""
        self.log_success(f"Saved checkpoint to: {path}")
        self.log_metrics(metrics)

    def get_memory_logs(self):
        """Get all memory logs."""
        return self.memory_logs

    def clear_memory_logs(self):
        """Clear memory logs."""
        self.memory_logs = []

# Global logger instance
logger = GlobalLogger()