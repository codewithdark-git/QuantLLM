from typing import Any, Dict
import datetime
from enum import Enum

class LogLevel(Enum):
    INFO = "\033[94m"  # Blue
    SUCCESS = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    RESET = "\033[0m"  # Reset color

class TrainingLogger:
    def __init__(self):
        """Initialize the training logger and display welcome message."""
        self.log_welcome_message()
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
            "╚══════════════════════════════════════════════════════════════════════════════════╝\n"            f"{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}╭────────────────── Welcome to QuantLLM v{__version__} ───────────────────╮{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}│                                                                         │{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}│  🎯 State-of-the-Art Model Quantization & Efficient Training           │{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}│  🚀 Optimized for Production Deployment                                │{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}│  💻 Supports CUDA, CPU, and Apple Silicon                              │{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}│                                                                         │{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}╰─────────────────────────────────────────────────────────────────────────╯{LogLevel.RESET.value}\n\n"
            f"{LogLevel.INFO.value}📊 Supported Methods:{LogLevel.RESET.value}\n"
            f"  {LogLevel.SUCCESS.value}• AWQ (Activation-aware Weight Quantization){LogLevel.RESET.value}\n"
            f"  {LogLevel.SUCCESS.value}• GPTQ (Goyal-Pham-Tan Quantization){LogLevel.RESET.value}\n"
            f"  {LogLevel.SUCCESS.value}• GGUF (GGML Universal Format){LogLevel.RESET.value}\n\n"
            f"{LogLevel.INFO.value}🛠️  Features:{LogLevel.RESET.value}\n"
            f"  {LogLevel.SUCCESS.value}• Advanced 2-8 bit Quantization{LogLevel.RESET.value}\n"
            f"  {LogLevel.SUCCESS.value}• Memory-Efficient Processing{LogLevel.RESET.value}\n"
            f"  {LogLevel.SUCCESS.value}• LoRA & QLoRA Integration{LogLevel.RESET.value}\n"
            f"  {LogLevel.SUCCESS.value}• Real-time Progress Tracking{LogLevel.RESET.value}\n"
            f"  {LogLevel.SUCCESS.value}• Comprehensive Benchmarking{LogLevel.RESET.value}\n\n"            f"{LogLevel.INFO.value}📚 Resources:{LogLevel.RESET.value}\n"
            f"  {LogLevel.SUCCESS.value}• Documentation: {LogLevel.RESET.value}https://github.com/codewithdark-git/QuantLLM/wiki\n"
            f"  {LogLevel.SUCCESS.value}• Source Code:   {LogLevel.RESET.value}https://github.com/codewithdark-git/QuantLLM\n"
            f"  {LogLevel.SUCCESS.value}• Discord:       {LogLevel.RESET.value}https://discord.gg/quantllm\n"
            f"  {LogLevel.SUCCESS.value}• Examples:      {LogLevel.RESET.value}https://github.com/codewithdark-git/QuantLLM/examples\n\n"
            f"{LogLevel.WARNING.value}💡 Get Started:{LogLevel.RESET.value}\n"
            f"  {LogLevel.SUCCESS.value}• Quick Start:   {LogLevel.RESET.value}https://quantllm.readthedocs.io/quickstart.html\n"
            f"  {LogLevel.SUCCESS.value}• Tutorials:     {LogLevel.RESET.value}https://quantllm.readthedocs.io/tutorials/\n"
            f"  {LogLevel.SUCCESS.value}• API Docs:      {LogLevel.RESET.value}https://quantllm.readthedocs.io/api/\n\n"
            f"{LogLevel.SUCCESS.value}╭─ Let's optimize your LLMs! Type help() for assistance ─╮{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}│                🚀 Happy Quantizing! 🚀                │{LogLevel.RESET.value}\n"
            f"{LogLevel.SUCCESS.value}╰───────────────────────────────────────────────────────╯{LogLevel.RESET.value}\n"
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

    def log_success(self, message: str):
        """Log success message."""
        print(self._format_message(LogLevel.SUCCESS, message))

    def log_warning(self, message: str):
        """Log warning message."""
        print(self._format_message(LogLevel.WARNING, message))

    def log_error(self, message: str):
        """Log error message."""
        print(self._format_message(LogLevel.ERROR, message))

    def log_metrics(self, metrics: dict, step: int = None):
        """Log training metrics."""
        step_str = f" (Step {step})" if step is not None else ""
        message = f"Metrics{step_str}: {self._format_metrics(metrics)}"
        print(self._format_message(LogLevel.INFO, message))

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