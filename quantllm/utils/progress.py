"""
QuantLLM Progress Utilities - Beautiful one-line progress bars and logging.

Provides:
- Custom progress bars for model loading and training
- Formatted logging with colors
- Streaming text output
"""

import sys
import time
from typing import Optional, Callable, Any, Iterator
from contextlib import contextmanager

# Try rich for beautiful output
try:
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeRemainingColumn, DownloadColumn,
    )
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class QuantLLMProgress:
    """One-line progress bar for QuantLLM operations."""
    
    def __init__(self, description: str = "Loading", total: int = 100):
        self.description = description
        self.total = total
        self.current = 0
        self._start_time = None
        self._progress = None
        self._task = None
        
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.finish()
    
    def start(self):
        """Start the progress bar."""
        self._start_time = time.time()
        
        if RICH_AVAILABLE:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=Console(stderr=True),
            )
            self._progress.start()
            self._task = self._progress.add_task(self.description, total=self.total)
        else:
            print(f"\r{self.description}... 0%", end="", flush=True)
    
    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update progress."""
        self.current = min(self.current + advance, self.total)
        
        if RICH_AVAILABLE and self._progress:
            update_kwargs = {"advance": advance}
            if description:
                update_kwargs["description"] = description
            self._progress.update(self._task, **update_kwargs)
        else:
            pct = int(100 * self.current / self.total)
            desc = description or self.description
            print(f"\r{desc}... {pct}%", end="", flush=True)
    
    def finish(self, message: Optional[str] = None):
        """Complete the progress bar."""
        if RICH_AVAILABLE and self._progress:
            self._progress.stop()
        else:
            print(f"\r{message or self.description}... Done!     ")


class ModelLoadingProgress:
    """Progress bar specifically for model loading."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._console = Console() if RICH_AVAILABLE else None
    
    @contextmanager
    def loading_tokenizer(self):
        """Context manager for tokenizer loading."""
        if RICH_AVAILABLE:
            with self._console.status("[bold blue]📝 Loading tokenizer..."):
                yield
            self._console.print("[green]✓[/green] Tokenizer loaded")
        else:
            print("📝 Loading tokenizer...", end=" ", flush=True)
            yield
            print("✓")
    
    @contextmanager
    def loading_model(self, bits: int = 16):
        """Context manager for model loading."""
        msg = f"📦 Loading model ({bits}-bit)"
        if RICH_AVAILABLE:
            with self._console.status(f"[bold blue]{msg}..."):
                yield
            self._console.print(f"[green]✓[/green] Model loaded")
        else:
            print(f"{msg}...", end=" ", flush=True)
            yield
            print("✓")
    
    def print_success(self, message: str):
        """Print success message."""
        if RICH_AVAILABLE:
            self._console.print(f"[bold green]✅ {message}[/bold green]")
        else:
            print(f"✅ {message}")
    
    def print_warning(self, message: str):
        """Print warning message."""
        if RICH_AVAILABLE:
            self._console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")
        else:
            print(f"⚠️  {message}")
    
    def print_error(self, message: str):
        """Print error message."""
        if RICH_AVAILABLE:
            self._console.print(f"[bold red]❌ {message}[/bold red]")
        else:
            print(f"❌ {message}")


class TrainingProgress:
    """Progress bar for training epochs."""
    
    def __init__(self, total_epochs: int, total_steps_per_epoch: int):
        self.total_epochs = total_epochs
        self.steps_per_epoch = total_steps_per_epoch
        self._progress = None
        self._epoch_task = None
        self._step_task = None
    
    def __enter__(self):
        if RICH_AVAILABLE:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=30),
                TaskProgressColumn(),
                TextColumn("•"),
                TextColumn("[cyan]{task.fields[loss]:.4f}[/cyan]", justify="right"),
                TimeRemainingColumn(),
                console=Console(stderr=True),
            )
            self._progress.start()
            self._epoch_task = self._progress.add_task(
                f"Epoch 0/{self.total_epochs}",
                total=self.total_epochs,
                loss=0.0,
            )
        return self
    
    def __exit__(self, *args):
        if self._progress:
            self._progress.stop()
    
    def update_epoch(self, epoch: int, loss: float = 0.0):
        """Update epoch progress."""
        if RICH_AVAILABLE and self._progress:
            self._progress.update(
                self._epoch_task,
                advance=1,
                description=f"Epoch {epoch}/{self.total_epochs}",
                loss=loss,
            )
        else:
            print(f"Epoch {epoch}/{self.total_epochs} | Loss: {loss:.4f}")
    
    def update_step(self, step: int, loss: float):
        """Update step within epoch."""
        if not RICH_AVAILABLE:
            pct = int(100 * step / self.steps_per_epoch)
            print(f"\r  Step {step}/{self.steps_per_epoch} ({pct}%) | Loss: {loss:.4f}", end="")


def stream_tokens(generator, print_fn=None):
    """
    Stream tokens as they're generated.
    
    Args:
        generator: Token generator
        print_fn: Custom print function (default: sys.stdout.write)
        
    Returns:
        Complete generated text
    """
    print_fn = print_fn or (lambda x: (sys.stdout.write(x), sys.stdout.flush()))
    tokens = []
    
    for token in generator:
        tokens.append(token)
        print_fn(token)
    
    return "".join(tokens)


def format_model_info(
    model_name: str,
    bits: int,
    device: str,
    dtype: str,
    params: int,
) -> str:
    """Format model info for display."""
    if RICH_AVAILABLE:
        console = Console()
        lines = [
            "",
            "═" * 50,
            f"[bold blue]Model:[/bold blue] {model_name}",
            f"[bold]Quantization:[/bold] {bits}-bit",
            f"[bold]Device:[/bold] {device}",
            f"[bold]Dtype:[/bold] {dtype}",
            f"[bold]Parameters:[/bold] {params:,}",
            "═" * 50,
            "",
        ]
        return "\n".join(lines)
    else:
        return f"""
{'='*50}
Model: {model_name}
Quantization: {bits}-bit
Device: {device}
Dtype: {dtype}
Parameters: {params:,}
{'='*50}
"""


# Export
__all__ = [
    "QuantLLMProgress",
    "ModelLoadingProgress", 
    "TrainingProgress",
    "stream_tokens",
    "format_model_info",
    "RICH_AVAILABLE",
]
