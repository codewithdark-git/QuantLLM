"""
Unified Progress Tracking and Logging System for QuantLLM.

Uses `rich` for beautiful output and progress bars.
"""

import logging
from typing import Optional, ContextManager, Generator, Iterable
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.theme import Theme

# Custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
})

console = Console(theme=custom_theme)

# ============================================
# Logging Configuration
# ============================================

def configure_logging(level: str = "INFO") -> None:
    """Configure rich-based logging."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)]
    )

def get_logger(name: str = "quantllm") -> logging.Logger:
    """Get the project logger."""
    return logging.getLogger(name)

logger = get_logger()

# ============================================
# Progress Tracking
# ============================================

class QuantLLMProgress:
    """
    Unified progress tracking context manager.
    
    Usage:
        with QuantLLMProgress() as p:
            task = p.add_task("Processing...", total=100)
            p.update(task, advance=1)
    """
    
    def __init__(self, transient: bool = True):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
            transient=transient,
        )
    
    def __enter__(self) -> Progress:
        self.progress.start()
        return self.progress
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()


def track_progress(
    sequence: Iterable,
    description: str = "Processing...",
    total: Optional[int] = None
) -> Iterable:
    """Simple wrapper around rich.progress.track."""
    columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        "•",
        TimeRemainingColumn(),
    ]
    
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task(description, total=total or len(sequence))
        for item in sequence:
            yield item
            progress.advance(task)


# ============================================
# Helper Functions
# ============================================

def print_header(title: str, icon: str = "🚀") -> None:
    """Print a styled header."""
    width = 60
    console.print()
    console.print(f"[bold blue]{'=' * width}[/]")
    console.print(f"{icon} [bold white]{title.center(width - 4)}[/]")
    console.print(f"[bold blue]{'=' * width}[/]")
    console.print()

def print_success(msg: str) -> None:
    """Print success message."""
    console.print(f"[bold green]✅ Success:[/][white] {msg}[/]")

def print_warning(msg: str) -> None:
    """Print warning message."""
    console.print(f"[bold yellow]⚠️  Warning:[/][white] {msg}[/]")

def print_error(msg: str) -> None:
    """Print error message."""
    console.print(f"[bold red]❌ Error:[/][white] {msg}[/]")

def print_info(msg: str) -> None:
    """Print info message."""
    console.print(f"[bold cyan]ℹ️  Info:[/][white] {msg}[/]")
