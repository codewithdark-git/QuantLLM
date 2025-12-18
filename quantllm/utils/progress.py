"""
Unified Progress Tracking and Logging System for QuantLLM.

Uses `rich` for beautiful output and progress bars.
Similar to Unsloth's clean UI but with QuantLLM styling.
"""

import logging
import sys
import re
from typing import Optional, ContextManager, Generator, Iterable, Callable, Any
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
    DownloadColumn,
    FileSizeColumn,
    MofNCompleteColumn,
)
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

# Custom theme - inspired by modern CLI tools
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green bold",
    "progress.description": "bold cyan",
    "progress.percentage": "bold magenta",
    "progress.remaining": "dim",
    "bar.complete": "green",
    "bar.finished": "green",
    "bar.pulse": "cyan",
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
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)]
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
            
        # Or with download style
        with QuantLLMProgress(style="download") as p:
            task = p.add_task("Downloading...", total=1024*1024)
            p.update(task, advance=chunk_size)
    """
    
    STYLES = {
        "default": [
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ],
        "spinner": [
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold cyan]{task.description}"),
        ],
        "download": [
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            DownloadColumn(),
            TextColumn("•"),
            TransferSpeedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ],
        "steps": [
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold cyan]{task.description}"),
            MofNCompleteColumn(),
            BarColumn(bar_width=30),
            TimeElapsedColumn(),
        ],
    }
    
    def __init__(
        self, 
        transient: bool = True, 
        disable_bar: bool = False,
        style: str = "default"
    ):
        if disable_bar:
            columns = self.STYLES["spinner"]
        else:
            columns = self.STYLES.get(style, self.STYLES["default"])
            
        self.progress = Progress(
            *columns,
            console=console,
            transient=transient,
            expand=False,
        )
    
    def __enter__(self) -> Progress:
        self.progress.start()
        return self.progress
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()


class StepProgress:
    """
    Multi-step progress tracker for complex operations.
    
    Usage:
        with StepProgress(["Download", "Convert", "Quantize"]) as steps:
            steps.start("Download")
            # ... download ...
            steps.complete("Download")
            
            steps.start("Convert")
            # ... convert ...
            steps.complete("Convert")
    """
    
    def __init__(self, steps: list, title: str = "Progress"):
        self.steps = steps
        self.title = title
        self.current_step = 0
        self.step_status = {step: "pending" for step in steps}
        
    def __enter__(self):
        self._print_status()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            console.print()
    
    def _print_status(self):
        """Print current status of all steps."""
        icons = {"pending": "○", "running": "◐", "complete": "●", "error": "✗"}
        colors = {"pending": "dim", "running": "cyan", "complete": "green", "error": "red"}
        
        lines = [f"[bold]{self.title}[/]"]
        for i, step in enumerate(self.steps, 1):
            status = self.step_status[step]
            icon = icons[status]
            color = colors[status]
            lines.append(f"  [{color}]{icon}[/] Step {i}: {step}")
        
        console.print("\n".join(lines))
    
    def start(self, step: str):
        """Mark step as running."""
        self.step_status[step] = "running"
        console.print(f"[cyan]▶[/] Starting: {step}...")
    
    def complete(self, step: str):
        """Mark step as complete."""
        self.step_status[step] = "complete"
        console.print(f"[green]✓[/] Completed: {step}")
    
    def error(self, step: str, message: str = ""):
        """Mark step as error."""
        self.step_status[step] = "error"
        console.print(f"[red]✗[/] Failed: {step}" + (f" - {message}" if message else ""))


def track_progress(
    sequence: Iterable,
    description: str = "Processing...",
    total: Optional[int] = None
) -> Iterable:
    """Simple wrapper around rich.progress.track."""
    columns = [
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]
    
    with Progress(*columns, console=console, transient=True) as progress:
        seq_list = list(sequence) if total is None else sequence
        task = progress.add_task(description, total=total or len(seq_list))
        for item in seq_list:
            yield item
            progress.advance(task)


def stream_subprocess_output(
    process,
    description: str = "Processing...",
    parse_progress: Optional[Callable[[str], Optional[float]]] = None,
) -> tuple:
    """
    Stream subprocess output with optional progress parsing.
    
    Args:
        process: subprocess.Popen object
        description: Description for progress bar
        parse_progress: Function that parses a line and returns progress (0-100) or None
        
    Returns:
        Tuple of (stdout_lines, stderr_lines)
    """
    stdout_lines = []
    stderr_lines = []
    
    if parse_progress:
        with QuantLLMProgress() as progress:
            task = progress.add_task(description, total=100)
            
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                line = line.strip()
                stdout_lines.append(line)
                
                # Try to parse progress
                pct = parse_progress(line)
                if pct is not None:
                    progress.update(task, completed=pct)
                else:
                    # Update description with latest line
                    progress.update(task, description=f"{description}: {line[:40]}...")
            
            progress.update(task, completed=100)
    else:
        # Just stream without progress
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            stdout_lines.append(line.strip())
            console.print(f"[dim]{line.strip()}[/]")
    
    process.wait()
    
    if process.stderr:
        stderr_lines = process.stderr.read().strip().split('\n')
    
    return stdout_lines, stderr_lines


# ============================================
# Helper Functions
# ============================================

def print_header(title: str, icon: str = "🚀") -> None:
    """Print a styled header."""
    width = 60
    console.print()
    console.print(f"[bold cyan]{'═' * width}[/]")
    console.print(f"[bold cyan]║[/] {icon} [bold white]{title.center(width - 6)}[/] [bold cyan]║[/]")
    console.print(f"[bold cyan]{'═' * width}[/]")

def print_subheader(title: str) -> None:
    """Print a smaller styled subheader."""
    console.print(f"\n[bold cyan]── {title} ──[/]")

def print_success(msg: str) -> None:
    """Print success message."""
    console.print(f"[green]✓[/] {msg}")

def print_warning(msg: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]⚠[/] {msg}")

def print_error(msg: str) -> None:
    """Print error message."""
    console.print(f"[red bold]✗ Error:[/] {msg}")

def print_info(msg: str) -> None:
    """Print info message."""
    console.print(f"[cyan]ℹ[/] {msg}")

def print_step(step: int, total: int, msg: str) -> None:
    """Print a step in a multi-step process."""
    console.print(f"[bold cyan][{step}/{total}][/] {msg}")

def print_table(title: str, data: dict) -> None:
    """Print a formatted table."""
    table = Table(title=title, show_header=False, border_style="cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    
    for key, value in data.items():
        table.add_row(str(key), str(value))
    
    console.print(table)

def print_model_card(
    model_name: str,
    params: str,
    quantization: str,
    memory: str,
    **extras
) -> None:
    """Print a model summary card."""
    table = Table(title=f"[bold]{model_name}[/]", border_style="cyan", show_header=False)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value")
    
    table.add_row("Parameters", params)
    table.add_row("Quantization", quantization)
    table.add_row("Memory Usage", memory)
    
    for key, value in extras.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(table)
