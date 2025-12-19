"""
Unified Progress Tracking and Logging System for QuantLLM.

Uses `rich` for beautiful output and progress bars.
Consistent orange theme across the entire project.
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

# QuantLLM Brand Colors - Orange Theme
QUANTLLM_ORANGE = "orange1"
QUANTLLM_ORANGE_LIGHT = "dark_orange"
QUANTLLM_ACCENT = "orange3"

# Custom theme - QuantLLM Orange Theme
custom_theme = Theme({
    # Primary colors
    "info": QUANTLLM_ORANGE,
    "warning": "yellow",
    "error": "red bold",
    "success": "green bold",
    
    # Progress bar styling
    "progress.description": f"bold {QUANTLLM_ORANGE}",
    "progress.percentage": f"bold {QUANTLLM_ORANGE}",
    "progress.remaining": "dim white",
    "bar.complete": QUANTLLM_ORANGE,
    "bar.finished": "green",
    "bar.pulse": QUANTLLM_ORANGE_LIGHT,
    
    # Table styling
    "table.header": f"bold {QUANTLLM_ORANGE}",
    "table.border": QUANTLLM_ACCENT,
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
            SpinnerColumn(spinner_name="dots", style=QUANTLLM_ORANGE),
            TextColumn(f"[bold {QUANTLLM_ORANGE}]{{task.description}}"),
            BarColumn(bar_width=40, complete_style=QUANTLLM_ORANGE, finished_style="green"),
            TaskProgressColumn(),
            TextColumn("[dim]•[/]"),
            TimeElapsedColumn(),
            TextColumn("[dim]•[/]"),
            TimeRemainingColumn(),
        ],
        "spinner": [
            SpinnerColumn(spinner_name="dots", style=QUANTLLM_ORANGE),
            TextColumn(f"[bold {QUANTLLM_ORANGE}]{{task.description}}"),
        ],
        "download": [
            SpinnerColumn(spinner_name="dots", style=QUANTLLM_ORANGE),
            TextColumn(f"[bold {QUANTLLM_ORANGE}]{{task.description}}"),
            BarColumn(bar_width=40, complete_style=QUANTLLM_ORANGE),
            DownloadColumn(),
            TextColumn("[dim]•[/]"),
            TransferSpeedColumn(),
            TextColumn("[dim]•[/]"),
            TimeRemainingColumn(),
        ],
        "steps": [
            SpinnerColumn(spinner_name="dots", style=QUANTLLM_ORANGE),
            TextColumn(f"[bold {QUANTLLM_ORANGE}]{{task.description}}"),
            MofNCompleteColumn(),
            BarColumn(bar_width=30, complete_style=QUANTLLM_ORANGE),
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
    Multi-step progress tracker for complex operations with orange theme.
    
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
        colors = {"pending": "dim", "running": QUANTLLM_ORANGE, "complete": "green", "error": "red"}
        
        lines = [f"[bold {QUANTLLM_ORANGE}]{self.title}[/]"]
        for i, step in enumerate(self.steps, 1):
            status = self.step_status[step]
            icon = icons[status]
            color = colors[status]
            lines.append(f"  [{color}]{icon}[/] Step {i}: {step}")
        
        console.print("\n".join(lines))
    
    def start(self, step: str):
        """Mark step as running."""
        self.step_status[step] = "running"
        console.print(f"[{QUANTLLM_ORANGE}]▶[/] Starting: {step}...")
    
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
    """Simple wrapper around rich.progress.track with orange theme."""
    columns = [
        SpinnerColumn(spinner_name="dots", style=QUANTLLM_ORANGE),
        TextColumn(f"[bold {QUANTLLM_ORANGE}]{{task.description}}"),
        BarColumn(bar_width=40, complete_style=QUANTLLM_ORANGE),
        TaskProgressColumn(),
        TextColumn("[dim]•[/]"),
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
# Helper Functions - Orange Theme
# ============================================

def print_header(title: str, icon: str = "🚀") -> None:
    """Print a styled header with orange theme."""
    width = 60
    console.print()
    console.print(f"[bold {QUANTLLM_ORANGE}]{'═' * width}[/]")
    console.print(f"[bold {QUANTLLM_ORANGE}]║[/] {icon} [bold white]{title.center(width - 6)}[/] [bold {QUANTLLM_ORANGE}]║[/]")
    console.print(f"[bold {QUANTLLM_ORANGE}]{'═' * width}[/]")

def print_subheader(title: str) -> None:
    """Print a smaller styled subheader."""
    console.print(f"\n[bold {QUANTLLM_ORANGE}]── {title} ──[/]")

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
    """Print info message with orange icon."""
    console.print(f"[{QUANTLLM_ORANGE}]ℹ[/] {msg}")

def print_step(step: int, total: int, msg: str) -> None:
    """Print a step in a multi-step process."""
    console.print(f"[bold {QUANTLLM_ORANGE}][{step}/{total}][/] {msg}")

def print_table(title: str, data: dict) -> None:
    """Print a formatted table with orange theme."""
    table = Table(title=title, show_header=False, border_style=QUANTLLM_ORANGE)
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
    """Print a model summary card with orange theme."""
    table = Table(title=f"[bold]{model_name}[/]", border_style=QUANTLLM_ORANGE, show_header=False)
    table.add_column("Property", style=f"bold {QUANTLLM_ORANGE}")
    table.add_column("Value")
    
    table.add_row("Parameters", params)
    table.add_row("Quantization", quantization)
    table.add_row("Memory Usage", memory)
    
    for key, value in extras.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(table)


def print_banner() -> None:
    """Print QuantLLM banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗            ║
    ║  ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝            ║
    ║  ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║               ║
    ║  ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║               ║
    ║  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   LLM         ║
    ║   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝               ║
    ║                                                           ║
    ║          Ultra-fast LLM Quantization & Export             ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(f"[bold {QUANTLLM_ORANGE}]{banner}[/]")


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_time(seconds: float) -> str:
    """Format seconds to human readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"
