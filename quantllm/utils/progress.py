"""
Terminal output, progress bars, and branded banner for QuantLLM.

All functions degrade gracefully in non-TTY environments.  Rich progress
bars are tried first, falling back to tqdm, then to plain text.  All
text output uses only ASCII characters -- no emoji, no unicode box
drawing that could cause encoding errors on Windows cp437.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Callable, Generator, Iterator, Optional, TypeVar, Union

# ---------------------------------------------------------------------------
# Theme constants (kept for API compatibility; consumers apply their own color)
# ---------------------------------------------------------------------------
QUANTLLM_ORANGE = "orange1"
QUANTLLM_ORANGE_LIGHT = "dark_orange"
QUANTLLM_ACCENT = "orange3"

# ---------------------------------------------------------------------------
# Console  --  plain-print wrapper (no rich dependency)
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


class _SimpleConsole:
    """Stand-in for ``rich.console.Console`` that writes to stderr via print.

    All ``print``-style methods are provided so callers that received a
    ``Console`` object from ``quantllm.utils.console`` continue to work.
    """

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print to stderr, stripping rich markup tags."""
        text = " ".join(str(a) for a in args)
        text = _strip_rich_markup(text)
        file = kwargs.pop("file", sys.stderr)
        print(text, file=file, **kwargs)

    # The rich Console also exposes ``log``, ``rule``, ``input`` etc.
    # We provide no-op stubs so callers that pass ``file=console`` don't
    # crash -- they just don't get rich formatting.
    def log(self, *args: Any, **kwargs: Any) -> None:
        self.print(*args, **kwargs)

    def rule(self, *args: Any, **kwargs: Any) -> None:
        pass

    def input(self, *args: Any, **kwargs: Any) -> str:
        text = " ".join(str(a) for a in args)
        return builtins_input(_strip_rich_markup(text))

    def __repr__(self) -> str:
        return "SimpleConsole()"


builtins_input = __builtins__["input"] if isinstance(__builtins__, dict) else __builtins__.input  # type: ignore[assignment]

console = _SimpleConsole()


def _strip_rich_markup(text: str) -> str:
    """Remove ``[bold ...][/]`` rich markup tags from *text*."""
    text = re.sub(r"\[/?\w+(?:\s+\w+(?:=.*?)?)*\]", "", text)
    return text


# ---------------------------------------------------------------------------
# Progress bars  --  try rich / tqdm, fall back to simple text
# ---------------------------------------------------------------------------

_ProgressObj = Any  # rich.Progress | tqdm.tqdm | _TextProgress | Progress


class _TextProgress:
    """Ultra-simple fallback progress indicator.

    Prints ``description: 42%`` lines as the progress advances.
    """

    def __init__(self, transient: bool = True) -> None:
        self._transient = transient
        self._tasks: dict[int, dict[str, Any]] = {}
        self._next_id = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def add_task(self, description: str, total: Optional[Union[int, float]] = None) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tasks[tid] = {"description": description, "total": total, "completed": 0.0}
        return tid

    def update(
        self,
        task_id: int,
        *,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return
        if description is not None:
            task["description"] = description
        if advance is not None:
            task["completed"] += advance
        if completed is not None:
            task["completed"] = completed

        total = task.get("total")
        pct = (task["completed"] / total * 100) if total else 0
        desc = _strip_rich_markup(task["description"])
        print(f"  {desc}: {pct:.0f}%", file=sys.stderr)

    def advance(self, task_id: int, advance: float = 1) -> None:
        self.update(task_id, advance=advance)

    def __enter__(self) -> "_TextProgress":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


# Module-level flag so we only log the "rich unavailable" warning once.
_HAS_RICH = False
_RICH_CHECKED = False


def _check_rich() -> bool:
    global _HAS_RICH, _RICH_CHECKED
    if not _RICH_CHECKED:
        try:
            import rich  # noqa: F401

            _HAS_RICH = True
        except ImportError:
            _HAS_RICH = False
        _RICH_CHECKED = True
    return _HAS_RICH


def _best_progress(style: str = "default", transient: bool = True) -> Any:
    """Return the best available progress-bar implementation.

    Preference: rich > tqdm > ``_TextProgress``.
    """
    if _check_rich():
        return _rich_progress(style, transient)
    try:
        from tqdm.auto import tqdm  # noqa: F401

        return _tqdm_progress(style, transient)
    except ImportError:
        pass
    return _TextProgress(transient=transient)


def _rich_progress(style: str, transient: bool) -> Any:
    """Build a rich Progress with columns matching *style*."""
    from rich.console import Console as RichConsole
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    theme_overrides = {
        "progress.description": "bold orange1",
        "progress.percentage": "bold orange1",
        "bar.complete": "orange1",
        "bar.finished": "green",
    }

    style_columns = {
        "default": [
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold orange1]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn(""),
            TimeElapsedColumn(),
        ],
        "spinner": [
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold orange1]{task.description}"),
        ],
        "download": [
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold orange1]{task.description}"),
            BarColumn(bar_width=40),
            DownloadColumn(),
            TextColumn(""),
            TransferSpeedColumn(),
        ],
        "steps": [
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold orange1]{task.description}"),
            MofNCompleteColumn(),
            BarColumn(bar_width=30),
            TimeElapsedColumn(),
        ],
    }

    columns = style_columns.get(style, style_columns["default"])
    return Progress(*columns, transient=transient)


def _tqdm_progress(style: str, transient: bool) -> Any:
    """Build a tqdm progress bar as a context manager."""
    from tqdm.auto import tqdm

    class _TqdmWrapper:
        def __init__(self) -> None:
            self._bar: Any = None

        def start(self) -> None:
            pass

        def stop(self) -> None:
            if self._bar is not None:
                self._bar.close()

        def add_task(self, description: str, total: Optional[Union[int, float]] = None) -> int:
            desc_clean = _strip_rich_markup(description)
            self._bar = tqdm(total=total, desc=desc_clean, unit="step", leave=not transient)
            return 0

        def update(
            self,
            task_id: int,
            *,
            completed: Optional[float] = None,
            advance: Optional[float] = None,
            description: Optional[str] = None,
            **kwargs: Any,
        ) -> None:
            if self._bar is None:
                return
            if advance is not None:
                self._bar.update(advance)
            elif completed is not None and self._bar.total:
                self._bar.n = int(completed)
                self._bar.refresh()

        def advance(self, task_id: int, advance: float = 1) -> None:
            self.update(task_id, advance=advance)

        def __enter__(self) -> "_TqdmWrapper":
            self.start()
            return self

        def __exit__(self, *args: Any) -> None:
            self.stop()

    return _TqdmWrapper()


class QuantLLMProgress:
    """Unified progress tracking that works in every environment.

    Tries rich (fancy) > tqdm (compact) > simple text (always works).

    Usage::

        with QuantLLMProgress() as p:
            task = p.add_task("Processing...", total=100)
            p.update(task, advance=1)
    """

    def __init__(self, transient: bool = True, disable_bar: bool = False, style: str = "default"):
        self._impl = _best_progress(style=style, transient=transient)

    def __enter__(self) -> Any:
        if hasattr(self._impl, "start"):
            self._impl.start()
        elif hasattr(self._impl, "__enter__"):
            return self._impl.__enter__()
        return self._impl

    def __exit__(self, *args: Any) -> None:
        if hasattr(self._impl, "stop"):
            self._impl.stop()
        elif hasattr(self._impl, "__exit__"):
            self._impl.__exit__(*args)


class StepProgress:
    """Multi-step progress tracker.

    Usage::

        with StepProgress(["Download", "Convert", "Quantize"]) as steps:
            steps.start("Download")
            steps.complete("Download")
    """

    def __init__(self, steps: list, title: str = "Progress"):
        self.steps = steps
        self.title = title
        self.step_status = {step: "pending" for step in steps}

    def __enter__(self) -> "StepProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def start(self, step: str) -> None:
        self.step_status[step] = "running"
        print(f"  >> {step}...", file=sys.stderr)

    def complete(self, step: str) -> None:
        self.step_status[step] = "complete"
        print(f"  OK  {step}", file=sys.stderr)

    def error(self, step: str, message: str = "") -> None:
        self.step_status[step] = "error"
        msg = f"  FAIL {step}" + (f" - {message}" if message else "")
        print(msg, file=sys.stderr)


def track_progress(
    sequence: Iterator[_T], description: str = "Processing...", total: Optional[int] = None
) -> Generator[_T, None, None]:
    """Iterate over *sequence* with a progress indicator."""
    seq = list(sequence)
    n = total if total is not None else len(seq)
    with QuantLLMProgress() as p:
        task = p.add_task(description, total=n)
        for item in seq:
            yield item
            p.update(task, advance=1)


def stream_subprocess_output(
    process: Any, description: str = "Processing...", parse_progress: Optional[Callable[[str], Optional[float]]] = None
) -> tuple[list[str], list[str]]:
    """Stream subprocess stdout/stderr with optional progress parsing."""
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    if parse_progress:
        with QuantLLMProgress() as progress:
            task = progress.add_task(description, total=100)
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                line = line.strip()
                stdout_lines.append(line)
                pct = parse_progress(line)
                if pct is not None:
                    progress.update(task, completed=pct)
            progress.update(task, completed=100)
    else:
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            stdout_lines.append(line.strip())
            print(f"  {line.strip()}", file=sys.stderr)

    process.wait()
    if process.stderr:
        stderr_lines = process.stderr.read().strip().split("\n")
    return stdout_lines, stderr_lines


# ---------------------------------------------------------------------------
# Simple print helpers (no rich, no emoji, no ANSI)
# ---------------------------------------------------------------------------


def print_header(title: str, icon: str = "") -> None:
    width = 50
    sep = "=" * width
    print(f"\n{sep}\n  {title}\n{sep}", file=sys.stderr)


def print_subheader(title: str) -> None:
    print(f"\n  -- {title} --", file=sys.stderr)


def print_success(msg: str) -> None:
    print(f"  OK  {msg}", file=sys.stderr)


def print_warning(msg: str) -> None:
    print(f"  WARN {msg}", file=sys.stderr)


def print_error(msg: str) -> None:
    print(f"  ERROR {msg}", file=sys.stderr)


def print_info(msg: str) -> None:
    print(f"  INFO {msg}", file=sys.stderr)


def print_step(step: int, total: int, msg: str) -> None:
    print(f"  [{step}/{total}] {msg}", file=sys.stderr)


def print_table(title: str, data: dict) -> None:
    print(f"\n  {title}:", file=sys.stderr)
    for key, value in data.items():
        print(f"    {key}: {value}", file=sys.stderr)
    print(file=sys.stderr)


def print_model_card(model_name: str, params: str, quantization: str, memory: str, **extras: str) -> None:
    print(f"\n  Model: {model_name}", file=sys.stderr)
    print(f"    Parameters:   {params}", file=sys.stderr)
    print(f"    Quantization: {quantization}", file=sys.stderr)
    print(f"    Memory:       {memory}", file=sys.stderr)
    for key, value in extras.items():
        print(f"    {key.replace('_', ' ').title()}: {value}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Banner  --  brand display with specs and functionality
# ---------------------------------------------------------------------------

_banner_shown = False


def print_banner(version: str = "2.0.0", force: bool = False) -> None:
    """Print the QuantLLM brand banner to stderr (once)."""
    global _banner_shown
    if _banner_shown and not force:
        return
    _banner_shown = True

    lines = [
        "+-------------------------------------------------+",
        f"|   QuantLLM v{version:<37} |",
        f"|   Ultra-fast LLM Quantization & Serving        |",
        "+-------------------------------------------------+",
        f"|   Formats:  GGUF | ONNX | MLX | SafeTensors    |",
        f"|   Features: Quantize | Finetune | Serve | Push  |",
        f"|   Server:   OpenAI-compatible API              |",
        f"|   Models:   Hub Registry + Auto-Architecture   |",
        f"|   Quant:    2-8 bit HQQ + GGUF (Q2-Q8, IQ)    |",
        "+-------------------------------------------------+",
    ]
    print("", file=sys.stderr)
    for line in lines:
        print(f"  {line}", file=sys.stderr)
    print("", file=sys.stderr)


def is_banner_shown() -> bool:
    return _banner_shown


def reset_banner() -> None:
    global _banner_shown
    _banner_shown = False


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_size(size_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
