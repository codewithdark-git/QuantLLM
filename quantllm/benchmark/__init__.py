"""
QuantLLM Benchmark Suite package.
"""

from .runner import BenchmarkRunner, BenchmarkResult, run_benchmark_cli
from .compare import BenchmarkComparer
from .report import format_report_table
from .auto_bench import HardwareAutoBench

# Add a simple run_benchmark function as alias
run_benchmark = run_benchmark_cli

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "run_benchmark_cli",
    "run_benchmark",
    "BenchmarkComparer",
    "format_report_table",
    "HardwareAutoBench",
]
