"""
QuantLLM Benchmark report generation.
"""

from .runner import BenchmarkResult


def format_report_table(results: list[BenchmarkResult]) -> str:
    if not results:
        return "No results."
    lines = ["=" * 90,
             f"  {'Model':<30} {'Backend':<12} {'Bits':<6} {'Decode':<10} {'VRAM':<10} {'Size':<10}",
             f"  {'-'*30} {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*10}"]
    for r in results:
        s = r.model.split("/")[-1][:28]
        lines.append(f"  {s:<30} {r.backend or 'transformers':<12} {r.bits:<6} {r.decode_tokens_per_sec:<10.1f} {r.peak_vram_gb:<10.2f} {r.model_size_gb:<10.2f}")
    lines.append("=" * 90)
    return "\n".join(lines)
