"""
QuantLLM Benchmark comparison -- QuantLLM vs raw transformers/BnB.
"""

from __future__ import annotations

from typing import Any, Dict

from .runner import BenchmarkRunner, BenchmarkResult


class BenchmarkComparer:
    def __init__(self, model_name: str, verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.results: Dict[str, BenchmarkResult] = {}

    def run_comparison(self, prompt: str = "The meaning of life is", max_new_tokens: int = 128) -> Dict[str, Any]:
        from .. import turbo
        from ..utils import print_header

        configs = [
            ("FP16", 16, None),
            ("BnB-8bit", 8, None),
            ("BnB-4bit", 4, None),
        ]
        runner = BenchmarkRunner(verbose=self.verbose)

        for name, bits, _ in configs:
            try:
                m = turbo(self.model_name, bits=bits, verbose=False)
                r = runner.benchmark_model(m.model, m.tokenizer, m.config, prompt=prompt, max_new_tokens=max_new_tokens)
                r.backend = "transformers"
                self.results[name] = r
            except Exception as e:
                if self.verbose:
                    from ..utils import print_warning
                    print_warning(f"{name} failed: {e}")
        return self.results

    def generate_comparison_report(self) -> str:
        if not self.results:
            return "No data."
        lines = ["=" * 80, "  QuantLLM Comparison", f"  Model: {self.model_name}", "=" * 80,
                 f"  {'Config':<15} {'Decode':<12} {'Prefill':<12} {'Latency':<12} {'VRAM':<10} {'Size':<10}",
                 f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}"]
        for name, r in self.results.items():
            lines.append(f"  {name:<15} {r.decode_tokens_per_sec:<12.1f} {r.prefill_tokens_per_sec:<12.1f} "
                         f"{r.total_latency_ms:<12.1f} {r.peak_vram_gb:<10.2f} {r.model_size_gb:<10.2f}")

        best = max(self.results, key=lambda k: self.results[k].decode_tokens_per_sec) if self.results else None
        if best:
            lines.append(f"\n  Recommended: {best}")
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
