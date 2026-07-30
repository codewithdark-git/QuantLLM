"""
QuantLLM Benchmark Suite -- measure throughput, latency, VRAM.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch


@dataclass
class BenchmarkResult:
    model: str = ""
    backend: str = ""
    quant_method: str = ""
    bits: int = 4
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0
    peak_vram_gb: float = 0.0
    load_time_seconds: float = 0.0
    total_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    num_params_billion: float = 0.0
    model_size_gb: float = 0.0


class BenchmarkRunner:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def benchmark_model(self, model, tokenizer, config, prompt: str = "The meaning of life is",
                        max_new_tokens: int = 128, num_warmup: int = 3, num_runs: int = 5) -> BenchmarkResult:
        from ..utils import print_info, print_success
        r = BenchmarkResult(
            model=getattr(getattr(model, "config", None), "_name_or_path", "unknown"),
            bits=getattr(config, "bits", 16),
        )
        try:
            r.num_params_billion = model.num_parameters() / 1e9
        except Exception:
            pass
        r.model_size_gb = r.num_params_billion * (r.bits / 8) * 1.05

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        prefill_len = input_ids.shape[1]

        for _ in range(num_warmup):
            with torch.inference_mode():
                model.generate(input_ids, max_new_tokens=16, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

        start = time.time()
        with torch.inference_mode():
            model.generate(input_ids, max_new_tokens=1, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        r.prefill_tokens_per_sec = prefill_len / (time.time() - start) if (time.time() - start) > 0 else 0

        decode_times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.inference_mode():
                model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            elapsed = time.time() - start
            decode_times.append(max_new_tokens / elapsed if elapsed > 0 else 0)
        r.decode_tokens_per_sec = sum(decode_times) / len(decode_times)
        r.total_latency_ms = (1000.0 / r.decode_tokens_per_sec) if r.decode_tokens_per_sec > 0 else 0
        r.throughput_tokens_per_sec = r.decode_tokens_per_sec

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            r.peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        self.results.append(r)
        return r

    def save_results(self, path: str = "benchmark_results.json") -> str:
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        return path

    def generate_report(self) -> str:
        if not self.results:
            return "No results."
        lines = ["=" * 70, "  QuantLLM Benchmark Report", "=" * 70]
        for i, r in enumerate(self.results):
            lines += [
                f"\n  Run {i+1}: {r.model}", f"  {'='*50}",
                f"    Params:       {r.num_params_billion:.2f}B",
                f"    Bits:         {r.bits}",
                f"    Prefill:      {r.prefill_tokens_per_sec:.1f} tok/s",
                f"    Decode:       {r.decode_tokens_per_sec:.1f} tok/s",
                f"    Latency:      {r.total_latency_ms:.1f} ms/tok",
                f"    Throughput:   {r.throughput_tokens_per_sec:.1f} tok/s",
                f"    Peak VRAM:    {r.peak_vram_gb:.2f} GB",
            ]
        return "\n".join(lines)


def run_benchmark_cli(model_name: str, bits: int = 4, prompt: str = "The meaning of life is",
                      max_new_tokens: int = 128, save: bool = True, verbose: bool = True) -> BenchmarkResult:
    from .. import turbo
    m = turbo(model_name, bits=bits, verbose=verbose)
    runner = BenchmarkRunner(verbose=verbose)
    result = runner.benchmark_model(m.model, m.tokenizer, m.config, prompt=prompt, max_new_tokens=max_new_tokens)
    if save:
        runner.save_results()
    if verbose:
        print(runner.generate_report())
    return result
