"""Tests for the QuantLLM benchmark suite."""

import pytest


class TestBenchmarkRunner:
    def test_init(self):
        from quantllm.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(verbose=False)
        assert runner.results == []

    def test_generate_report_empty(self):
        from quantllm.benchmark import BenchmarkRunner

        runner = BenchmarkRunner(verbose=False)
        report = runner.generate_report()
        assert "No results" in report


class TestBenchmarkResult:
    def test_default_values(self):
        from quantllm.benchmark import BenchmarkResult

        r = BenchmarkResult()
        assert r.bits == 4
        assert r.decode_tokens_per_sec == 0.0


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, return_tensors=None):
        import torch

        return {"input_ids": torch.ones(1, len(text.split()), dtype=torch.long)}


class _FakeLM:
    """Fake LM on a fake clock: forward = 0.5s, generate = 0.5s prefill + 0.1s per extra token.

    It stops early (EOS) after ``new_tokens`` tokens regardless of max_new_tokens.
    """

    device = "cpu"

    def __init__(self, clock, new_tokens: int = 5):
        self.clock = clock
        self.new_tokens = new_tokens
        self.forward_lens = []

    def num_parameters(self):
        return 1000

    def __call__(self, input_ids=None, **kwargs):
        self.forward_lens.append(input_ids.shape[1])
        self.clock[0] += 0.5

    def generate(self, input_ids, max_new_tokens=16, **kwargs):
        import torch

        n = min(self.new_tokens, max_new_tokens)
        self.clock[0] += 0.5 + 0.1 * (n - 1)
        return torch.cat([input_ids, torch.ones(1, n, dtype=torch.long)], dim=1)


class TestBenchmarkMeasurement:
    def test_prefill_long_prompt_and_decode_actual_tokens(self, monkeypatch):
        import quantllm.benchmark.runner as runner_mod
        from quantllm.benchmark import BenchmarkRunner

        clock = [0.0]
        monkeypatch.setattr(runner_mod.time, "perf_counter", lambda: clock[0])

        model = _FakeLM(clock, new_tokens=5)
        r = BenchmarkRunner(verbose=False).benchmark_model(
            model, _FakeTok(), None, prompt="one two three", max_new_tokens=128, num_warmup=1, num_runs=2
        )
        # Prefill is a plain forward over a ~512-token prompt, not the 3-token benchmark prompt.
        assert max(model.forward_lens) == 512
        assert r.prefill_tokens_per_sec == pytest.approx(512 / 0.5)
        # 5 tokens actually generated (not 128); prefill (0.5s) excluded: 4 decode steps in 0.4s.
        assert r.decode_tokens_per_sec == pytest.approx(10.0)


class TestBenchmarkCLI:
    def test_bench_imports(self):
        from quantllm.benchmark import run_benchmark_cli, format_report_table

        assert callable(run_benchmark_cli)
        assert callable(format_report_table)
