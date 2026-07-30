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


class TestBenchmarkCLI:
    def test_bench_imports(self):
        from quantllm.benchmark import run_benchmark_cli, format_report_table
        assert callable(run_benchmark_cli)
        assert callable(format_report_table)
