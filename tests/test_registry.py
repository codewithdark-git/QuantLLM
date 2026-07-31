"""Tests for the QuantLLM model registry."""

import pytest


class TestModelRegistry:
    def test_builtin_models_exist(self):
        from quantllm.registry import ModelRegistry

        reg = ModelRegistry()
        models = reg.models
        assert len(models) > 0

    def test_search_by_name(self):
        from quantllm.registry import ModelRegistry

        reg = ModelRegistry()
        results = reg.search("llama")
        assert len(results) > 0

    def test_search_by_family(self):
        from quantllm.registry import ModelRegistry

        reg = ModelRegistry()
        results = reg.search("llama")
        assert any("llama" in m["name"].lower() or m["family"] == "llama" for m in results)

    def test_get_model(self):
        from quantllm.registry import ModelRegistry

        reg = ModelRegistry()
        m = reg.get_model("QuantLLM/TinyLlama-1.1B-GGUF")
        if m:
            assert m["params"] == 1.1
            assert "gguf" in m["formats"]

    def test_get_nonexistent_model(self):
        from quantllm.registry import ModelRegistry

        reg = ModelRegistry()
        m = reg.get_model("nonexistent/model")
        assert m is None

    def test_filter_by_family(self):
        from quantllm.registry import ModelRegistry

        reg = ModelRegistry()
        results = reg.list_models(family="llama")
        for m in results:
            assert m["family"] == "llama"

    def test_format_list(self):
        from quantllm.registry import ModelRegistry

        reg = ModelRegistry()
        output = reg.format_list()
        assert "Model" in output
        assert "Family" in output


class TestRegistryModuleFunctions:
    def test_list_models(self):
        from quantllm.registry import list_models

        output = list_models()
        assert len(output) > 0

    def test_search_models(self):
        from quantllm.registry import search_models

        output = search_models("llama")
        assert len(output) > 0

    def test_model_info(self):
        from quantllm.registry import model_info

        output = model_info("QuantLLM/TinyLlama-1.1B-GGUF")
        assert "Params:" in output
