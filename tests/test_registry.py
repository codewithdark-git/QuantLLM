"""Tests for the QuantLLM model registry.

All tests are offline-deterministic: the Hub is never contacted.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

import quantllm.registry as registry_mod

_REAL_FETCH = registry_mod.ModelRegistry._fetch_from_hub


@pytest.fixture(autouse=True)
def _offline_registry(monkeypatch):
    """Force the built-in model list and reset all caches."""
    monkeypatch.setattr(registry_mod.ModelRegistry, "_fetch_from_hub", lambda self: [])
    registry_mod._HUB_CACHE.clear()
    registry_mod._registry._models = None
    yield
    registry_mod._HUB_CACHE.clear()
    registry_mod._registry._models = None


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
        assert "1.1" in output  # 1.1B params


def _fake_api(models):
    api = mock.MagicMock()
    api.list_models.return_value = models
    return api


class TestHubFetch:
    def _fetch(self, models, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setattr(registry_mod, "_hub_offline", lambda: False)
        api = _fake_api(models)
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            result = _REAL_FETCH(registry_mod.ModelRegistry(org="TestOrg"))
        return result, api

    def test_new_style_id_attribute(self, monkeypatch):
        m = SimpleNamespace(id="TestOrg/Foo-GGUF", card_data={"quantllm_family": "Qwen", "params": 1.5})
        result, api = self._fetch([m], monkeypatch)
        assert result[0]["name"] == "TestOrg/Foo-GGUF"
        assert result[0]["family"] == "qwen"
        assert result[0]["params"] == 1.5
        assert "gguf" in result[0]["formats"]
        # one list call with card data, no per-model model_info calls
        assert api.list_models.call_args.kwargs.get("cardData") is True
        api.model_info.assert_not_called()

    def test_legacy_model_id_attribute(self, monkeypatch):
        m = SimpleNamespace(modelId="TestOrg/Legacy", cardData={"tags": ["quantllm", "mistral"]})
        result, _ = self._fetch([m], monkeypatch)
        assert result[0]["name"] == "TestOrg/Legacy"
        assert result[0]["family"] == "mistral"

    def test_card_data_object_with_to_dict(self, monkeypatch):
        cd = SimpleNamespace(to_dict=lambda: {"quantllm_family": "phi"})
        m = SimpleNamespace(id="TestOrg/Phi", card_data=cd)
        result, _ = self._fetch([m], monkeypatch)
        assert result[0]["family"] == "phi"

    def test_results_cached_in_memory(self, monkeypatch):
        m = SimpleNamespace(id="TestOrg/Cached", card_data={})
        self._fetch([m], monkeypatch)
        api = _fake_api([])
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            again = _REAL_FETCH(registry_mod.ModelRegistry(org="TestOrg"))
        assert again[0]["name"] == "TestOrg/Cached"
        api.list_models.assert_not_called()

    def test_hub_errors_fall_back_to_empty(self, monkeypatch):
        monkeypatch.setattr(registry_mod, "_hub_offline", lambda: False)
        api = mock.MagicMock()
        api.list_models.side_effect = OSError("network down")
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            assert _REAL_FETCH(registry_mod.ModelRegistry(org="TestOrg")) == []

    def test_offline_mode_skips_hub(self, monkeypatch):
        monkeypatch.setattr(registry_mod, "_hub_offline", lambda: True)
        with mock.patch("huggingface_hub.HfApi") as hf_api:
            assert _REAL_FETCH(registry_mod.ModelRegistry(org="TestOrg")) == []
        hf_api.assert_not_called()


class TestFamilyParsing:
    def test_dedicated_key_wins(self):
        assert registry_mod._parse_family({"quantllm_family": "Llama", "tags": ["quantllm", "gguf"]}) == "llama"

    def test_skips_generic_quantllm_tag(self):
        assert registry_mod._parse_family({"tags": ["quantllm", "gemma"]}) == "gemma"

    def test_unknown_when_missing(self):
        assert registry_mod._parse_family({}) == "unknown"
        assert registry_mod._parse_family({"tags": ["quantllm"]}) == "unknown"


class TestRegisterModel:
    def test_existing_card_updates_metadata_only(self):
        api = mock.MagicMock()
        api.file_exists.return_value = True
        existing = SimpleNamespace(data=SimpleNamespace(tags=["custom"]))
        with (
            mock.patch("huggingface_hub.HfApi", return_value=api),
            mock.patch("huggingface_hub.ModelCard.load", return_value=existing),
            mock.patch("huggingface_hub.metadata_update") as mu,
        ):
            ok = registry_mod.ModelRegistry().register_model("me/model", {"family": "qwen"})
        assert ok is True
        api.upload_file.assert_not_called()
        meta = mu.call_args.args[1]
        assert meta["quantllm_family"] == "qwen"
        assert meta["tags"] == ["custom", "quantllm", "qwen"]

    def test_new_repo_uploads_card(self):
        api = mock.MagicMock()
        api.file_exists.return_value = False
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            ok = registry_mod.ModelRegistry().register_model("me/model", {"family": "qwen"})
        assert ok is True
        content = api.upload_file.call_args.kwargs["path_or_fileobj"].decode()
        assert '"quantllm_family": "qwen"' in content

    def test_failure_returns_false(self):
        api = mock.MagicMock()
        api.create_repo.side_effect = RuntimeError("401")
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            assert registry_mod.ModelRegistry().register_model("me/model", {}) is False
