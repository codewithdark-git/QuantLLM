from types import SimpleNamespace
from unittest.mock import Mock

import transformers

from quantllm.core.turbo_model import TurboModel
import quantllm.core.turbo_model as turbo_model_module


class _DummySmartConfig(SimpleNamespace):
    def print_summary(self):
        return None


def _make_smart_config():
    return _DummySmartConfig(
        bits=16,
        effective_loading_bits=16,
        dtype="float16",
        cpu_offload=False,
        device="cpu",
        gradient_checkpointing=False,
        use_flash_attention=False,
        compile_model=False,
    )


def _make_tokenizer():
    return SimpleNamespace(pad_token=None, eos_token="</s>", eos_token_id=2)


def test_resolve_model_type_detects_common_patterns():
    assert TurboModel.resolve_model_type("meta-llama/Llama-3.2-3B") == "llama"
    # Newer Qwen names still fall back to the qwen2 base family.
    assert TurboModel.resolve_model_type("Qwen/Qwen3-8B") == "qwen2"
    assert TurboModel.resolve_model_type("org/custom-arch-1b") is None


def test_register_architecture_maps_new_model_to_base_family(monkeypatch):
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})
    TurboModel.register_architecture("newmodel", base_model_type="llama")

    assert TurboModel.resolve_model_type("org/newmodel-7b") == "llama"


def test_registered_class_fallback_is_used(monkeypatch):
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})
    monkeypatch.setattr(
        turbo_model_module.SmartConfig,
        "detect",
        lambda *args, **kwargs: _make_smart_config(),
    )
    monkeypatch.setattr(
        turbo_model_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _make_tokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="newmodel",
            quantization_config=None,
        ),
    )

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError("Unrecognized configuration class")

        @staticmethod
        def from_config(*args, **kwargs):
            return SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    registered_call = Mock()

    def _registered_from_pretrained(cls, *args, **kwargs):
        registered_call()
        return SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    class _RegisteredModel:
        from_pretrained = classmethod(_registered_from_pretrained)

    monkeypatch.setattr(
        turbo_model_module,
        "AutoModelForCausalLM",
        _FakeAutoModel,
    )

    TurboModel.register_architecture("newmodel", base_model_type="llama")
    TurboModel.register_architecture("llama", model_class=_RegisteredModel)

    loaded = TurboModel.from_pretrained(
        "org/newmodel-7b",
        quantize=False,
        verbose=False,
    )

    assert registered_call.called is True
    assert loaded.model.config.model_type == "llama"


def test_from_pretrained_supports_from_config_only(monkeypatch):
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})
    monkeypatch.setattr(
        turbo_model_module.SmartConfig,
        "detect",
        lambda *args, **kwargs: _make_smart_config(),
    )
    monkeypatch.setattr(
        turbo_model_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _make_tokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="llama",
            quantization_config=None,
        ),
    )

    class _FakeAutoModel:
        called_from_pretrained = False
        called_from_config = False

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.called_from_pretrained = True
            return SimpleNamespace(config=SimpleNamespace(model_type="llama"))

        @classmethod
        def from_config(cls, *args, **kwargs):
            cls.called_from_config = True
            return SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    monkeypatch.setattr(
        turbo_model_module,
        "AutoModelForCausalLM",
        _FakeAutoModel,
    )

    loaded = TurboModel.from_pretrained(
        "org/llama-like-7b",
        quantize=False,
        verbose=False,
        from_config_only=True,
    )

    assert _FakeAutoModel.called_from_pretrained is False
    assert _FakeAutoModel.called_from_config is True
    assert loaded.model.config.model_type == "llama"
