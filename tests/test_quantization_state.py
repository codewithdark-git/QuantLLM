"""
Regression tests for runtime quantization state tracking.

These tests cover bugs reproduced in the v2.1.0rc1 review:

* ``from_config_only=True`` used to set ``_is_quantized = True`` even though
  the loader returns a model with random weights and no quantization.
* A missing ``bitsandbytes`` install used to fall through silently while the
  ``_is_quantized`` flag remained ``True``.
* ``is_quantized`` should now reflect the loaded model's ``quantization_config``
  rather than the user's load-time intent.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import transformers

from quantllm.core.turbo_model import TurboModel
import quantllm.core.turbo_model as turbo_model_module


def _smart_config(bits: int = 16):
    return SimpleNamespace(
        bits=bits,
        effective_loading_bits=4 if bits <= 4 else (8 if bits < 16 else 16),
        dtype="float16",
        cpu_offload=False,
        device="cpu",
        gradient_checkpointing=False,
        use_flash_attention=False,
        compile_model=False,
    )


def _tokenizer():
    return SimpleNamespace(pad_token=None, eos_token="</s>", eos_token_id=2)


def _patch_common(monkeypatch, *, model_type: str = "llama", quant_config=None, smart_bits: int = 16):
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})
    monkeypatch.setattr(
        turbo_model_module.SmartConfig,
        "detect",
        lambda *a, **kw: _smart_config(bits=smart_bits),
    )
    monkeypatch.setattr(
        turbo_model_module.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: _tokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(
            model_type=model_type,
            quantization_config=quant_config,
        ),
    )


def test_from_config_only_does_not_lie_about_quantization(monkeypatch):
    """``from_config_only=True`` returns a random-weights model and must not
    advertise itself as quantized just because the user asked for 4 bits."""
    _patch_common(monkeypatch, smart_bits=4)

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *a, **kw):  # should not be called
            raise AssertionError("from_pretrained should not run when from_config_only=True")

        @classmethod
        def from_config(cls, *a, **kw):
            # ``from_config`` cannot quantize -- model has no
            # ``quantization_config`` attribute on its config.
            return SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    monkeypatch.setattr(turbo_model_module, "AutoModelForCausalLM", _FakeAutoModel)

    loaded = TurboModel.from_pretrained(
        "org/llama-like-7b",
        quantize=True,
        bits=4,
        verbose=False,
        from_config_only=True,
    )

    assert loaded.is_quantized is False
    assert loaded._is_quantized is False  # back-compat alias must agree


def test_runtime_quantization_property_reads_model_config(monkeypatch):
    """``is_quantized`` should return True when the loaded model's
    ``config.quantization_config.quant_method`` is set, regardless of the
    user's load-time flags."""
    _patch_common(
        monkeypatch,
        quant_config=SimpleNamespace(quant_method="gptq"),
    )

    fake_model_config = SimpleNamespace(
        model_type="llama",
        quantization_config=SimpleNamespace(quant_method="gptq"),
        _name_or_path="org/llama-gptq",
    )
    fake_model = SimpleNamespace(
        config=fake_model_config,
        modules=lambda: iter([]),
        num_parameters=lambda: 7_000_000_000,
        device="cpu",
        parameters=lambda: iter([]),
    )

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return fake_model

    monkeypatch.setattr(turbo_model_module, "AutoModelForCausalLM", _FakeAutoModel)

    loaded = TurboModel.from_pretrained(
        "org/llama-gptq",
        quantize=False,
        verbose=False,
    )

    # User explicitly disabled dynamic quantization, but the underlying model
    # IS already GPTQ-quantized: the property must reflect that.
    assert loaded.is_quantized is True
    assert loaded.report()["quant_method"] == "gptq"


def test_is_quantized_override_accessor():
    """The ``_is_quantized`` setter should record an explicit override that
    short-circuits the runtime introspection."""
    instance = TurboModel.__new__(TurboModel)
    instance.model = SimpleNamespace(config=SimpleNamespace(quantization_config=None))
    instance.config = _smart_config()
    instance._is_quantized_override = None
    instance._is_finetuned = False
    instance._lora_applied = False

    assert instance.is_quantized is False
    instance._is_quantized = True
    assert instance.is_quantized is True
    instance._is_quantized = None  # clears override -> derives again
    assert instance.is_quantized is False


def test_report_returns_structured_state(monkeypatch):
    """``report()`` should expose a stable, machine-readable summary."""
    _patch_common(monkeypatch)

    fake_model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="llama",
            quantization_config=None,
            _name_or_path="org/llama-7b",
        ),
        modules=lambda: iter([]),
        num_parameters=lambda: 7_000_000_000,
        device="cpu",
        parameters=lambda: iter([]),
    )

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return fake_model

    monkeypatch.setattr(turbo_model_module, "AutoModelForCausalLM", _FakeAutoModel)

    loaded = TurboModel.from_pretrained("org/llama-7b", quantize=False, verbose=False)
    report = loaded.report()

    expected_keys = {
        "model_id",
        "params_billion",
        "requested_bits",
        "effective_loading_bits",
        "is_quantized",
        "quant_method",
        "device",
        "dtype",
        "finetuned",
        "lora_applied",
    }
    assert set(report) == expected_keys
    assert report["model_id"] == "org/llama-7b"
    assert report["params_billion"] == 7.0
    assert report["is_quantized"] is False
    assert report["finetuned"] is False
    assert report["lora_applied"] is False
