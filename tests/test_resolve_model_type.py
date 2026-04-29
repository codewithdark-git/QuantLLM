"""
Tests for :meth:`TurboModel.resolve_model_type`.

PR #27 introduced ``DEFAULT_ARCHITECTURE_FALLBACKS`` but the resolution
function consulted them only when the HF config returned an empty
``model_type`` -- which never happens in practice. These tests pin the
post-fix behaviour: the default fallback table is consulted for unknown
``model_type`` values, family-style suffixes are recognised, and explicit
registrations still win.
"""

import pytest

from quantllm.core.turbo_model import TurboModel


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})


def test_unknown_model_type_falls_back_to_family():
    """``qwen3`` is not registered in transformers <= 4.x but is a Qwen2
    derivative; resolution should return ``qwen2``."""
    assert TurboModel.resolve_model_type(
        "Qwen/Qwen3-8B",
        config_model_type="qwen3",
    ) == "qwen2"


def test_family_match_against_model_type_directly():
    """Resolving by ``model_type`` alone should still pick up the family
    even when the repo name does not contain the family marker."""
    assert TurboModel.resolve_model_type(
        "Qwen/private-fork",
        config_model_type="qwen3",
    ) == "qwen2"


def test_specific_family_wins_over_generic():
    """``phi3`` has its own entry and must NOT be flattened to ``phi``."""
    assert TurboModel.resolve_model_type(
        "microsoft/phi-4",
        config_model_type="phi4",
    ) == "phi3"


def test_user_registered_alias_takes_precedence():
    TurboModel.register_architecture("zoolm", base_model_type="mistral")
    assert TurboModel.resolve_model_type(
        "org/zoolm-13b",
        config_model_type="zoolm",
    ) == "mistral"


def test_override_takes_precedence_over_everything():
    TurboModel.register_architecture("zoolm", base_model_type="mistral")
    assert TurboModel.resolve_model_type(
        "Qwen/Qwen3-8B",
        config_model_type="qwen3",
        model_type_override="llama",
    ) == "llama"


def test_truly_unknown_model_type_is_returned_unchanged():
    """When nothing matches we surface the original ``model_type`` so the
    caller can decide how to react (registering, overriding, etc.)."""
    assert TurboModel.resolve_model_type(
        "org/exotic-1b",
        config_model_type="something-totally-new",
    ) == "something-totally-new"


def test_no_config_returns_none_when_name_has_no_marker():
    assert TurboModel.resolve_model_type("org/exotic-1b") is None


def test_name_pattern_used_when_config_missing():
    """When ``config_model_type`` is empty the name is consulted for tokens."""
    assert TurboModel.resolve_model_type("meta-llama/Llama-3.2-3B") == "llama"


def test_register_architecture_class_lookup_uses_original_name(monkeypatch):
    """Bug from review: ``register_architecture("newmodel", base_model_type="llama",
    model_class=Cls)`` used to register the class under ``"newmodel"`` but
    look it up under ``"llama"`` and find nothing. Verify the class is now
    discoverable by the original architecture name."""
    sentinel = object()

    class _Stub:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return sentinel

    TurboModel.register_architecture(
        "newmodel",
        base_model_type="llama",
        model_class=_Stub,
    )

    # Direct registry assertions.
    assert TurboModel._architecture_registry["newmodel"] == "llama"
    assert TurboModel._model_class_registry["newmodel"] is _Stub
