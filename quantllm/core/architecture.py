"""Architecture resolution and fallback logic for QuantLLM.

Provides the :class:`ArchitectureMixin` used by :class:`TurboModel` to
map unknown HuggingFace ``model_type`` values to known loadable base
families at load time.
"""

import re
from functools import lru_cache
from typing import Dict, Optional, Type

from transformers import PreTrainedModel


DEFAULT_ARCHITECTURE_FALLBACKS: Dict[str, str] = {
    "llama": "llama",
    "llama2": "llama",
    "llama3": "llama",
    "llama4": "llama",
    "code_llama": "llama",
    "codellama": "llama",
    "tinyllama": "llama",
    "smollm": "llama",
    "smollm2": "llama",
    "smollm3": "llama",
    "yi": "llama",
    "deepseek": "llama",
    "deepseek_v2": "llama",
    "deepseek_v3": "llama",
    "command_r": "llama",
    "cohere": "llama",
    "olmo": "llama",
    "olmo2": "llama",
    "stablelm": "llama",
    "starcoder": "llama",
    "starcoder2": "llama",
    "internlm": "llama",
    "internlm2": "llama",
    "baichuan": "llama",
    "chatglm": "llama",
    "mistral": "mistral",
    "mixtral": "mistral",
    "qwen2_moe": "qwen2",
    "qwen2": "qwen2",
    "qwen3": "qwen2",
    "qwen": "qwen2",
    "phi3": "phi3",
    "phi4": "phi3",
    "phi": "phi",
    "phi2": "phi",
    "gemma3": "gemma2",
    "gemma2": "gemma2",
    "gemma": "gemma",
    "falcon": "falcon",
}

PREQUANTIZED_NAME_MARKERS: tuple = (
    "-bnb-4bit",
    "-bnb-8bit",
    "-4bit",
    "-8bit",
    "-awq",
    "-gptq",
    "-int4",
    "-int8",
    "-fp8",
    "-eetq",
    "-hqq",
    "-aqlm",
)

GGUF_NAME_MARKERS: tuple = ("-gguf", ".gguf")


class ArchitectureMixin:
    """Architecture-resolution methods mixed into :class:`TurboModel`."""

    _architecture_registry: Dict[str, str] = {}
    _model_class_registry: Dict[str, Type[PreTrainedModel]] = {}

    @classmethod
    def register_architecture(
        cls,
        architecture: str,
        *,
        base_model_type: Optional[str] = None,
        model_class: Optional[Type[PreTrainedModel]] = None,
    ) -> None:
        """Register a new architecture alias and optional explicit model class.

        Args:
            architecture: Architecture or model-type name to register.
            base_model_type: Base model family to fall back to (e.g. ``"llama"``).
            model_class: Explicit model class providing ``from_pretrained()``.
        """
        normalized = architecture.lower().strip()
        if not normalized:
            raise ValueError("architecture must be a non-empty string")

        if base_model_type:
            cls._architecture_registry[normalized] = base_model_type.lower().strip()

        if model_class is not None:
            cls._model_class_registry[normalized] = model_class

    @classmethod
    def resolve_model_type(
        cls,
        model_name: str,
        *,
        config_model_type: Optional[str] = None,
        model_type_override: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a HuggingFace ``model_type`` to a known-loadable base family.

        Resolution order (first non-``None`` match wins):

        1. Explicit ``model_type_override`` from the caller.
        2. Exact match in :attr:`_architecture_registry` (user-registered alias).
        3. Exact match in :data:`DEFAULT_ARCHITECTURE_FALLBACKS`.
        4. Family-style match against the config's ``model_type``.
        5. Family-style match against the repository name.
        6. The original ``config_model_type`` unchanged, or ``None``.
        """
        if model_type_override:
            return model_type_override.lower().strip()

        model_type = (config_model_type or "").lower().strip()
        name = model_name.lower()

        if model_type and model_type in cls._architecture_registry:
            return cls._architecture_registry[model_type]

        if model_type and model_type in DEFAULT_ARCHITECTURE_FALLBACKS:
            return DEFAULT_ARCHITECTURE_FALLBACKS[model_type]

        if model_type:
            for pattern, fallback in cls._architecture_registry.items():
                if cls._matches_family(model_type, pattern):
                    return fallback
            for pattern, fallback in DEFAULT_ARCHITECTURE_FALLBACKS.items():
                if cls._matches_family(model_type, pattern):
                    return fallback

        for pattern, fallback in cls._architecture_registry.items():
            if cls._matches_model_name_pattern(name, pattern):
                return fallback
        for pattern, fallback in DEFAULT_ARCHITECTURE_FALLBACKS.items():
            if cls._matches_model_name_pattern(name, pattern):
                return fallback

        return model_type or None

    @classmethod
    def _matches_model_name_pattern(cls, model_name: str, pattern: str) -> bool:
        """Return True when ``pattern`` appears as a token in ``model_name``."""
        return cls._compiled_model_name_pattern(pattern).search(model_name) is not None

    @staticmethod
    @lru_cache(maxsize=None)
    def _compiled_model_name_pattern(pattern: str):
        """Compile and cache token-boundary regex for model-name matching."""
        escaped = re.escape(pattern)
        return re.compile(rf"(^|[^a-z0-9]){escaped}([^a-z0-9]|$)")

    @classmethod
    def _matches_family(cls, model_type: str, family: str) -> bool:
        """Return True when ``model_type`` belongs to ``family`` via version-suffix patterns."""
        if not model_type or not family:
            return False
        if model_type == family:
            return True
        return bool(cls._compiled_family_pattern(family).match(model_type))

    @staticmethod
    @lru_cache(maxsize=None)
    def _compiled_family_pattern(family: str):
        """Cache regex used by :meth:`_matches_family`."""
        return re.compile(rf"^{re.escape(family)}[\d_\-]")
