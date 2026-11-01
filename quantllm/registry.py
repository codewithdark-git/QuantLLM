"""
QuantLLM Model Registry -- dynamic model registry on HuggingFace Hub.

Supports:
    - Listing models under the QuantLLM HF organization
    - Searching models by name, family, or parameters
    - Loading models by name
    - Registering user models under the QuantLLM org
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

QUANTLLM_ORG = "QuantLLM"
FAMILY_KEY = "quantllm_family"

logger = logging.getLogger("quantllm.registry")

# Process-wide cache of Hub results, keyed by organization.
_HUB_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _hub_offline() -> bool:
    """Return True when HF_HUB_OFFLINE is set."""
    try:
        from huggingface_hub import constants

        if getattr(constants, "HF_HUB_OFFLINE", False):
            return True
    except ImportError:
        pass
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _model_id(m: Any) -> Optional[str]:
    """Return the repo id of a ModelInfo (``id`` in hub>=1.0, ``modelId`` before)."""
    return getattr(m, "id", None) or getattr(m, "modelId", None)


def _card_data_dict(m: Any) -> Dict[str, Any]:
    card_data = getattr(m, "card_data", None) or getattr(m, "cardData", None) or {}
    if hasattr(card_data, "to_dict"):
        card_data = card_data.to_dict()
    return card_data if isinstance(card_data, dict) else {}


def _parse_family(card_data: Dict[str, Any]) -> str:
    """Read the model family from card data.

    Prefers the dedicated ``quantllm_family`` key, then falls back to the
    first card tag that is not the generic ``quantllm`` tag.
    """
    family = card_data.get(FAMILY_KEY)
    if isinstance(family, str) and family:
        return family.lower()
    tags = card_data.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag and tag.lower() != "quantllm":
                return tag.lower()
    return "unknown"


class ModelRegistry:
    """Dynamic model registry backed by HuggingFace Hub.

    Reads model metadata from the QuantLLM HF organization. Falls back
    to a built-in curated list when the Hub is unreachable.
    """

    def __init__(self, org: str = QUANTLLM_ORG):
        self.org = org
        self._models: Optional[List[Dict[str, Any]]] = None

    def _fetch_from_hub(self) -> List[Dict[str, Any]]:
        """Fetch model list from the QuantLLM HF organization.

        Uses a single ``list_models`` call with card data, honours
        ``HF_HUB_OFFLINE`` and caches results in memory for the process.
        """
        if self.org in _HUB_CACHE:
            return [dict(m) for m in _HUB_CACHE[self.org]]
        if _hub_offline():
            return []
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return []

        try:
            api = HfApi()
            models = api.list_models(author=self.org, sort="downloads", limit=50, cardData=True)
            result = []
            for m in models:
                model_id = _model_id(m)
                if not model_id:
                    continue
                card_data = _card_data_dict(m)
                result.append(
                    {
                        "name": model_id,
                        "family": _parse_family(card_data),
                        "params": card_data.get("params", 0) or 0,
                        "formats": self._detect_formats(model_id),
                        "quality_score": card_data.get("quality_score", 3.0) or 3.0,
                        "speed_score": card_data.get("speed_score", 3.0) or 3.0,
                        "verified": card_data.get("verified", False) or False,
                        "recommended": card_data.get("recommended_quant", "Q4_K_M") or "Q4_K_M",
                    }
                )
        except Exception as e:  # network / auth / API errors -> fall back to built-in list
            logger.debug("Failed to fetch QuantLLM registry from Hub: %s", e)
            return []
        _HUB_CACHE[self.org] = [dict(m) for m in result]
        return result

    def _detect_formats(self, model_id: str) -> List[str]:
        mid = model_id.lower()
        formats = []
        if "-gguf" in mid or ".gguf" in mid:
            formats.append("gguf")
        if "-safetensors" in mid or "-hf" in mid:
            formats.append("safetensors")
        if "-mlx" in mid:
            formats.append("mlx")
        if not formats:
            formats.append("safetensors")
        return formats

    def _load_builtin(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "QuantLLM/TinyLlama-1.1B-GGUF",
                "family": "tinyllama",
                "params": 1.1,
                "formats": ["gguf"],
                "quality_score": 3.2,
                "speed_score": 5.0,
                "verified": True,
                "recommended": "Q4_K_M",
            },
            {
                "name": "QuantLLM/Llama-3.2-3B-GGUF",
                "family": "llama",
                "params": 3.2,
                "formats": ["gguf"],
                "quality_score": 4.5,
                "speed_score": 4.2,
                "verified": True,
                "recommended": "Q4_K_M",
            },
            {
                "name": "QuantLLM/Mistral-7B-GGUF",
                "family": "mistral",
                "params": 7.0,
                "formats": ["gguf"],
                "quality_score": 4.3,
                "speed_score": 4.0,
                "verified": True,
                "recommended": "Q4_K_M",
            },
            {
                "name": "QuantLLM/Phi-3-mini-4k-instruct",
                "family": "phi",
                "params": 3.8,
                "formats": ["safetensors"],
                "quality_score": 4.2,
                "speed_score": 4.3,
                "verified": True,
                "recommended": "4bit",
            },
        ]

    @property
    def models(self) -> List[Dict[str, Any]]:
        if self._models is None:
            hub_models = self._fetch_from_hub()
            builtin = self._load_builtin()
            if hub_models:
                # Merge hub models with built-in, preferring hub versions but keeping
                # built-in as fallback for models not on hub
                hub_names = {m["name"].lower() for m in hub_models}
                for bm in builtin:
                    if bm["name"].lower() not in hub_names:
                        hub_models.append(bm)
                self._models = hub_models
            else:
                self._models = builtin
        return self._models

    def list_models(self, family: Optional[str] = None, min_quality: float = 0.0) -> List[Dict[str, Any]]:
        results = list(self.models)
        if family:
            results = [m for m in results if m["family"] == family.lower()]
        if min_quality > 0:
            results = [m for m in results if m["quality_score"] >= min_quality]
        return results

    def search(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        results = []
        for m in self.models:
            if q in m["name"].lower() or q in m["family"].lower():
                results.append(m)
                continue
            try:
                params = float(q.replace("b", ""))
                if abs(m["params"] - params) <= params * 0.3:
                    results.append(m)
            except ValueError:
                pass
        return results

    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        for m in self.models:
            if m["name"].lower() == name.lower():
                return m
        return None

    def register_model(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """Register a model by creating/updating its HF model card.

        If the repo already has a README, only its YAML metadata is updated
        (the card body is preserved).
        """
        try:
            from huggingface_hub import HfApi, ModelCard, metadata_update

            api = HfApi()
            family = str(metadata.get("family", "llama")).lower()
            card_data: Dict[str, Any] = {
                FAMILY_KEY: family,
                "params": metadata.get("params", 0),
                "quality_score": metadata.get("quality_score", 3.0),
                "speed_score": metadata.get("speed_score", 3.0),
                "verified": metadata.get("verified", False),
                "recommended_quant": metadata.get("recommended", "Q4_K_M"),
            }
            api.create_repo(model_id, exist_ok=True, private=False)

            if api.file_exists(model_id, "README.md"):
                existing = ModelCard.load(model_id)
                tags = list(getattr(existing.data, "tags", None) or [])
                for tag in ("quantllm", family):
                    if tag not in tags:
                        tags.append(tag)
                card_data["tags"] = tags
                metadata_update(model_id, card_data, overwrite=True)
            else:
                card_data["tags"] = ["quantllm", family]
                card = f"---\n{json.dumps(card_data, indent=2)}\n---\n# {model_id}\n\nQuantLLM-registered model."
                api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=model_id)
            self._models = None
            _HUB_CACHE.pop(self.org, None)
            return True
        except Exception as e:
            try:
                from quantllm.utils import print_warning

                print_warning(f"Failed to register model: {e}")
            except ImportError:
                print(f"Warning: Failed to register model: {e}")
            return False

    def format_list(self, models: Optional[List[Dict[str, Any]]] = None) -> str:
        items = models if models is not None else self.models
        if not items:
            return "No models found."
        lines = [
            "=" * 110,
            f"  {'Model':<55} {'Family':<12} {'Params':<8} {'Quality':<9} {'Speed':<7} {'Verified':<9}",
            f"  {'-' * 55} {'-' * 12} {'-' * 8} {'-' * 9} {'-' * 7} {'-' * 9}",
        ]
        for m in items:
            lines.append(
                f"  {m['name'][:53]:<55} {m['family']:<12} {m['params']:<8.1f} "
                f"{m['quality_score']:<9.1f} {m['speed_score']:<7.1f} {'Yes' if m.get('verified') else 'No':<9}"
            )
        lines.append("=" * 110)
        return "\n".join(lines)

    def format_search(self, query: str) -> str:
        results = self.search(query)
        return f"No models found matching '{query}'." if not results else self.format_list(results)


_registry = ModelRegistry()


def list_models(family: Optional[str] = None) -> str:
    return _registry.format_list(_registry.list_models(family=family))


def search_models(query: str) -> str:
    return _registry.format_search(query)


def model_info(name: str) -> str:
    m = _registry.get_model(name)
    if not m:
        return f"Model '{name}' not found."
    return (
        f"  Model:        {m['name']}\n"
        f"  Family:       {m['family']}\n"
        f"  Params:       {m['params']}B\n"
        f"  Formats:      {', '.join(m['formats'])}\n"
        f"  Recommended:  {m['recommended']}\n"
        f"  Quality:      {m['quality_score']}/5.0\n"
        f"  Speed:        {m['speed_score']}/5.0\n"
        f"  Verified:     {'Yes' if m.get('verified') else 'No'}"
    )


def register_model(
    model_id: str,
    family: str = "llama",
    params: float = 0.0,
    quality_score: float = 3.0,
    speed_score: float = 3.0,
    **kwargs,
) -> bool:
    metadata = {
        "family": family,
        "params": params,
        "quality_score": quality_score,
        "speed_score": speed_score,
        **kwargs,
    }
    return _registry.register_model(model_id, metadata)
