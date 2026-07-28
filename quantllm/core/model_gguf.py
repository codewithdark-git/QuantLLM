"""GGUF model loading support for QuantLLM.

Provides the :class:`ModelGGUFMixin` used by :class:`TurboModel` for
the ``from_gguf`` entry point.
"""

import os
from typing import Any, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .smart_config import SmartConfig
from ..utils import (
    logger,
    print_header,
    print_info,
    print_success,
    print_warning,
    QuantLLMProgress,
)


class ModelGGUFMixin:
    """GGUF-loading methods mixed into :class:`TurboModel`."""

    _is_quantized_override: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def from_gguf(
        cls,
        model_id: str,
        filename: Optional[str] = None,
        *,
        device: Optional[str] = None,
        verbose: bool = True,
        **kwargs,
    ) -> "TurboModel":  # type: ignore[name-defined]  # noqa: F821
        """Load a GGUF model directly from HuggingFace Hub or local path.

        Uses transformers' native GGUF support (requires transformers>=4.36.0).
        """
        if verbose:
            print_header(f"Loading GGUF: {model_id}")

        try:
            import gguf

            gguf_version = getattr(gguf, "__version__", "unknown")
            if verbose:
                print_info(f"Using gguf version: {gguf_version}")
        except ImportError:
            raise ImportError("Loading GGUF models requires the 'gguf' package.\nPlease run: pip install gguf>=0.10.0")

        if filename is None:
            if verbose:
                print_info("No filename specified, searching for GGUF files...")
            try:
                available_files = cls.list_gguf_files(model_id)
                if available_files:
                    q4_files = [f for f in available_files if "q4_k_m" in f.lower()]
                    filename = q4_files[0] if q4_files else available_files[0]
                    if verbose:
                        print_info(f"Found {len(available_files)} GGUF files, using: {filename}")
            except Exception as e:
                if verbose:
                    print_warning(f"Could not list GGUF files: {e}")

        if verbose and filename:
            print_info(f"Loading: {filename}")

        smart_config = SmartConfig.detect(model_id, device=device)
        smart_config.quant_type = "GGUF"

        with QuantLLMProgress() as progress:
            if verbose:
                task = progress.add_task("Loading GGUF model...", total=None)

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    gguf_file=filename,
                    torch_dtype=smart_config.dtype,
                    trust_remote_code=True,
                    **kwargs,
                )
            except ImportError as e:
                if "gguf" in str(e).lower():
                    raise ImportError(
                        "transformers requires a newer version of 'gguf'.\n"
                        "Please run: pip install --upgrade gguf>=0.10.0"
                    ) from e
                raise
            except Exception as e:
                if "gguf" in str(e).lower() and "not found" in str(e).lower():
                    available = cls.list_gguf_files(model_id)
                    raise FileNotFoundError(
                        f"GGUF file '{filename}' not found in {model_id}.\nAvailable files: {available}"
                    ) from e
                raise

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
            except Exception:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                except Exception:
                    if verbose:
                        print_warning("Could not load tokenizer from GGUF repo, using default")
                    tokenizer = None

            if verbose:
                print_success("GGUF Model loaded successfully!")
                if hasattr(model, "num_parameters"):
                    params = model.num_parameters() / 1e9
                    print_info(f"Parameters: {params:.2f}B")

        instance = cls(model, tokenizer, smart_config, verbose=verbose)
        instance._is_quantized_override = True
        return instance

    @staticmethod
    def list_gguf_files(model_id: str) -> List[str]:
        """List available GGUF files in a HuggingFace repository."""
        try:
            from huggingface_hub import list_repo_files

            all_files = list_repo_files(model_id)
            gguf_files = [f for f in all_files if f.endswith(".gguf")]

            def quant_sort_key(name):
                name_lower = name.lower()
                if "f32" in name_lower:
                    return 0
                if "f16" in name_lower:
                    return 1
                if "q8" in name_lower:
                    return 2
                if "q6" in name_lower:
                    return 3
                if "q5_k_m" in name_lower:
                    return 4
                if "q5_k_s" in name_lower:
                    return 5
                if "q4_k_m" in name_lower:
                    return 6
                if "q4_k_s" in name_lower:
                    return 7
                if "q3_k" in name_lower:
                    return 8
                if "q2_k" in name_lower:
                    return 9
                return 10

            return sorted(gguf_files, key=quant_sort_key)

        except Exception as e:
            if os.path.isdir(model_id):
                return [f for f in os.listdir(model_id) if f.endswith(".gguf")]
            raise ValueError(f"Could not list GGUF files from {model_id}: {e}")
