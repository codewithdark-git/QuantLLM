"""Model loading with automatic optimization for QuantLLM.

Provides the :class:`ModelLoaderMixin` used by :class:`TurboModel` for
the ``from_pretrained`` entry point.
"""

import copy
import importlib.util
import os
import tempfile
from dataclasses import asdict
from typing import Any, Dict, Optional, Type

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from ..utils.hf_compat import dtype_kwargs
from .architecture import (
    DEFAULT_ARCHITECTURE_FALLBACKS,
    GGUF_NAME_MARKERS,
    PREQUANTIZED_NAME_MARKERS,
)
from .smart_config import SmartConfig
from ..utils import (
    logger,
    print_header,
    print_info,
    print_success,
    print_warning,
    QuantLLMProgress,
)
from transformers.utils.logging import disable_progress_bar as disable_hf_progress_bar
from datasets.utils.logging import disable_progress_bar as disable_ds_progress_bar

DEFAULT_EXPORT_PUSH_CONFIG: Dict[str, Any] = {
    "format": "safetensors",
    "push_format": "safetensors",
    "quantization": "Q4_K_M",
    "push_quantization": None,
}


class ModelLoaderMixin:
    """Model-loading methods mixed into :class:`TurboModel`."""

    _architecture_registry: Any
    _model_class_registry: Any
    resolve_model_type: Any
    verbose: bool
    _has_runtime_quantization: Any
    _is_quantized_override: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        bits: Optional[int] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        quantize: bool = True,
        model_type_override: Optional[str] = None,
        base_model_fallback: bool = True,
        from_config_only: bool = False,
        config_override: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        token: Optional[str] = None,
        attn_implementation: Optional[str] = None,
    ) -> "TurboModel":  # type: ignore[name-defined]  # noqa: F821
        """Load a model with automatic optimization.

        This is the main entry point for QuantLLM. It automatically:
        - Detects your hardware capabilities
        - Chooses optimal quantization settings
        - Configures memory management
        - Enables speed optimizations

        ``attn_implementation`` overrides the automatic choice
        (``"flash_attention_2"`` when usable, otherwise ``"sdpa"``).
        """
        disable_hf_progress_bar()
        disable_ds_progress_bar()

        if verbose:
            print_header(f"Loading {model_name}")

        if cls._looks_like_gguf_repo(model_name):
            logger.warning(
                "Repository name '%s' looks like a GGUF-only repo. "
                "Use TurboModel.from_gguf(...) for GGUF weights; "
                "from_pretrained() is intended for standard transformers "
                "checkpoints (safetensors / pytorch_model.bin).",
                model_name,
            )

        prequant_marker = cls._looks_prequantized(model_name)
        if prequant_marker and verbose:
            logger.info(
                "Detected pre-quantized repo (marker '%s'); honouring the "
                "model's own quantization config and skipping dynamic "
                "BitsAndBytes quantization.",
                prequant_marker,
            )

        if verbose:
            logger.info("Detecting hardware and configuration...")

        # When quantize=False, force full precision (bits=16)
        effective_bits = 16 if not quantize else bits

        smart_config = SmartConfig.detect(
            model_name,
            bits=effective_bits,
            max_seq_length=max_length,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            token=token,
        )

        if config_override:
            if isinstance(config_override, SmartConfig):
                override_dict = asdict(config_override)
            else:
                override_dict = config_override
            for key, value in override_dict.items():
                if hasattr(smart_config, key):
                    setattr(smart_config, key, value)

        if verbose:
            smart_config.print_summary()

        if verbose:
            logger.info("Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            token=token,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if verbose:
            if smart_config.bits != smart_config.effective_loading_bits and smart_config.bits < 16:
                logger.info(
                    "Loading model (%s-bit, for %s-bit GGUF export)...",
                    smart_config.effective_loading_bits,
                    smart_config.bits,
                )
            else:
                logger.info("Loading model (%s-bit)...", smart_config.bits)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            **dtype_kwargs(smart_config.dtype),
        }

        if token is not None:
            model_kwargs["token"] = token

        hf_config = None

        try:
            hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code, token=token)

            existing_quant = getattr(hf_config, "quantization_config", None)
            if existing_quant:
                if verbose:
                    logger.warning(
                        "Model is already quantized (%s). Disabling dynamic quantization.",
                        existing_quant.__class__.__name__,
                    )
                quantize = False

        except Exception as config_error:
            logger.warning("Could not load config for '%s': %s", model_name, config_error)

        quantization_applied = False
        if cls._should_apply_quantization(quantize, smart_config.bits, from_config_only):
            quant_kwargs = cls._get_quantization_kwargs(smart_config)
            quantization_applied = bool(quant_kwargs)
            model_kwargs.update(quant_kwargs)

        if smart_config.cpu_offload:
            model_kwargs["device_map"] = "auto"
            model_kwargs["offload_folder"] = tempfile.mkdtemp(prefix="quantllm_offload_")
            max_memory = cls._offload_max_memory(smart_config.device)
            if max_memory:
                model_kwargs["max_memory"] = max_memory
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = {"": smart_config.device}

        auto_attn = attn_implementation is None
        model_kwargs["attn_implementation"] = attn_implementation or cls._choose_attn_implementation(smart_config)

        with QuantLLMProgress() as p:
            if verbose:
                task = p.add_task("Downloading & Loading model...", total=None)

            model = cls._load_model_with_fallback(
                model_name,
                model_kwargs,
                trust_remote_code=trust_remote_code,
                hf_config=hf_config,
                model_type_override=model_type_override,
                base_model_fallback=base_model_fallback,
                from_config_only=from_config_only,
                auto_attn_implementation=auto_attn,
            )

            if verbose:
                p.update(task, completed=100)

        if smart_config.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                if verbose:
                    logger.info("  Gradient checkpointing enabled")

        # torch.compile is opt-in (compile_model=True via config_override) and
        # never applied to quantized models. Only ``forward`` is compiled so the
        # model stays a PreTrainedModel (PEFT / isinstance checks keep working).
        if smart_config.compile_model:
            if smart_config.effective_loading_bits >= 16 and not quantization_applied and not from_config_only:
                try:
                    model.forward = torch.compile(model.forward)  # type: ignore[method-assign]
                    if verbose:
                        logger.info("  torch.compile enabled (model.forward)")
                except Exception as e:
                    if verbose:
                        print_warning(f"torch.compile failed: {e}")
            elif verbose:
                logger.info("  torch.compile skipped (quantized model)")

        if verbose:
            print_success("Model loaded successfully!")
            logger.info("")

        instance = cls(model, tokenizer, smart_config, export_push_config=config)
        instance.verbose = verbose

        actual_quantized = instance._has_runtime_quantization()
        if from_config_only:
            instance._is_quantized_override = bool(actual_quantized)
            if verbose:
                print_warning(
                    "from_config_only=True returned a model with random "
                    "weights and no quantization. Call model.load_weights(...) "
                    "or reload with from_config_only=False before using it "
                    "for inference."
                )
        else:
            wanted_quantization = cls._should_apply_quantization(quantize, smart_config.bits, from_config_only=False)
            if wanted_quantization and not actual_quantized:
                instance._is_quantized_override = False
                if verbose:
                    print_warning(
                        "Requested quantization was not applied at load time "
                        "(typically because bitsandbytes is not installed "
                        "or the model was already pre-quantized). Continuing "
                        "in full precision."
                    )

        return instance

    @staticmethod
    def _should_apply_quantization(
        quantize: bool,
        bits: int,
        from_config_only: bool,
    ) -> bool:
        """Decide whether BitsAndBytes kwargs should be added at load time."""
        return quantize and bits < 16 and not from_config_only

    @staticmethod
    def _choose_attn_implementation(smart_config: Any) -> str:
        """Pick ``flash_attention_2`` when installed and supported, else ``sdpa``."""
        dtype = getattr(smart_config, "dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, None)
        device = getattr(smart_config, "device", None)
        device_type = device.type if isinstance(device, torch.device) else str(device or "").split(":")[0]
        if (
            getattr(smart_config, "use_flash_attention", False)
            and device_type == "cuda"
            and dtype in (torch.float16, torch.bfloat16)
            and importlib.util.find_spec("flash_attn") is not None
        ):
            return "flash_attention_2"
        return "sdpa"

    @staticmethod
    def _offload_max_memory(device: Any) -> Optional[Dict[Any, str]]:
        """Build a ``max_memory`` map for CPU offload (~10% GPU headroom)."""
        if not torch.cuda.is_available():
            return None
        try:
            index = torch.device(device).index if device is not None else None
            if index is None:
                index = torch.cuda.current_device()
            free_bytes, _ = torch.cuda.mem_get_info(index)
        except Exception as e:
            logger.warning("Could not query free GPU memory for offloading: %s", e)
            return None
        max_memory: Dict[Any, str] = {index: f"{max(int(free_bytes * 0.9 / (1024**2)), 1)}MiB"}
        try:
            import psutil

            max_memory["cpu"] = f"{int(psutil.virtual_memory().available * 0.9 / (1024**2))}MiB"
        except ImportError:
            pass
        return max_memory

    @staticmethod
    def _is_architecture_error(error: BaseException) -> bool:
        """True when ``error`` means transformers did not recognise the architecture."""
        if not isinstance(error, (ValueError, KeyError)):
            return False
        message = str(error).lower()
        markers = (
            "model type",
            "model_type",
            "configuration class",
            "unrecognized",
            "does not recognize",
        )
        return any(marker in message for marker in markers)

    @staticmethod
    def _is_attn_implementation_error(error: BaseException) -> bool:
        """True when ``error`` is about an unsupported attention implementation."""
        message = str(error).lower()
        return isinstance(error, (ValueError, ImportError)) and (
            "attn_implementation" in message
            or "scaled_dot_product_attention" in message
            or "flash attention" in message
            or "flashattention" in message
        )

    @staticmethod
    def _looks_prequantized(model_name: str) -> Optional[str]:
        """Return a marker when the repo name suggests pre-quantization."""
        lowered = model_name.lower()
        for marker in PREQUANTIZED_NAME_MARKERS:
            if marker in lowered:
                return marker
        return None

    @staticmethod
    def _looks_like_gguf_repo(model_name: str) -> bool:
        """Heuristic: repo name looks like a GGUF-only weights repository."""
        lowered = model_name.lower()
        return any(marker in lowered for marker in GGUF_NAME_MARKERS)

    @classmethod
    def _load_model_with_fallback(
        cls,
        model_name: str,
        model_kwargs: Dict[str, Any],
        *,
        trust_remote_code: bool,
        hf_config: Optional[Any],
        model_type_override: Optional[str],
        base_model_fallback: bool,
        from_config_only: bool,
        auto_attn_implementation: bool = False,
    ) -> PreTrainedModel:
        """Load model with architecture fallback and optional config-only mode.

        Only unrecognised-architecture errors trigger the fallback chain;
        everything else (missing/gated repos, missing deps, OOM, ...) is
        re-raised immediately.
        """
        config_model_type = (getattr(hf_config, "model_type", None) or "").lower().strip()
        is_registered_architecture = config_model_type in cls._architecture_registry if config_model_type else False
        resolved_model_type = cls.resolve_model_type(
            model_name,
            config_model_type=getattr(hf_config, "model_type", None),
            model_type_override=model_type_override,
        )
        resolved_config = hf_config

        if hf_config is not None and resolved_model_type:
            current_model_type = getattr(hf_config, "model_type", None)
            if current_model_type != resolved_model_type:
                resolved_config = copy.deepcopy(hf_config)
                setattr(resolved_config, "model_type", resolved_model_type)

        if (
            trust_remote_code
            and config_model_type
            and not is_registered_architecture
            and config_model_type not in DEFAULT_ARCHITECTURE_FALLBACKS.values()
        ):
            logger.warning(
                "trust_remote_code=True is enabled for unregistered architecture '%s' "
                "(resolved fallback: '%s'). Only use this for models from trusted sources.",
                config_model_type,
                resolved_model_type,
            )

        if from_config_only:
            if resolved_config is None:
                raise ValueError(
                    "from_config_only=True requires a loadable config. "
                    "Try trust_remote_code=True or set model_type_override."
                )
            return AutoModelForCausalLM.from_config(
                resolved_config,
                trust_remote_code=trust_remote_code,
                **{k: v for k, v in model_kwargs.items() if k in ("dtype", "torch_dtype")},
            )

        try:
            try:
                return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            except (ValueError, ImportError) as attn_error:
                # The auto-selected attention backend is not supported by this
                # architecture: retry once with the transformers default.
                if not (
                    auto_attn_implementation
                    and "attn_implementation" in model_kwargs
                    and cls._is_attn_implementation_error(attn_error)
                ):
                    raise
                logger.warning(
                    "attn_implementation=%r not supported (%s); retrying with the transformers default.",
                    model_kwargs["attn_implementation"],
                    attn_error,
                )
                model_kwargs = dict(model_kwargs)
                model_kwargs.pop("attn_implementation")
                return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as primary_error:
            oom_error = getattr(torch.cuda, "OutOfMemoryError", None)
            if isinstance(primary_error, (OSError, ImportError)) or (
                oom_error is not None and isinstance(primary_error, oom_error)
            ):
                raise
            if not base_model_fallback or not cls._is_architecture_error(primary_error):
                raise
            fallback_error: Optional[BaseException] = None

            if resolved_config is not None:
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs["config"] = resolved_config
                try:
                    return AutoModelForCausalLM.from_pretrained(model_name, **fallback_kwargs)
                except Exception as fallback_config_error:
                    fallback_error = fallback_config_error

            registered_cls: Optional[Type[PreTrainedModel]] = None
            if config_model_type:
                registered_cls = cls._model_class_registry.get(config_model_type)
            if registered_cls is None and resolved_model_type:
                registered_cls = cls._model_class_registry.get(resolved_model_type)
            if registered_cls is not None:
                class_kwargs = dict(model_kwargs)
                if resolved_config is not None:
                    class_kwargs["config"] = resolved_config
                try:
                    return registered_cls.from_pretrained(model_name, **class_kwargs)
                except Exception as fallback_registered_error:
                    fallback_error = fallback_registered_error

            error_details = f"\nPrimary error: {type(primary_error).__name__}: {primary_error}"
            if fallback_error:
                error_details += f"\nLast fallback error: {type(fallback_error).__name__}: {fallback_error}"
            architecture_label = config_model_type or "<unknown>"
            resolved_label = resolved_model_type or "<none>"

            raise RuntimeError(
                "Failed to load model with AutoModelForCausalLM and fallback resolution.\n"
                f"Architecture '{architecture_label}' resolved to base model type '{resolved_label}'.\n"
                "Try one of:\n"
                f"1) register_architecture('{architecture_label}', base_model_type='llama').\n"
                "2) Use model_type_override='llama' (or your compatible base family).\n"
                "3) Use from_config_only=True with a loadable config "
                "(usually trust_remote_code=True)." + error_details
            ) from primary_error

    @staticmethod
    def _get_quantization_kwargs(config: SmartConfig) -> Dict[str, Any]:
        """Get kwargs for quantized model loading.

        BitsAndBytes only supports 4-bit and 8-bit for model loading.
        Other bit widths (2, 3, 5, 6) are available during GGUF export.
        Returns ``{}`` (full precision) when bitsandbytes or a CUDA/XPU
        device is unavailable.
        """
        if importlib.util.find_spec("bitsandbytes") is None:
            logger.warning(
                "bitsandbytes is not installed; falling back to full "
                "precision. Install with pip install bitsandbytes to "
                "enable 4-bit / 8-bit quantization on CUDA."
            )
            return {}

        xpu = getattr(torch, "xpu", None)
        has_accelerator = torch.cuda.is_available() or bool(xpu is not None and xpu.is_available())
        if not has_accelerator:
            logger.warning(
                "BitsAndBytes quantization requires a CUDA (or XPU) device; "
                "none was found, so the model is loaded in full precision."
            )
            return {}

        cpu_offload = bool(getattr(config, "cpu_offload", False))

        try:
            from transformers import BitsAndBytesConfig

            if config.bits <= 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=config.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=cpu_offload,
                )
                if config.bits < 4:
                    logger.info(
                        "  BitsAndBytes supports 4/8-bit only. Using 4-bit for requested %s-bit.",
                        config.bits,
                    )
                    logger.info("     Tip: Export to GGUF for Q%s_K quantization!", config.bits)
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=cpu_offload,
                )
                if config.bits != 8:
                    logger.info(
                        "  BitsAndBytes supports 4/8-bit only. Using 8-bit for requested %s-bit.",
                        config.bits,
                    )
                    logger.info("     Tip: Export to GGUF for Q%s_K quantization!", config.bits)

            return {"quantization_config": quantization_config}

        except ImportError:
            logger.warning(
                "bitsandbytes is not installed; falling back to full "
                "precision. Install with pip install bitsandbytes to "
                "enable 4-bit / 8-bit quantization on CUDA."
            )
            return {}

    @staticmethod
    def _build_export_push_config(
        config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build shared export/push config with deterministic defaults."""
        resolved = dict(DEFAULT_EXPORT_PUSH_CONFIG)
        if config:
            aliases = {
                "export_format": "format",
                "export_quantization": "quantization",
            }
            nullable_overrides = {"push_quantization"}
            for key, value in config.items():
                mapped_key = aliases.get(key, key)
                if mapped_key in resolved and (value is not None or mapped_key in nullable_overrides):
                    resolved[mapped_key] = value

            if "format" in config and "push_format" not in config:
                resolved["push_format"] = resolved["format"]
            if "quantization" in config and "push_quantization" not in config:
                resolved["push_quantization"] = resolved["quantization"]

        return resolved
