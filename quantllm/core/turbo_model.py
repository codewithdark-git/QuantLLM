"""
TurboModel - The Ultra-Simple QuantLLM API.

Load, quantize, fine-tune, and export LLMs with one line each.
"""

import os
import re
import shutil
import tempfile
import copy
from functools import lru_cache
from typing import Optional, Dict, Any, Union, List, Type
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
)

from .smart_config import SmartConfig
from .hardware import HardwareProfiler
from ..utils import logger, print_header, print_success, print_error, print_info, print_warning, QuantLLMProgress
from transformers.utils.logging import disable_progress_bar as disable_hf_progress_bar
from datasets.utils.logging import disable_progress_bar as disable_ds_progress_bar
from .memory import memory_optimized_tensor_order

DEFAULT_CHUNKED_SHARD_SIZE = "2GB"
DEFAULT_EXPORT_PUSH_CONFIG = {
    "format": "safetensors",
    "push_format": "safetensors",
    "quantization": "Q4_K_M",
    "push_quantization": None,
}
# Default mapping of HuggingFace ``config.model_type`` values (or model-name
# tokens) to a known-loadable base family. Used as a best-effort fallback for
# brand-new architectures that ``transformers`` does not yet recognize. The
# mapping is consulted only when the user has not registered an explicit
# fallback via :func:`register_architecture`.
#
# Order matters: more specific patterns must come before more generic ones
# (e.g. ``qwen2_moe`` before ``qwen``).
DEFAULT_ARCHITECTURE_FALLBACKS: Dict[str, str] = {
    # Llama family and direct derivatives
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
    # Mistral / Mixtral
    "mistral": "mistral",
    "mixtral": "mistral",
    # Qwen family (note: qwen2_moe must come before qwen)
    "qwen2_moe": "qwen2",
    "qwen2": "qwen2",
    "qwen3": "qwen2",
    "qwen": "qwen2",
    # Phi family
    "phi3": "phi3",
    "phi4": "phi3",
    "phi": "phi",
    "phi2": "phi",
    # Gemma family
    "gemma3": "gemma2",
    "gemma2": "gemma2",
    "gemma": "gemma",
    # Falcon
    "falcon": "falcon",
}

# Substring markers in HF repo names that indicate the model is already
# pre-quantized at rest. When detected, QuantLLM should let ``transformers``
# load the existing quantized weights instead of dynamically applying its own
# BitsAndBytes quantization on top.
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

# Markers in HF repo names indicating GGUF-only repositories. Loading these
# via :meth:`TurboModel.from_pretrained` (instead of ``from_gguf``) is almost
# always a user mistake; we surface a helpful hint.
GGUF_NAME_MARKERS: tuple = ("-gguf", ".gguf")


class TurboModel:
    """
    High-performance LLM with the simplest possible API.
    
    Features:
        - One-line loading with automatic quantization
        - One-line fine-tuning with LoRA
        - One-line export to multiple formats
        - Automatic hardware optimization
    
    Example:
        >>> # Load any model in one line
        >>> model = TurboModel.from_pretrained("meta-llama/Llama-3-8B")
        >>> 
        >>> # Generate text
        >>> response = model.generate("Explain quantum computing")
        >>> 
        >>> # Fine-tune with your data
        >>> model.finetune("my_data.json", epochs=3)
        >>> 
        >>> # Export to GGUF
        >>> model.export("gguf", "my_model.gguf")
    """
    
    _architecture_registry: Dict[str, str] = {}
    _model_class_registry: Dict[str, Type[PreTrainedModel]] = {}
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: SmartConfig,
        export_push_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        """
        Initialize TurboModel. Use from_pretrained() instead of direct init.
        
        Args:
            model: The loaded/quantized model
            tokenizer: Associated tokenizer
            config: Configuration used for this model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        # ``_is_quantized_override`` is consulted by :pyattr:`is_quantized`
        # *only* when the caller explicitly asserts a quantization state
        # (e.g. :meth:`from_gguf` knows GGUF is always quantized). When
        # ``None`` the property derives the answer from the loaded model.
        self._is_quantized_override: Optional[bool] = None
        self._is_finetuned = False
        self._lora_applied = False
        self.export_push_config = self._build_export_push_config(export_push_config)
        self.verbose = verbose

    def save(self, output_dir: str, safe_serialization: bool = True) -> str:
        """
        Save the model and tokenizer to a local directory.
        
        Args:
            output_dir: Directory to save the model to
            safe_serialization: Use safetensors format (default: True)
            
        Returns:
            Path to saved model directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Merge LoRA if applied
        model_to_save = self.model
        if self._lora_applied:
            if self.verbose:
                print_info("Merging LoRA weights before saving...")
            model_to_save = self.model.merge_and_unload()
            self._lora_applied = False
            self.model = model_to_save
        
        model_to_save.save_pretrained(output_dir, safe_serialization=safe_serialization)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        
        if self.verbose:
            print_success(f"Model saved to {output_dir}")
        
        return output_dir


    @classmethod
    def register_architecture(
        cls,
        architecture: str,
        *,
        base_model_type: Optional[str] = None,
        model_class: Optional[Type[PreTrainedModel]] = None,
    ) -> None:
        """
        Register a new architecture alias and optional explicit model class.
        
        Args:
            architecture: Architecture or model type name to register
            base_model_type: Base model family to fall back to (e.g., "llama")
            model_class: Explicit model class with from_pretrained()
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
        """
        Resolve a HuggingFace ``model_type`` to a known-loadable base family.

        Resolution order (first non-``None`` match wins):

        1. Explicit ``model_type_override`` from the caller.
        2. Exact match in :attr:`_architecture_registry` (user-registered alias).
        3. Exact match in :data:`DEFAULT_ARCHITECTURE_FALLBACKS`.
        4. Family-style match against the config's ``model_type`` (e.g.
           ``qwen3`` -> ``qwen``).
        5. Family-style match against the repository name (e.g.
           ``Qwen/Qwen3-8B`` -> ``qwen``).
        6. The original ``config_model_type`` unchanged, or ``None`` when no
           config was loadable.

        The function never raises; callers are expected to handle ``None``.
        """
        if model_type_override:
            return model_type_override.lower().strip()

        model_type = (config_model_type or "").lower().strip()
        name = model_name.lower()

        # 2. Exact registry hit.
        if model_type and model_type in cls._architecture_registry:
            return cls._architecture_registry[model_type]

        # 3. Exact default-fallback hit.
        if model_type and model_type in DEFAULT_ARCHITECTURE_FALLBACKS:
            return DEFAULT_ARCHITECTURE_FALLBACKS[model_type]

        # 4. Family-style match against model_type itself (qwen3 -> qwen).
        if model_type:
            for pattern, fallback in cls._architecture_registry.items():
                if cls._matches_family(model_type, pattern):
                    return fallback
            for pattern, fallback in DEFAULT_ARCHITECTURE_FALLBACKS.items():
                if cls._matches_family(model_type, pattern):
                    return fallback

        # 5. Token-boundary match against the repo name.
        for pattern, fallback in cls._architecture_registry.items():
            if cls._matches_model_name_pattern(name, pattern):
                return fallback
        for pattern, fallback in DEFAULT_ARCHITECTURE_FALLBACKS.items():
            if cls._matches_model_name_pattern(name, pattern):
                return fallback

        # 6. Nothing matched.
        return model_type or None

    @classmethod
    def _matches_model_name_pattern(cls, model_name: str, pattern: str) -> bool:
        """Return True when ``pattern`` appears as a token in ``model_name``."""
        return cls._compiled_model_name_pattern(pattern).search(model_name) is not None

    @staticmethod
    @lru_cache(maxsize=None)
    def _compiled_model_name_pattern(pattern: str):
        """Compile and cache token-boundary regex patterns for model-name matching."""
        escaped = re.escape(pattern)
        # Match architecture tokens as standalone chunks split by separators.
        return re.compile(rf"(^|[^a-z0-9]){escaped}([^a-z0-9]|$)")

    @classmethod
    def _matches_family(cls, model_type: str, family: str) -> bool:
        """
        Decide whether ``model_type`` belongs to ``family``.

        Recognises common version-suffix patterns used by HuggingFace, e.g.
        ``qwen2``, ``qwen2_5``, ``qwen-2``, ``qwen3`` all match family ``qwen``.
        Plain prefix matches (``llamafication``) are intentionally rejected;
        only digit / underscore / dash separators count as family suffixes.
        """
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

    @staticmethod
    def _should_apply_quantization(
        quantize: bool,
        bits: int,
        from_config_only: bool,
    ) -> bool:
        """Decide whether ``BitsAndBytes`` kwargs should be added at load time.

        Returns False whenever the model is being constructed from config only
        (no weights to quantize) or whenever the user explicitly disabled
        quantization or asked for full precision.
        """
        return quantize and bits < 16 and not from_config_only

    @staticmethod
    def _looks_prequantized(model_name: str) -> Optional[str]:
        """
        Return a marker (e.g. ``"-bnb-4bit"``) when the repo name suggests it
        is already pre-quantized at rest. ``None`` otherwise.
        """
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
    ) -> PreTrainedModel:
        """Load model with architecture fallback and optional config-only mode."""
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
                torch_dtype=model_kwargs.get("torch_dtype"),
            )
        
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as primary_error:
            if not base_model_fallback:
                raise
            fallback_error = None
            # Fallback priority: resolved config model_type -> explicitly registered model class.
            if resolved_config is not None:
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs["config"] = resolved_config
                try:
                    return AutoModelForCausalLM.from_pretrained(model_name, **fallback_kwargs)
                except Exception as fallback_config_error:
                    fallback_error = fallback_config_error
            
            # Look up an explicit user-registered model class. Try the
            # original ``config.model_type`` first (most natural API:
            # ``register_architecture("newmodel", model_class=NewModel)``)
            # and fall back to the resolved base family for users that prefer
            # to register a class under the family name.
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
            
            error_details = f" Last fallback error: {fallback_error}" if fallback_error else ""
            architecture_label = config_model_type or "<unknown>"
            resolved_label = resolved_model_type or "<none>"
            
            raise RuntimeError(
                "Failed to load model with AutoModelForCausalLM and fallback resolution.\n"
                f"Architecture '{architecture_label}' resolved to base model type '{resolved_label}'.\n"
                "Try one of:\n"
                f"1) register_architecture('{architecture_label}', base_model_type='llama').\n"
                "2) Use model_type_override='llama' (or your compatible base family).\n"
                "3) Use from_config_only=True with a loadable config "
                "(usually trust_remote_code=True)."
                + error_details
            ) from (fallback_error or primary_error)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        # Simple overrides (all optional with smart defaults)
        bits: Optional[int] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        # Advanced options
        trust_remote_code: bool = True,
        quantize: bool = True,
        model_type_override: Optional[str] = None,
        base_model_fallback: bool = True,
        from_config_only: bool = False,
        config_override: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> "TurboModel":
        """
        Load a model with automatic optimization.
        
        This is the main entry point for QuantLLM. It automatically:
        - Detects your hardware capabilities
        - Chooses optimal quantization settings
        - Configures memory management
        - Enables speed optimizations
        
        Args:
            model_name: HuggingFace model name or local path
            bits: Override quantization bits (default: auto-detect)
            max_length: Override max sequence length (default: from model)
            device: Override device (default: best available GPU)
            dtype: Override dtype (default: bf16 if available, else fp16)
            trust_remote_code: Trust remote code in model
            quantize: Whether to quantize the model
            model_type_override: Override detected model_type for very new architectures
            base_model_fallback: Retry loading with resolved base model config on failure
            from_config_only: Build model from config only (without loading weights)
            config_override: Dict to override any auto-detected settings
            config: Shared export/push config (format, quantization, push_format, etc.)
            verbose: Print loading progress
            
        Returns:
            TurboModel ready for inference or fine-tuning
            
        Example:
            >>> # Simplest usage - everything automatic
            >>> model = TurboModel.from_pretrained("meta-llama/Llama-3-8B")
            >>> 
            >>> # With specific bits
            >>> model = TurboModel.from_pretrained("mistral-7b", bits=4)
            >>>
            >>> # For long context
            >>> model = TurboModel.from_pretrained("Qwen2-72B", max_length=32768)
        """
        # Disable default progress bars
        disable_hf_progress_bar()
        disable_ds_progress_bar()

        if verbose:
            print_header(f"Loading {model_name}")

        # Friendly hint when a user accidentally points ``from_pretrained`` at
        # a GGUF repository. ``transformers`` *can* load some GGUF repos via
        # ``from_pretrained`` with a ``gguf_file`` arg, but the dedicated
        # :meth:`from_gguf` path handles tokenizer fall-back and version
        # validation more safely.
        if cls._looks_like_gguf_repo(model_name):
            logger.warning(
                "Repository name '%s' looks like a GGUF-only repo. "
                "Use TurboModel.from_gguf(...) for GGUF weights; "
                "from_pretrained() is intended for standard transformers "
                "checkpoints (safetensors / pytorch_model.bin).",
                model_name,
            )

        # Friendly hint when the repo name advertises pre-quantization. We
        # still attempt to load it: ``transformers`` honours the embedded
        # ``quantization_config`` automatically.
        prequant_marker = cls._looks_prequantized(model_name)
        if prequant_marker and verbose:
            logger.info(
                "Detected pre-quantized repo (marker '%s'); honouring the "
                "model's own quantization config and skipping dynamic "
                "BitsAndBytes quantization.",
                prequant_marker,
            )

        # Auto-configure everything
        if verbose:
            logger.info("🚀 Detecting hardware and configuration...")

        smart_config = SmartConfig.detect(
            model_name,
            bits=bits,
            max_seq_length=max_length,
            device=device,
            dtype=dtype,
        )
        
        from dataclasses import asdict
        
        # Apply user overrides
        if config_override:
            # Handle SmartConfig objects
            if isinstance(config_override, SmartConfig):
                override_dict = asdict(config_override)
            else:
                override_dict = config_override
                
            for key, value in override_dict.items():
                if hasattr(smart_config, key):
                    setattr(smart_config, key, value)
        
        if verbose:
            smart_config.print_summary()
        
        # Load tokenizer
        if verbose:
            logger.info("📝 Loading tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model with optimizations
        if verbose:
            if smart_config.bits != smart_config.effective_loading_bits and smart_config.bits < 16:
                logger.info(f"📦 Loading model ({smart_config.effective_loading_bits}-bit, for {smart_config.bits}-bit GGUF export)...")
            else:
                logger.info(f"📦 Loading model ({smart_config.bits}-bit)...")
        
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": smart_config.dtype,
        }
        
        hf_config = None
        
        # Check if model is already quantized to prevent conflicts
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            
            existing_quant = getattr(hf_config, "quantization_config", None)
            if existing_quant:
                allow_requantize = False
                
                # Allow 8-bit -> 4-bit re-quantization (if B&B)
                is_bnb = "BitsAndBytesConfig" in existing_quant.__class__.__name__
                is_8bit = getattr(existing_quant, "load_in_8bit", False)
                
                if is_bnb and is_8bit and smart_config.bits == 4:
                    allow_requantize = True
                    if verbose:
                        logger.info("  ℹ️ Re-quantizing 8-bit model to 4-bit")
                
                if not allow_requantize:
                    if verbose:
                        logger.warning(f"⚠️ Model is already quantized ({existing_quant.__class__.__name__}). Disabling dynamic quantization.")
                    quantize = False
                
        except Exception:
            pass # Ignore config loading errors, proceed with defaults

        # Apply quantization if requested
        if cls._should_apply_quantization(quantize, smart_config.bits, from_config_only):
            model_kwargs.update(cls._get_quantization_kwargs(smart_config))
        
        # Device map for memory management
        if smart_config.cpu_offload:
            model_kwargs["device_map"] = "auto"
            model_kwargs["offload_folder"] = "offload"
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = {"": smart_config.device}
        
        # Load the model with progress spinner
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
            )
            
            if verbose:
                p.update(task, completed=100)
        
        # Apply additional optimizations
        if smart_config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                if verbose:
                    logger.info("  ✓ Gradient checkpointing enabled")
        
        # Enable Flash Attention if available
        if smart_config.use_flash_attention:
            cls._enable_flash_attention(model, verbose)
        
        # Compile model if beneficial
        if smart_config.compile_model:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                if verbose:
                    logger.info("  ✓ torch.compile enabled")
            except Exception as e:
                if verbose:
                    print_warning(f"torch.compile failed: {e}")
        
        if verbose:
            print_success("Model loaded successfully!")
            logger.info("")
        
        instance = cls(model, tokenizer, smart_config, export_push_config=config)
        instance.verbose = verbose

        # Reflect the *actual* runtime state of the loaded model rather than
        # the user's load-time intent. ``from_config_only=True`` returns a
        # randomly-initialised model with no quantization regardless of the
        # ``quantize`` flag, and a missing ``bitsandbytes`` install also
        # silently falls back to full precision -- both of which used to leave
        # ``_is_quantized=True`` set incorrectly.
        actual_quantized = instance._has_runtime_quantization()
        if from_config_only:
            # ``AutoModelForCausalLM.from_config`` returns a model with random
            # weights and never honours ``quantization_config`` -- so the
            # actual quantization state is whatever the loader produced
            # (almost always ``False``).
            instance._is_quantized_override = bool(actual_quantized)
            if verbose:
                print_warning(
                    "from_config_only=True returned a model with random "
                    "weights and no quantization. Call model.load_weights(...) "
                    "or reload with from_config_only=False before using it "
                    "for inference."
                )
        else:
            # Let the property derive truth from the model state. Override is
            # only set when the caller explicitly asked for quantization but
            # the runtime layer silently skipped it (e.g. bitsandbytes
            # missing) -- in that case we set False so downstream code does
            # not try to call BnB-only training paths on a full-precision
            # model.
            wanted_quantization = cls._should_apply_quantization(
                quantize, smart_config.bits, from_config_only=False
            )
            if wanted_quantization and not actual_quantized:
                instance._is_quantized_override = False
                if verbose:
                    print_warning(
                        "Requested quantization was not applied at load time "
                        "(typically because ``bitsandbytes`` is not installed "
                        "or the model was already pre-quantized). Continuing "
                        "in full precision."
                    )

        return instance
    
    @classmethod
    def from_gguf(
        cls,
        model_id: str,
        filename: Optional[str] = None,
        *,
        device: Optional[str] = None,
        verbose: bool = True,
        **kwargs
    ) -> "TurboModel":
        """
        Load a GGUF model directly from HuggingFace Hub or local path.
        
        This uses transformers' native GGUF support (requires transformers>=4.36.0).
        
        Args:
            model_id: HuggingFace repo ID (e.g., "TheBloke/Llama-2-7B-GGUF") or local directory
            filename: GGUF filename (e.g., "llama-2-7b.Q4_K_M.gguf"). 
                      If None, tries to auto-find. Use list_gguf_files() to see available options.
            device: Override device (default: auto-detect best GPU)
            verbose: Print progress
            **kwargs: Additional args for AutoModelForCausalLM.from_pretrained
            
        Returns:
            TurboModel with loaded GGUF model
            
        Example:
            >>> # List available GGUF files in a repo
            >>> files = TurboModel.list_gguf_files("TheBloke/Llama-2-7B-GGUF")
            >>> print(files)
            >>> 
            >>> # Load specific quantization
            >>> model = TurboModel.from_gguf(
            ...     "TheBloke/Llama-2-7B-GGUF", 
            ...     filename="llama-2-7b.Q4_K_M.gguf"
            ... )
            >>> 
            >>> # Generate text
            >>> model.generate("Hello!")
        """
        if verbose:
            print_header(f"Loading GGUF: {model_id}")
            
        # Check for GGUF package
        try:
            import gguf
            gguf_version = getattr(gguf, '__version__', 'unknown')
            if verbose:
                print_info(f"Using gguf version: {gguf_version}")
        except ImportError:
            print_error("Missing 'gguf' package!")
            raise ImportError(
                "Loading GGUF models requires the 'gguf' package.\n"
                "Please run: pip install gguf>=0.10.0"
            )
        
        # If no filename specified, try to find one
        if filename is None:
            if verbose:
                print_info("No filename specified, searching for GGUF files...")
            try:
                available_files = cls.list_gguf_files(model_id)
                if available_files:
                    # Prefer Q4_K_M if available, otherwise take first
                    q4_files = [f for f in available_files if 'q4_k_m' in f.lower()]
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
                    **kwargs
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
                        f"GGUF file '{filename}' not found in {model_id}.\n"
                        f"Available files: {available}"
                    ) from e
                raise
            
            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
            except Exception:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                except Exception:
                    # Some GGUF repos might not have tokenizer, try base model
                    if verbose:
                        print_warning("Could not load tokenizer from GGUF repo, using default")
                    tokenizer = None
                    
            if verbose:
                print_success("GGUF Model loaded successfully!")
                
                # Print model info
                if hasattr(model, 'num_parameters'):
                    params = model.num_parameters() / 1e9
                    print_info(f"Parameters: {params:.2f}B")
             
        instance = cls(model, tokenizer, smart_config, verbose=verbose)
        # GGUF models are inherently quantized; set the override so the
        # property does not need to introspect the (often opaque) loaded
        # weights.
        instance._is_quantized_override = True
        return instance
    
    @staticmethod
    def list_gguf_files(model_id: str) -> List[str]:
        """
        List available GGUF files in a HuggingFace repository.
        
        Args:
            model_id: HuggingFace repo ID (e.g., "TheBloke/Llama-2-7B-GGUF")
            
        Returns:
            List of GGUF filenames available in the repository
            
        Example:
            >>> files = TurboModel.list_gguf_files("TheBloke/Llama-2-7B-GGUF")
            >>> print(files)
            ['llama-2-7b.Q2_K.gguf', 'llama-2-7b.Q4_K_M.gguf', ...]
        """
        try:
            from huggingface_hub import list_repo_files
            
            all_files = list_repo_files(model_id)
            gguf_files = [f for f in all_files if f.endswith('.gguf')]
            
            # Sort by quantization quality (Q4_K_M before Q2_K, etc.)
            def quant_sort_key(name):
                name_lower = name.lower()
                # Higher number = better quality, listed first
                if 'f32' in name_lower: return 0
                if 'f16' in name_lower: return 1
                if 'q8' in name_lower: return 2
                if 'q6' in name_lower: return 3
                if 'q5_k_m' in name_lower: return 4
                if 'q5_k_s' in name_lower: return 5
                if 'q4_k_m' in name_lower: return 6
                if 'q4_k_s' in name_lower: return 7
                if 'q3_k' in name_lower: return 8
                if 'q2_k' in name_lower: return 9
                return 10
            
            return sorted(gguf_files, key=quant_sort_key)
            
        except Exception as e:
            # If it's a local path, list directory
            if os.path.isdir(model_id):
                return [f for f in os.listdir(model_id) if f.endswith('.gguf')]
            raise ValueError(f"Could not list GGUF files from {model_id}: {e}")

    @staticmethod
    def _get_quantization_kwargs(config: SmartConfig) -> Dict[str, Any]:
        """
        Get kwargs for quantized model loading.
        
        Note: BitsAndBytes only supports 4-bit and 8-bit quantization for loading.
        Other bit widths (2, 3, 5, 6) are only available during GGUF export.
        
        For loading:
        - bits <= 4: Uses 4-bit NF4 quantization
        - bits 5-7: Uses 8-bit quantization  
        - bits >= 8: Uses 8-bit quantization
        """
        try:
            from transformers import BitsAndBytesConfig
            
            # BitsAndBytes only supports 4-bit and 8-bit
            # Map requested bits to available options
            if config.bits <= 4:
                # 2, 3, 4-bit requests -> 4-bit NF4 (smallest available)
                effective_bits = 4
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=config.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                if config.bits < 4:
                    logger.info(f"  ℹ️ BitsAndBytes supports 4/8-bit only. Using 4-bit for requested {config.bits}-bit.")
                    logger.info(f"     Tip: Export to GGUF for Q{config.bits}_K quantization!")
            else:
                # 5, 6, 7, 8-bit requests -> 8-bit
                effective_bits = 8
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                if config.bits != 8:
                    logger.info(f"  ℹ️ BitsAndBytes supports 4/8-bit only. Using 8-bit for requested {config.bits}-bit.")
                    logger.info(f"     Tip: Export to GGUF for Q{config.bits}_K quantization!")
            
            return {"quantization_config": quantization_config}
            
        except ImportError:
            logger.warning(
                "\u26a0 bitsandbytes is not installed; falling back to full "
                "precision. Install with ``pip install bitsandbytes`` to "
                "enable 4-bit / 8-bit quantization on CUDA."
            )
            return {}

    @staticmethod
    def _build_export_push_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
                if mapped_key in resolved and (
                    value is not None or mapped_key in nullable_overrides
                ):
                    resolved[mapped_key] = value

            if "format" in config and "push_format" not in config:
                resolved["push_format"] = resolved["format"]
            if "quantization" in config and "push_quantization" not in config:
                resolved["push_quantization"] = resolved["quantization"]

        return resolved
    
    @staticmethod
    def _enable_flash_attention(model: PreTrainedModel, verbose: bool = True) -> None:
        """Enable Flash Attention if available.
        
        Note: For full effect, flash_attention_2 should be specified at
        load time via attn_implementation='flash_attention_2' in from_pretrained().
        This method does a best-effort post-load enable via config and SDPA.
        """
        try:
            # Try to enable SDPA (Scaled Dot Product Attention) which is
            # the native PyTorch path and works without flash-attn package
            if hasattr(model, 'config'):
                if hasattr(model.config, '_attn_implementation'):
                    # Try flash_attention_2 first, fall back to sdpa
                    try:
                        import flash_attn
                        model.config._attn_implementation = "flash_attention_2"
                        if verbose:
                            logger.info("  ✓ Flash Attention 2 configured")
                    except ImportError:
                        model.config._attn_implementation = "sdpa"
                        if verbose:
                            logger.info("  ✓ SDPA (Scaled Dot-Product Attention) configured")
        except Exception:
            if verbose:
                logger.warning("  ⚠ Flash Attention / SDPA not available")
    
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        stream: bool = False,
        repetition_penalty: float = 1.1,
        stop_strings: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample (False = greedy)
            stream: Whether to stream output token by token
            repetition_penalty: Penalty for repeating tokens (>1.0 = less repetition)
            stop_strings: List of strings that stop generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Example:
            >>> response = model.generate("Explain quantum computing")
            >>> 
            >>> # With streaming
            >>> response = model.generate("Tell me a story", stream=True)
        """
        import sys
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - max_new_tokens,
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Default stop strings
        stop_strings = stop_strings or []
        
        # Streaming generation
        if stream:
            return self._generate_streaming(
                inputs, max_new_tokens, temperature, top_p, top_k,
                do_sample, repetition_penalty, stop_strings, **kwargs
            )
        
        # Non-streaming generation
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
        
        # Decode, removing the prompt
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # Check for stop strings and truncate
        for stop in stop_strings:
            if stop in response:
                response = response.split(stop)[0]
                break
        
        return response.strip()
    
    def _generate_streaming(
        self,
        inputs: Dict,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: float,
        stop_strings: List[str],
        **kwargs,
    ) -> str:
        """Generate with streaming output."""
        import sys
        
        try:
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True,
                skip_special_tokens=True,
            )
            
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature if do_sample else 1.0,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": repetition_penalty,
                "streamer": streamer,
                **kwargs,
            }
            
            # Run generation in background thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream output
            generated_text = []
            for new_text in streamer:
                sys.stdout.write(new_text)
                sys.stdout.flush()
                generated_text.append(new_text)
                
                # Check stop strings
                full_text = "".join(generated_text)
                should_stop = False
                for stop in stop_strings:
                    if stop in full_text:
                        should_stop = True
                        break
                if should_stop:
                    break
            
            thread.join()
            print()  # New line after streaming
            
            response = "".join(generated_text)
            for stop in stop_strings:
                if stop in response:
                    response = response.split(stop)[0]
            
            return response.strip()
            
        except ImportError:
            # Fallback to non-streaming
            print("(Streaming not available, using batch generation)")
            return self.generate(
                self.tokenizer.decode(inputs["input_ids"][0]),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stream=False,
            )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Generate response for chat-format messages.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            **kwargs: Additional generation parameters
            
        Returns:
            Assistant's response
            
        Example:
            >>> response = model.chat([
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Hello!"}
            ... ])
        """
        # Try to apply chat template
        prompt = None
        
        # First, try native chat template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        
        # If no template, use a sensible default
        if prompt is None:
            # Default chat format (works for most models)
            parts = []
            for m in messages:
                role = m.get('role', 'user')
                content = m.get('content', '')
                if role == 'system':
                    parts.append(f"System: {content}\n")
                elif role == 'user':
                    parts.append(f"User: {content}\n")
                elif role == 'assistant':
                    parts.append(f"Assistant: {content}\n")
            parts.append("Assistant:")
            prompt = "".join(parts)
        
        return self.generate(prompt, **kwargs)
    
    def finetune(
        self,
        data: Union[str, List[Dict[str, str]], Any],
        *,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        output_dir: Optional[str] = None,
        hub_manager: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Fine-tune the model with LoRA.
        
        Args:
            data: Training data - file path, list of dicts, or HF dataset
            epochs: Number of training epochs (default: auto)
            learning_rate: Learning rate (default: auto based on model size)
            batch_size: Batch size (default: auto based on GPU memory)
            lora_r: LoRA rank (default: auto)
            lora_alpha: LoRA alpha (default: 2x lora_r)
            output_dir: Where to save (default: ./output/{model_name})
            hub_manager: QuantLLMHubManager instance for auto-tracking
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        print_header("Starting Fine-tuning")

        # 1. Memory & Environment Optimizations
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if "WANDB_DISABLED" not in os.environ:
            os.environ["WANDB_DISABLED"] = "true"
            
        # Suppress noise
        import warnings
        warnings.filterwarnings("ignore", module="peft")
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        except ImportError:
            raise ImportError("peft is required for fine-tuning. Install with: pip install peft")
        
        # 2. Prepare model for training
        if self._is_quantized:
            self.model = prepare_model_for_kbit_training(self.model)
            
        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # 3. Auto-configure LoRA
        r = lora_r or (16 if self.model.num_parameters() < 10e9 else 64)
        alpha = lora_alpha or (r * 2)
        
        # Determine target modules based on model type
        target_modules = self._get_lora_target_modules()
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA if not already applied
        if not self._lora_applied:
            self.model = get_peft_model(self.model, lora_config)
            self._lora_applied = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print_info(f"LoRA applied: {trainable:,} trainable params ({100*trainable/total:.2f}%)")
        
        # 4. Load and prepare data
        train_dataset = self._prepare_dataset(data)
        
        # 5. Auto-configure training settings
        epochs = epochs or 3
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or 2e-4
        output_dir = output_dir or f"./output/{self.model.config._name_or_path.split('/')[-1]}"
        
        # Auto-track parameters if hub_manager provided
        if hub_manager:
            hub_manager.track_hyperparameters({
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "lora_r": r,
                "lora_alpha": alpha,
                "base_model": getattr(self.config, "model_name", "unknown"),
                "output_dir": output_dir
            })
        
        # Training loop
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=learning_rate,
                fp16=self.config.dtype == torch.float16,
                bf16=self.config.dtype == torch.bfloat16,
                logging_steps=10,
                save_strategy="epoch",
                warmup_ratio=0.03,
                lr_scheduler_type="cosine",
                gradient_checkpointing=self.config.gradient_checkpointing,
                optim="paged_adamw_32bit" if self.config.bits <= 4 else "adamw_torch",
                torch_compile=self.config.compile_model,
                **kwargs,
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                processing_class=self.tokenizer,
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            
            result = trainer.train()
            self._is_finetuned = True
            
            print_success(f"Training complete! Model saved to {output_dir}")
            
            return {
                "train_loss": result.training_loss,
                "epochs": epochs,
                "output_dir": output_dir,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "lora_r": r
            }
            
        except Exception as e:
            print_error(f"Training failed: {e}")
            # Hint about OOM
            if "out of memory" in str(e).lower():
                print_info("Tip: Try reducing batch_size or enabling gradient_checkpointing in config.")
            raise
    
    def _get_lora_target_modules(self) -> List[str]:
        """Get appropriate LoRA target modules for the model."""
        model_type = getattr(self.model.config, 'model_type', '').lower()
        
        # Common patterns by model type
        target_patterns = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        
        return target_patterns.get(model_type, ["q_proj", "v_proj"])
    
    def _prepare_dataset(self, data: Union[str, List[Dict], Any]) -> Any:
        """Prepare dataset for training."""
        from datasets import Dataset, load_dataset
        
        if isinstance(data, str):
            # Load from file
            if data.endswith('.json') or data.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=data)['train']
            else:
                dataset = load_dataset(data)['train']
        elif isinstance(data, list):
            dataset = Dataset.from_list(data)
        else:
            dataset = data  # Assume it's already a Dataset
        
        # Tokenize
        def tokenize_function(examples):
            # Handle different data formats
            if 'text' in examples:
                texts = examples['text']
            elif 'instruction' in examples and 'output' in examples:
                texts = [
                    f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                    for inst, out in zip(examples['instruction'], examples['output'])
                ]
            else:
                # Try to concatenate all string fields
                keys = [k for k in examples.keys() if isinstance(examples[k][0], str)]
                texts = [' '.join([examples[k][i] for k in keys]) for i in range(len(examples[keys[0]]))]
            
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False, # Use DataCollator for dynamic padding
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False, # Avoid hash warnings
            desc="Tokenizing dataset",
        )
        
        return tokenized
    
    def export(
        self,
        format: Optional[str] = None,
        output_path: Optional[str] = None,
        *,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Export model to various formats.
        
        Supported formats:
            - "gguf": For llama.cpp, Ollama, LM Studio (Q4_K_M, Q5_K_M, etc.)
            - "safetensors": For HuggingFace ecosystem
            - "onnx": For ONNX Runtime, TensorRT
            - "mlx": For Apple Silicon Macs
        
        Args:
            format: Target format (gguf, safetensors, onnx, mlx). Uses shared config when omitted.
            output_path: Output file/directory path
            quantization: Format-specific quantization:
                - GGUF: Q4_K_M, Q5_K_M, Q8_0, etc.
                - ONNX: dynamic, int8
                - MLX: 4bit, 8bit
            **kwargs: Format-specific options
            
        Returns:
            Path to exported model
            
        Example:
            >>> model.export("gguf")  # Uses auto name
            >>> model.export("gguf", "my_model.gguf", quantization="Q4_K_M")
            >>> model.export("onnx", "./my_model_onnx/")
            >>> model.export("mlx", "./my_model_mlx/", quantization="4bit")
        """
        format = (
            format
            if format is not None
            else self.export_push_config.get("format", DEFAULT_EXPORT_PUSH_CONFIG["format"])
        ).lower()
        effective_quantization = quantization
        if effective_quantization is None and format == "gguf":
            effective_quantization = self.export_push_config.get(
                "quantization", DEFAULT_EXPORT_PUSH_CONFIG["quantization"]
            )
        
        # Merge LoRA if applied
        if self._lora_applied:
            if self.verbose:
                print_info("Merging LoRA weights before export...")
            self.model = self.model.merge_and_unload()
            self._lora_applied = False
        
        # Auto-generate output path
        if output_path is None:
            model_name = self.model.config._name_or_path.split('/')[-1]
            if format == "gguf":
                quant = effective_quantization
                output_path = f"{model_name}.{quant.upper()}.gguf"
            elif format == "safetensors":
                output_path = f"./{model_name}-quantllm/"
            elif format == "onnx":
                output_path = f"./{model_name}-onnx/"
            elif format == "mlx":
                output_path = f"./{model_name}-mlx/"
            else:
                output_path = f"./{model_name}-{format}/"
        
        exporters = {
            "gguf": self._export_gguf,
            "safetensors": self._export_safetensors,
            "onnx": self._export_onnx,
            "mlx": self._export_mlx,
        }
        if format not in exporters:
            raise ValueError(f"Unknown format: {format}. Supported: {list(exporters.keys())}")
        
        print_header(f"Exporting to {format.upper()}")
        result = exporters[format](output_path, quantization=effective_quantization, **kwargs)
        print_success(f"Exported to: {result}")
        
        return result

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        format: Optional[str] = None,
        quantization: Optional[str] = None,
        commit_message: str = "Upload model via QuantLLM",
        license: str = "apache-2.0",
        private: bool = False,
        **kwargs
    ):
        """
        Push model to HuggingFace Hub with proper model card.
        
        Args:
            repo_id: Repository ID (e.g. "username/model")
            token: HF Token
            format: "safetensors", "gguf", "onnx", or "mlx"
            quantization: Quantization type (for gguf/onnx/mlx)
            commit_message: Commit message
            license: License type (default: apache-2.0)
            **kwargs: Arguments for export
            
        Supported formats:
            - safetensors: Standard HuggingFace format
            - gguf: For llama.cpp, Ollama, LM Studio
            - onnx: For ONNX Runtime, TensorRT
            - mlx: For Apple Silicon (requires macOS)
            
        The model card will be automatically generated with:
            - Proper YAML frontmatter for HuggingFace
            - Format-specific usage examples
            - "Use this model" button compatibility
        """
        from ..hub import QuantLLMHubManager
        
        format_lower = (
            format
            if format is not None
            else self.export_push_config.get("push_format", DEFAULT_EXPORT_PUSH_CONFIG["push_format"])
        ).lower()
        push_quantization = quantization or self.export_push_config.get(
            "push_quantization", DEFAULT_EXPORT_PUSH_CONFIG["push_quantization"]
        )
        
        # Get the original base model name (full path for HuggingFace link)
        base_model_full = self.model.config._name_or_path
        model_name = base_model_full.split('/')[-1]
        
        print_header(f"Pushing to {repo_id}")
        print_info(f"Format: {format_lower.upper()}")
        print_info(f"Base model: {base_model_full}")
        
        manager = QuantLLMHubManager(repo_id=repo_id, hf_token=token)
        
        if format_lower == "gguf":
            # Export GGUF directly to staging
            quant_label = push_quantization or self.export_push_config.get(
                "quantization", DEFAULT_EXPORT_PUSH_CONFIG["quantization"]
            )
            filename = f"{model_name}.{quant_label.upper()}.gguf"
            save_path = os.path.join(manager.staging_dir, filename)
            
            self.export(format="gguf", output_path=save_path, quantization=quant_label, **kwargs)
            
            manager.track_hyperparameters({
                "format": "gguf",
                "quantization": quant_label.upper(),
                "base_model": base_model_full,
                "license": license,
            })
            manager._generate_model_card(format="gguf")
            
        elif format_lower == "onnx":
            # Export to ONNX format
            print_info("Exporting to ONNX format...")
            save_path = manager.staging_dir
            
            self._export_onnx(save_path, quantization=push_quantization, **kwargs)
            
            manager.track_hyperparameters({
                "format": "onnx",
                "quantization": push_quantization,
                "base_model": base_model_full,
                "license": license,
            })
            manager._generate_model_card(format="onnx")
            
        elif format_lower == "mlx":
            # Export to MLX format
            print_info("Exporting to MLX format...")
            save_path = manager.staging_dir
            
            self._export_mlx(save_path, quantization=push_quantization, **kwargs)
            
            manager.track_hyperparameters({
                "format": "mlx",
                "quantization": push_quantization,
                "base_model": base_model_full,
                "license": license,
            })
            manager._generate_model_card(format="mlx")
            
        else:
            # SafeTensors or PyTorch format
            manager.track_hyperparameters({
                "format": format_lower,
                "base_model": base_model_full,
                "license": license,
            })
            manager.save_final_model(self, format=format_lower)
            manager._generate_model_card(format=format_lower)
            
        manager.push(commit_message=commit_message)
    
    # Alias for convenience
    push = push_to_hub
    
    def _export_gguf(
        self, 
        output_path: str, 
        quantization: Optional[str] = None,
        fast_mode: bool = False,
        chunked_conversion: bool = False,
        max_shard_size: Optional[str] = None,
        smart_tensor_ordering: bool = False,
        disk_offloading: bool = False,
        disk_offload_dir: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export to GGUF format using optimized llama.cpp converter.
        
        Automatically installs and configures llama.cpp tools.
        Handles BitsAndBytes quantized models by dequantizing first.
        
        Flow:
            1. Save model to temp directory (dequantize if needed)
            2. Convert to F16 GGUF using convert_hf_to_gguf.py
            3. Quantize to target format (Q4_K_M, Q5_K_M, etc.) using llama-quantize
            
        Args:
            output_path: Output file path for GGUF
            quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)
            fast_mode: Skip intermediate F16 step for faster export (slightly less optimal)
            chunked_conversion: Save model shards during conversion for large checkpoints
            max_shard_size: Max shard size used when chunked conversion is active
            smart_tensor_ordering: Save tensors in memory-optimized order
            disk_offloading: Use a dedicated temp/offload directory for intermediate artifacts
            disk_offload_dir: Directory used when disk_offloading=True
        """
        from ..quant import convert_to_gguf, quantize_gguf, ensure_llama_cpp_installed, GGUF_QUANT_TYPES
        from ..utils import QuantLLMProgress, format_time, format_size
        import time
        
        start_time = time.time()
        
        effective_shard_size = max_shard_size or (
            DEFAULT_CHUNKED_SHARD_SIZE if chunked_conversion else None
        )
        
        quant_type = quantization or self.config.quant_type or "q4_k_m"
        quant_type_upper = quant_type.upper()
        quant_type_lower = quant_type.lower()
        
        # Check if this is a passthrough format (f16, bf16, f32 - no quantization needed)
        passthrough_types = {'f16', 'f32', 'bf16', 'float16', 'float32', 'bfloat16'}
        needs_quantization = quant_type_lower not in passthrough_types
        
        if self.verbose:
            print_info(f"Target quantization: {quant_type_upper}")
            if fast_mode:
                print_info("Fast mode enabled")
            if chunked_conversion:
                print_info(f"Chunked conversion enabled (max_shard_size={effective_shard_size})")
            if smart_tensor_ordering:
                print_info("Smart tensor ordering enabled")
                print_warning("Smart tensor ordering may temporarily materialize a full state dict in memory.")
            if disk_offloading:
                print_info(f"Disk offloading enabled ({disk_offload_dir or 'system temp'})")
        
        # Ensure llama.cpp
        if self.verbose:
            print_info("Checking llama.cpp installation...")
        ensure_llama_cpp_installed()
        
        # Check if model is BitsAndBytes quantized and needs dequantization
        model_to_save = self.model
        is_bnb_quantized = self._is_bnb_quantized()
        
        if is_bnb_quantized:
            if self.verbose:
                print_warning("Model is BitsAndBytes quantized. Dequantizing for GGUF export...")
                print_info("This may use significant memory. For large models, consider loading with quantize=False.")
            
            model_to_save = self._dequantize_model()
            if self.verbose:
                print_success("Model dequantized successfully!")
        
        # Determine dtype for initial conversion (always F16 for best quality)
        model_dtype = "f16"
        
        # Get model name for file naming
        model_name = self.model.config._name_or_path.split('/')[-1]
        
        temp_parent = disk_offload_dir if disk_offloading else None
        if temp_parent:
            os.makedirs(temp_parent, exist_ok=True)
        
        # Create temp dir for conversion
        with tempfile.TemporaryDirectory(dir=temp_parent) as temp_dir:
            # Step 1: Save model to temp directory
            if self.verbose:
                print_header("Step 1/3: Saving Model", icon="💾")
                print_info(f"Staging model to {temp_dir}...")
            
            with QuantLLMProgress() as progress:
                task = progress.add_task("Saving model weights...", total=None)
                save_kwargs = {
                    "safe_serialization": True,
                }
                if effective_shard_size:
                    save_kwargs["max_shard_size"] = effective_shard_size
                
                if smart_tensor_ordering:
                    save_kwargs["state_dict"] = memory_optimized_tensor_order(model_to_save.state_dict())
                
                try:
                    model_to_save.save_pretrained(temp_dir, **save_kwargs)
                except Exception as e:
                    if self.verbose:
                        print_warning(f"SafeTensors save failed ({e}), using PyTorch format...")
                    save_kwargs["safe_serialization"] = False
                    model_to_save.save_pretrained(temp_dir, **save_kwargs)
                
                self.tokenizer.save_pretrained(temp_dir)
                progress.update(task, completed=100)
            
            if self.verbose:
                print_success("Model saved to staging area!")
            
            # Step 2: Convert to F16 GGUF
            if self.verbose:
                print_header("Step 2/3: Converting to GGUF", icon="🔄")
            
            # F16 intermediate file (or final if no quantization needed)
            if needs_quantization:
                f16_gguf_file = os.path.join(temp_dir, f"{model_name}.F16.gguf")
            else:
                f16_gguf_file = f"{model_name}.{quant_type_upper}.gguf"
            
            output_files, _ = convert_to_gguf(
                model_name=model_name,
                input_folder=temp_dir,
                model_dtype=model_dtype,
                quantization_type="f16" if needs_quantization else quant_type_lower,
                print_output=self.verbose
            )
            
            if not output_files:
                raise RuntimeError("GGUF conversion failed to produce output file.")
            
            f16_file = output_files[0]
            
            # If conversion produced a different name, use that
            if os.path.exists(f16_file):
                f16_gguf_file = f16_file
            
            if self.verbose:
                print_success(f"F16 GGUF created: {f16_gguf_file}")
            
            # Step 3: Apply quantization if needed
            if needs_quantization:
                if self.verbose:
                    print_header(f"Step 3/3: Quantizing to {quant_type_upper}", icon="⚡")
                    print_info(f"Applying {quant_type_upper} quantization...")
                
                # Final quantized output
                quantized_file = f"{model_name}.{quant_type_upper}.gguf"
                
                quantize_gguf(
                    input_gguf=f16_gguf_file,
                    output_gguf=quantized_file,
                    quant_type=quant_type_upper,
                    print_output=self.verbose
                )
                
                final_file = quantized_file
                
                # Clean up intermediate F16 file
                if os.path.exists(f16_gguf_file) and f16_gguf_file != quantized_file:
                    os.remove(f16_gguf_file)
                
                if self.verbose:
                    print_success(f"Quantization complete: {quantized_file}")
            else:
                final_file = f16_gguf_file
                if self.verbose:
                    print_info("No quantization needed (already in target format)")
            
            # Move to output path if different
            if os.path.abspath(final_file) != os.path.abspath(output_path):
                if self.verbose:
                    print_info(f"Moving {final_file} → {output_path}")
                shutil.move(final_file, output_path)
        
        # Clean up dequantized model if created
        if is_bnb_quantized and model_to_save is not self.model:
            del model_to_save
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Print final summary
        elapsed = time.time() - start_time
        if self.verbose:
            file_size_bytes = os.path.getsize(output_path)
            print_header("Export Complete! 🎉", icon="✅")
            print_info(f"Output: {output_path}")
            print_info(f"Format: GGUF {quant_type_upper}")
            print_info(f"Size: {format_size(file_size_bytes)}")
            print_info(f"Time: {format_time(elapsed)}")
                
        return output_path

    def _is_bnb_quantized(self) -> bool:
        """Return True iff the loaded model is BitsAndBytes-quantized.

        Checks both the model's ``quantization_config`` metadata and the
        actual layer types (``Linear4bit`` / ``Linear8bitLt``) so it works
        whether the model came from a pre-quantized HF repo or from a
        dynamic BitsAndBytes load.
        """
        if hasattr(self.model, 'config'):
            quant_config = getattr(self.model.config, 'quantization_config', None)
            if quant_config:
                quant_method = getattr(quant_config, 'quant_method', None)
                if quant_method in ('bitsandbytes', 'bnb'):
                    return True
                if getattr(quant_config, 'load_in_4bit', False):
                    return True
                if getattr(quant_config, 'load_in_8bit', False):
                    return True

        try:
            import bitsandbytes as bnb
            for module in self.model.modules():
                if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                    return True
        except ImportError:
            pass

        return False

    def _has_runtime_quantization(self) -> bool:
        """Return True iff the loaded model carries *any* quantization.

        Detects BitsAndBytes (4-bit / 8-bit), GPTQ, AWQ, AQLM, HQQ, FP8
        and EETQ via the standard ``quantization_config.quant_method`` slot
        on a ``transformers`` ``PretrainedConfig``. This is the canonical
        source-of-truth used by :pyattr:`is_quantized`.
        """
        if self._is_bnb_quantized():
            return True

        if hasattr(self.model, 'config'):
            quant_config = getattr(self.model.config, 'quantization_config', None)
            if quant_config:
                quant_method = getattr(quant_config, 'quant_method', None)
                if quant_method:
                    return True
                if isinstance(quant_config, dict) and quant_config.get('quant_method'):
                    return True
        return False

    @property
    def is_quantized(self) -> bool:
        """Whether the underlying model is currently quantized.

        Derived from the loaded model state (``config.quantization_config``
        and the actual layer types). When :meth:`from_gguf` or another
        loader explicitly knows the quantization status it can set
        :pyattr:`_is_quantized_override` to short-circuit the introspection.
        """
        if self._is_quantized_override is not None:
            return self._is_quantized_override
        return self._has_runtime_quantization()

    # Backwards-compatible alias kept for existing internal callers and any
    # downstream code that read the previous attribute name. New code should
    # prefer the :pyattr:`is_quantized` public property.
    @property
    def _is_quantized(self) -> bool:  # type: ignore[override]
        return self.is_quantized

    @_is_quantized.setter
    def _is_quantized(self, value: Optional[bool]) -> None:
        self._is_quantized_override = None if value is None else bool(value)

    def report(self) -> Dict[str, Any]:
        """Return a structured snapshot of the actual loaded-model state.

        Keys:
            * ``model_id``: HF repo id or local path (when known).
            * ``params_billion``: parameter count in billions.
            * ``requested_bits``: bits the user (or :class:`SmartConfig`)
              asked for.
            * ``effective_loading_bits``: bits actually used for BnB loading
              (4 / 8 / 16). Differs from ``requested_bits`` when GGUF export
              targets sub-4-bit quantization but loading falls back to 4-bit.
            * ``is_quantized``: real runtime quantization state.
            * ``quant_method``: e.g. ``"bitsandbytes"`` / ``"gptq"`` / ``None``.
            * ``device``: torch device the model lives on.
            * ``dtype``: torch dtype of the model parameters.
            * ``finetuned`` / ``lora_applied``: training-state flags.
        """
        params_billion: Optional[float]
        try:
            params_billion = self.model.num_parameters() / 1e9
        except Exception:
            params_billion = None

        quant_method = None
        if hasattr(self.model, 'config'):
            quant_config = getattr(self.model.config, 'quantization_config', None)
            if quant_config is not None:
                quant_method = (
                    getattr(quant_config, 'quant_method', None)
                    or (quant_config.get('quant_method') if isinstance(quant_config, dict) else None)
                )
                if not quant_method and self._is_bnb_quantized():
                    quant_method = 'bitsandbytes'

        device = getattr(self.model, 'device', None)
        try:
            dtype = next(self.model.parameters()).dtype
        except (StopIteration, AttributeError):
            dtype = getattr(self.config, 'dtype', None)

        return {
            "model_id": getattr(getattr(self.model, 'config', None), '_name_or_path', None),
            "params_billion": params_billion,
            "requested_bits": getattr(self.config, 'bits', None),
            "effective_loading_bits": getattr(self.config, 'effective_loading_bits', None),
            "is_quantized": self.is_quantized,
            "quant_method": quant_method,
            "device": str(device) if device is not None else None,
            "dtype": str(dtype) if dtype is not None else None,
            "finetuned": self._is_finetuned,
            "lora_applied": self._lora_applied,
        }

    def _dequantize_model(self) -> nn.Module:
        """
        Dequantize a BitsAndBytes model to full precision for GGUF export.
        
        Returns:
            Dequantized model in float16/bfloat16
        """
        import gc
        
        # Get the model name for reloading
        model_name = getattr(self.model.config, '_name_or_path', None)
        
        if model_name:
            # Best approach: Reload model in full precision
            if self.verbose:
                print_info(f"Reloading {model_name} in full precision...")
            
            # Determine target dtype
            target_dtype = self.config.dtype if self.config.dtype in [torch.float16, torch.bfloat16] else torch.float16
            
            try:
                dequant_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=target_dtype,
                    trust_remote_code=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )
                return dequant_model
            except Exception as e:
                if self.verbose:
                    print_warning(f"Failed to reload model: {e}")
                    print_info("Attempting in-place dequantization...")
        
        # Fallback: In-place dequantization (less reliable but works for some models)
        try:
            import bitsandbytes as bnb
            
            target_dtype = self.config.dtype if self.config.dtype in [torch.float16, torch.bfloat16] else torch.float16
            
            # Create a copy of the model state dict with dequantized weights
            dequant_model = AutoModelForCausalLM.from_config(
                self.model.config,
                torch_dtype=target_dtype,
            )
            
            # Copy and dequantize weights
            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if isinstance(module, bnb.nn.Linear4bit):
                        # Dequantize 4-bit weights
                        target_module = dict(dequant_model.named_modules()).get(name)
                        if target_module is not None and hasattr(target_module, 'weight'):
                            # Get dequantized weight
                            weight = module.weight
                            if hasattr(weight, 'dequantize'):
                                dequant_weight = weight.dequantize()
                            else:
                                # Manual dequantization for older versions
                                dequant_weight = bnb.functional.dequantize_4bit(
                                    weight.data, weight.quant_state
                                )
                            target_module.weight.data.copy_(dequant_weight.to(target_dtype))
                            
                            if module.bias is not None:
                                target_module.bias.data.copy_(module.bias.data.to(target_dtype))
                    
                    elif isinstance(module, bnb.nn.Linear8bitLt):
                        # Dequantize 8-bit weights
                        target_module = dict(dequant_model.named_modules()).get(name)
                        if target_module is not None and hasattr(target_module, 'weight'):
                            weight = module.weight
                            if hasattr(weight, 'dequantize'):
                                dequant_weight = weight.dequantize()
                            else:
                                dequant_weight = weight.data.to(target_dtype)
                            target_module.weight.data.copy_(dequant_weight.to(target_dtype))
                            
                            if module.bias is not None:
                                target_module.bias.data.copy_(module.bias.data.to(target_dtype))
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return dequant_model
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to dequantize BitsAndBytes model: {e}\n\n"
                "To export to GGUF, please reload your model without quantization:\n"
                "  model = TurboModel.from_pretrained('your-model', quantize=False)\n"
                "  model.export('gguf', quantization='Q4_K_M')"
            )
    
    def _export_safetensors(
        self,
        output_path: str,
        **kwargs
    ) -> str:
        """Export to safetensors format."""
        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path, safe_serialization=True)
        self.tokenizer.save_pretrained(output_path)
        return output_path
    
    def _export_onnx(
        self,
        output_path: str,
        quantization: Optional[str] = None,
        opset_version: int = 14,
        **kwargs
    ) -> str:
        """
        Export to ONNX format with proper structure.
        
        Uses Optimum's ONNX exporter which properly handles LLMs like Llama.
        torch.onnx.export does NOT work for modern LLMs due to dynamic attention.
        
        Args:
            output_path: Output directory for ONNX files
            quantization: ONNX quantization type (dynamic, static, int8, avx2, avx512)
            opset_version: ONNX opset version (default: 14)
        """
        from ..utils import QuantLLMProgress
        
        # Check for required dependencies
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            HAS_OPTIMUM = True
        except ImportError:
            HAS_OPTIMUM = False
        
        if not HAS_OPTIMUM:
            # Cannot export LLMs without Optimum - torch.onnx.export doesn't work
            error_msg = """
ONNX export requires the Optimum library for LLM models.

torch.onnx.export does NOT support modern LLMs (Llama, Mistral, etc.) 
due to dynamic attention patterns and complex operations.

Please install the required packages:

    pip install onnx onnxruntime optimum[onnxruntime] onnxscript

Or install with extras:

    pip install quantllm[onnx]
"""
            print_error(error_msg)
            raise ImportError("ONNX export requires: pip install onnx onnxruntime optimum[onnxruntime] onnxscript")
        
        os.makedirs(output_path, exist_ok=True)
        model_name = self.model.config._name_or_path
        
        if self.verbose:
            print_info("Using Optimum for ONNX export...")
        
        with QuantLLMProgress() as progress:
            task = progress.add_task("Exporting to ONNX...", total=None)
            
            try:
                # Check if model is quantized - need to export from original
                if self._is_bnb_quantized():
                    if self.verbose:
                        print_info("BNB quantized model detected. Exporting from original HuggingFace model...")
                    
                    # Export directly from HuggingFace (not our quantized version)
                    ort_model = ORTModelForCausalLM.from_pretrained(
                        model_name,
                        export=True,
                        trust_remote_code=True,
                    )
                else:
                    # Save model first, then convert
                    temp_path = os.path.join(output_path, "_temp_hf")
                    os.makedirs(temp_path, exist_ok=True)
                    
                    try:
                        self.model.save_pretrained(temp_path, safe_serialization=True)
                        self.tokenizer.save_pretrained(temp_path)
                        
                        ort_model = ORTModelForCausalLM.from_pretrained(
                            temp_path,
                            export=True,
                            trust_remote_code=True,
                        )
                    finally:
                        # Clean temp
                        shutil.rmtree(temp_path, ignore_errors=True)
                
                # Save ONNX model
                ort_model.save_pretrained(output_path)
                self.tokenizer.save_pretrained(output_path)
                
            except Exception as e:
                progress.update(task, completed=100)
                error_str = str(e)
                
                # Check for common issues and provide helpful messages
                if "onnxscript" in error_str.lower():
                    print_error("Missing onnxscript package. Install with: pip install onnxscript")
                    raise ImportError("ONNX export requires onnxscript: pip install onnxscript") from e
                elif "cannot export" in error_str.lower() or "unsupported" in error_str.lower():
                    print_error(f"Model architecture may not support ONNX export: {error_str}")
                    raise
                else:
                    raise
            
            progress.update(task, completed=100)
        
        # Apply quantization if requested
        if quantization:
            if self.verbose:
                print_info(f"Applying {quantization} ONNX quantization...")
            self._quantize_onnx_model(output_path, quantization)
        
        if self.verbose:
            print_success(f"ONNX model exported to {output_path}")
        
        return output_path
    
    def _quantize_onnx_model(self, model_path: str, quant_type: str) -> None:
        """
        Apply ONNX quantization.
        
        ONNX supports INT8 (8-bit integer) quantization only.
        Unlike GGUF, ONNX doesn't support 2/3/4/5/6-bit quantization.
        
        Args:
            model_path: Path to ONNX model directory
            quant_type: Quantization type:
                - Bit-based: "8", "8bit", "int8" → INT8 quantization
                - Platform: "avx2", "avx512", "arm64" → Platform-optimized INT8
                - Type: "dynamic", "static" → Quantization method
                
        Note: Requests for 4-bit or other bit widths will use INT8 with a warning.
        """
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            
            quantizer = ORTQuantizer.from_pretrained(model_path)
            
            # Normalize quantization type
            quant_lower = quant_type.lower().replace("_", "").replace("-", "")
            
            # Check for bit-based requests (ONNX only supports 8-bit)
            bit_request = None
            for bit_pattern in ["2bit", "3bit", "4bit", "5bit", "6bit", "q2", "q3", "q4", "q5", "q6"]:
                if bit_pattern in quant_lower:
                    bit_request = bit_pattern
                    break
            
            if bit_request:
                print_warning(f"ONNX only supports INT8 (8-bit) quantization, not {quant_type}.")
                print_info("For lower bit quantization, use GGUF format instead.")
                print_info("Proceeding with INT8 quantization...")
            
            # Determine optimal config based on platform or explicit request
            if "avx512" in quant_lower or "vnni" in quant_lower:
                qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
                if self.verbose:
                    print_info("Using AVX512 VNNI INT8 quantization (Intel Xeon/Ice Lake+)")
            elif "arm64" in quant_lower or "arm" in quant_lower:
                qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=True)
                if self.verbose:
                    print_info("Using ARM64 INT8 quantization (Apple Silicon/ARM)")
            elif "static" in quant_lower:
                # Static quantization (requires calibration data)
                qconfig = AutoQuantizationConfig.avx2(is_static=True, per_channel=True)
                if self.verbose:
                    print_info("Using static INT8 quantization (AVX2)")
            else:
                # Default: Dynamic INT8 with AVX2 (widely compatible)
                qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
                if self.verbose:
                    print_info("Using dynamic INT8 quantization (AVX2)")
            
            # Apply quantization
            quantizer.quantize(save_dir=model_path, quantization_config=qconfig)
            
            if self.verbose:
                print_success("ONNX INT8 quantization applied successfully")
            
        except ImportError:
            print_warning("Optimum quantization not available. Skipping ONNX quantization.")
            print_info("Install with: pip install optimum[onnxruntime]")
        except Exception as e:
            print_warning(f"ONNX quantization failed: {e}")
            print_info("The unquantized ONNX model is still available.")
    
    def _export_mlx(
        self,
        output_path: str,
        quantization: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export to MLX format for Apple Silicon.
        
        MLX supports 4-bit and 8-bit quantization only.
        
        Args:
            output_path: Output directory
            quantization: MLX quantization options:
                - "4bit", "4", "Q4", "Q4_K_M" → 4-bit quantization
                - "8bit", "8", "Q8" → 8-bit quantization
                - None → No quantization (FP16)
                
        Note: 2-bit, 3-bit, 5-bit, 6-bit requests will map to closest (4 or 8-bit).
        """
        from ..utils import QuantLLMProgress
        import subprocess
        import sys
        
        # Check platform
        import platform
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            print_warning("MLX export is optimized for Apple Silicon Macs.")
            print_info("The model will be saved but may not run efficiently on this system.")
        
        try:
            import mlx
            HAS_MLX = True
        except ImportError:
            HAS_MLX = False
        
        os.makedirs(output_path, exist_ok=True)
        model_name = self.model.config._name_or_path
        
        if HAS_MLX:
            try:
                from mlx_lm import convert
                
                if self.verbose:
                    print_info("Using mlx-lm for conversion...")
                
                with QuantLLMProgress() as progress:
                    task = progress.add_task("Converting to MLX...", total=None)
                    
                    # Save HF model first if quantized
                    if self._is_bnb_quantized():
                        # Use original model name
                        source_path = model_name
                    else:
                        source_path = os.path.join(output_path, "_temp_hf")
                        os.makedirs(source_path, exist_ok=True)
                        self.model.save_pretrained(source_path)
                        self.tokenizer.save_pretrained(source_path)
                    
                    # Build convert arguments
                    convert_args = {
                        "hf_path": source_path,
                        "mlx_path": output_path,
                    }
                    
                    # Parse quantization request
                    if quantization:
                        quant_lower = quantization.lower().replace("_", "").replace("-", "")
                        
                        # MLX only supports 4-bit and 8-bit
                        if any(x in quant_lower for x in ["2", "3"]):
                            print_warning(f"MLX only supports 4-bit and 8-bit, not {quantization}.")
                            print_info("Using 4-bit quantization (smallest available).")
                            convert_args["quantize"] = True
                            convert_args["q_bits"] = 4
                        elif any(x in quant_lower for x in ["5", "6", "7"]):
                            print_warning(f"MLX only supports 4-bit and 8-bit, not {quantization}.")
                            print_info("Using 8-bit quantization (closest available).")
                            convert_args["quantize"] = True
                            convert_args["q_bits"] = 8
                        elif "4" in quant_lower:
                            convert_args["quantize"] = True
                            convert_args["q_bits"] = 4
                            if self.verbose:
                                print_info("Using 4-bit MLX quantization")
                        elif "8" in quant_lower:
                            convert_args["quantize"] = True
                            convert_args["q_bits"] = 8
                            if self.verbose:
                                print_info("Using 8-bit MLX quantization")
                        else:
                            # Default to 4-bit for any other quantization request
                            convert_args["quantize"] = True
                            convert_args["q_bits"] = 4
                            if self.verbose:
                                print_info("Using 4-bit MLX quantization (default)")
                    
                    # Run conversion
                    convert(**convert_args)
                    
                    # Clean temp
                    if not self._is_bnb_quantized():
                        shutil.rmtree(source_path, ignore_errors=True)
                    
                    progress.update(task, completed=100)
                    
            except Exception as e:
                print_error(f"MLX conversion failed: {e}")
                raise
        else:
            # Fallback: save as HF format with instructions
            if self.verbose:
                print_warning("mlx-lm not installed. Saving as HuggingFace format.")
                print_info("To convert to MLX: pip install mlx-lm && python -m mlx_lm.convert ...")
            
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            # Create README with conversion instructions
            readme_path = os.path.join(output_path, "CONVERT_TO_MLX.md")
            with open(readme_path, "w") as f:
                f.write("# Convert to MLX\n\n")
                f.write("This model was saved in HuggingFace format.\n")
                f.write("To convert to MLX format on Apple Silicon:\n\n")
                f.write("```bash\n")
                f.write("pip install mlx-lm\n")
                f.write(f"python -m mlx_lm.convert --hf-path {output_path} --mlx-path ./mlx_model\n")
                f.write("```\n")
        
        if self.verbose:
            print_success(f"MLX model exported to {output_path}")
        
        return output_path
    
    def __repr__(self) -> str:
        try:
            params = self.model.num_parameters() / 1e9
            params_str = f"{params:.2f}B"
        except Exception:
            params_str = "?"
        model_id = getattr(getattr(self.model, "config", None), "_name_or_path", "?")
        return (
            "TurboModel(\n"
            f"  model={model_id},\n"
            f"  params={params_str},\n"
            f"  bits={self.config.bits},\n"
            f"  quantized={self.is_quantized},\n"
            f"  finetuned={self._is_finetuned}\n"
            ")"
        )


    def optimize_inference(self, backend: str = "triton", bits: int = 4):
        """
        Optimize model for inference using high-performance kernels.
        
        Args:
            backend: Optimization backend ("triton")
            bits: Quantization bits (4 or 8)
        """
        if backend == "triton":
            from ..kernels.triton import TritonQuantizedLinear, is_triton_available
            if not is_triton_available():
                print_warning("Triton is not available or no GPU detected. Skipping optimization.")
                return
            
            if self.verbose:
                print_header("Optimizing with Triton Kernels ⚡")
                
            count = self._replace_with_triton(self.model, bits)
            
            if self.verbose:
                print_success(f"Optimized {count} layers with Triton fused kernels!")
                
    def _replace_with_triton(self, module: nn.Module, bits: int) -> int:
        """Recursively replace Linear layers with TritonQuantizedLinear."""
        from ..kernels.triton import TritonQuantizedLinear
        count = 0
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with Triton Linear
                if self.verbose:
                    print_info(f"Quantizing {name}...")
                
                quantized = TritonQuantizedLinear(
                    child.in_features, 
                    child.out_features, 
                    bits=bits, 
                    bias=child.bias is not None,
                    group_size=128
                )
                quantized.to(child.weight.device)
                quantized.quantize_from(child)
                
                setattr(module, name, quantized)
                count += 1
            else:
                count += self._replace_with_triton(child, bits)
        return count


def register_architecture(
    architecture: str,
    *,
    base_model_type: Optional[str] = None,
    model_class: Optional[Type[PreTrainedModel]] = None,
) -> None:
    """
    Register a new architecture alias and optional explicit model class.
    
    Example:
        >>> register_architecture("my-new-model", base_model_type="llama")
    """
    TurboModel.register_architecture(
        architecture,
        base_model_type=base_model_type,
        model_class=model_class,
    )


def turbo(
    model: str,
    *,
    bits: Optional[int] = None,
    max_length: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    base_model_fallback: bool = True,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> TurboModel:
    """
    Load and quantize any LLM in one line.
    
    This is the simplest way to use QuantLLM. Everything is
    automatically configured based on your hardware.
    
    Args:
        model: HuggingFace model name or local path
        bits: Override quantization bits (default: auto)
        max_length: Override max sequence length (default: auto)
        device: Override device (default: best GPU)
        dtype: Override dtype (default: bf16/fp16)
        base_model_fallback: Retry with resolved base model config on first-load failure
        config: Shared export/push config (format, quantization, push_format, etc.)
        **kwargs: Additional options passed to from_pretrained
        
    Returns:
        TurboModel ready for use
        
    Examples:
        >>> # Simplest usage - everything automatic
        >>> model = turbo("meta-llama/Llama-3-8B")
        >>> 
        >>> # Override quantization
        >>> model = turbo("mistralai/Mistral-7B", bits=4)
        >>> 
        >>> # For long context
        >>> model = turbo("Qwen/Qwen2-72B", max_length=32768)
        >>> 
        >>> # Generate text
        >>> print(model.generate("Hello, world!"))
        >>> 
        >>> # Fine-tune
        >>> model.finetune("my_data.json")
        >>> 
        >>> # Export
        >>> model.export("gguf")
    """
    return TurboModel.from_pretrained(
        model,
        bits=bits,
        max_length=max_length,
        device=device,
        dtype=dtype,
        base_model_fallback=base_model_fallback,
        config=config,
        **kwargs,
    )
