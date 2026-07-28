"""
TurboModel - The Ultra-Simple QuantLLM API.

Load, quantize, fine-tune, and export LLMs with one line each.

This module composes domain-specific mixins into the unified
:class:`TurboModel` class exposed to users.
"""

from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from .architecture import ArchitectureMixin
from .export_router import ExportRouterMixin
from .finetune_mixin import FinetuneMixin
from .generation import GenerationMixin
from .model_gguf import ModelGGUFMixin
from .model_loader import ModelLoaderMixin
from .push import PushToHubMixin
from .smart_config import SmartConfig
from ..utils import print_header, print_info, print_success, print_warning


class TurboModel(
    ArchitectureMixin,
    ModelLoaderMixin,
    ModelGGUFMixin,
    GenerationMixin,
    FinetuneMixin,
    ExportRouterMixin,
    PushToHubMixin,
):
    """High-performance LLM with the simplest possible API.

    Features:
        - One-line loading with automatic quantization
        - One-line fine-tuning with LoRA
        - One-line export to multiple formats
        - Automatic hardware optimization
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: SmartConfig,
        export_push_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._is_quantized_override: Optional[bool] = None
        self._is_finetuned = False
        self._lora_applied = False
        self.export_push_config = (
            ModelLoaderMixin._build_export_push_config(export_push_config)
        )
        self.verbose = verbose

    def save(self, output_dir: str, safe_serialization: bool = True) -> str:
        """Save the model and tokenizer to a local directory."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.model
        if self._lora_applied:
            if self.verbose:
                print_info("Merging LoRA weights before saving...")
            model_to_save = self.model.merge_and_unload()
            self._lora_applied = False
            self.model = model_to_save

        model_to_save.save_pretrained(
            output_dir, safe_serialization=safe_serialization
        )
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

        if self.verbose:
            print_success(f"Model saved to {output_dir}")

        return output_dir

    def _is_bnb_quantized(self) -> bool:
        """Return True if the loaded model is BitsAndBytes-quantized."""
        if hasattr(self.model, "config"):
            quant_config = getattr(
                self.model.config, "quantization_config", None
            )
            if quant_config:
                quant_method = getattr(quant_config, "quant_method", None)
                if quant_method in ("bitsandbytes", "bnb"):
                    return True
                if getattr(quant_config, "load_in_4bit", False):
                    return True
                if getattr(quant_config, "load_in_8bit", False):
                    return True

        try:
            import bitsandbytes as bnb

            for module in self.model.modules():
                if isinstance(
                    module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
                ):
                    return True
        except ImportError:
            pass

        return False

    def _has_runtime_quantization(self) -> bool:
        """Return True if the loaded model carries any quantization."""
        if self._is_bnb_quantized():
            return True

        if hasattr(self.model, "config"):
            quant_config = getattr(
                self.model.config, "quantization_config", None
            )
            if quant_config:
                quant_method = getattr(quant_config, "quant_method", None)
                if quant_method:
                    return True
                if isinstance(quant_config, dict) and quant_config.get(
                    "quant_method"
                ):
                    return True
        return False

    @property
    def is_quantized(self) -> bool:
        """Whether the underlying model is currently quantized.

        Derived from the loaded model state (``config.quantization_config``
        and the actual layer types).
        """
        if self._is_quantized_override is not None:
            return self._is_quantized_override
        return self._has_runtime_quantization()

    @property
    def _is_quantized(self) -> bool:
        return self.is_quantized

    @_is_quantized.setter
    def _is_quantized(self, value: Optional[bool]) -> None:
        self._is_quantized_override = (
            None if value is None else bool(value)
        )

    def report(self) -> Dict[str, Any]:
        """Return a structured snapshot of the actual loaded-model state.

        Keys:
            model_id, params_billion, requested_bits,
            effective_loading_bits, is_quantized, quant_method,
            device, dtype, finetuned, lora_applied.
        """
        params_billion: Optional[float]
        try:
            params_billion = self.model.num_parameters() / 1e9
        except Exception:
            params_billion = None

        quant_method = None
        if hasattr(self.model, "config"):
            quant_config = getattr(
                self.model.config, "quantization_config", None
            )
            if quant_config is not None:
                quant_method = getattr(
                    quant_config, "quant_method", None
                ) or (
                    quant_config.get("quant_method")
                    if isinstance(quant_config, dict)
                    else None
                )
                if not quant_method and self._is_bnb_quantized():
                    quant_method = "bitsandbytes"

        device = getattr(self.model, "device", None)
        try:
            dtype = next(self.model.parameters()).dtype
        except (StopIteration, AttributeError):
            dtype = getattr(self.config, "dtype", None)

        return {
            "model_id": getattr(
                getattr(self.model, "config", None),
                "_name_or_path",
                None,
            ),
            "params_billion": params_billion,
            "requested_bits": getattr(self.config, "bits", None),
            "effective_loading_bits": getattr(
                self.config, "effective_loading_bits", None
            ),
            "is_quantized": self.is_quantized,
            "quant_method": quant_method,
            "device": str(device) if device is not None else None,
            "dtype": str(dtype) if dtype is not None else None,
            "finetuned": self._is_finetuned,
            "lora_applied": self._lora_applied,
        }

    def _dequantize_model(self) -> nn.Module:
        """Dequantize a BitsAndBytes model to full precision for GGUF export."""
        import gc

        from transformers import AutoModelForCausalLM

        model_name = getattr(self.model.config, "_name_or_path", None)

        if model_name:
            if self.verbose:
                print_info(f"Reloading {model_name} in full precision...")

            target_dtype = (
                self.config.dtype
                if self.config.dtype in [torch.float16, torch.bfloat16]
                else torch.float16
            )

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

        try:
            import bitsandbytes as bnb

            target_dtype = (
                self.config.dtype
                if self.config.dtype in [torch.float16, torch.bfloat16]
                else torch.float16
            )

            dequant_model = AutoModelForCausalLM.from_config(
                self.model.config,
                torch_dtype=target_dtype,
            )

            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if isinstance(module, bnb.nn.Linear4bit):
                        target_module = dict(
                            dequant_model.named_modules()
                        ).get(name)
                        if target_module is not None and hasattr(
                            target_module, "weight"
                        ):
                            weight = module.weight
                            if hasattr(weight, "dequantize"):
                                dequant_weight = weight.dequantize()
                            else:
                                dequant_weight = (
                                    bnb.functional.dequantize_4bit(
                                        weight.data, weight.quant_state
                                    )
                                )
                            target_module.weight.data.copy_(
                                dequant_weight.to(target_dtype)
                            )
                            if module.bias is not None:
                                target_module.bias.data.copy_(
                                    module.bias.data.to(target_dtype)
                                )

                    elif isinstance(module, bnb.nn.Linear8bitLt):
                        target_module = dict(
                            dequant_model.named_modules()
                        ).get(name)
                        if target_module is not None and hasattr(
                            target_module, "weight"
                        ):
                            weight = module.weight
                            if hasattr(weight, "dequantize"):
                                dequant_weight = weight.dequantize()
                            else:
                                dequant_weight = weight.data.to(target_dtype)
                            target_module.weight.data.copy_(
                                dequant_weight.to(target_dtype)
                            )
                            if module.bias is not None:
                                target_module.bias.data.copy_(
                                    module.bias.data.to(target_dtype)
                                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return dequant_model

        except Exception as e:
            raise RuntimeError(
                f"Failed to dequantize BitsAndBytes model: {e}\n\n"
                "To export to GGUF, please reload your model without "
                "quantization:\n"
                "  model = TurboModel.from_pretrained('your-model', "
                "quantize=False)\n"
                "  model.export('gguf', quantization='Q4_K_M')"
            )

    def optimize_inference(self, backend: str = "triton", bits: int = 4):
        """Optimize model for inference using high-performance kernels."""
        if backend == "triton":
            from ..kernels.triton import (
                TritonQuantizedLinear,
                is_triton_available,
            )

            if not is_triton_available():
                print_warning(
                    "Triton is not available or no GPU detected. "
                    "Skipping optimization."
                )
                return

            if self.verbose:
                print_header("Optimizing with Triton Kernels")

            count = self._replace_with_triton(self.model, bits)

            if self.verbose:
                print_success(
                    f"Optimized {count} layers with Triton fused kernels!"
                )

    def _replace_with_triton(self, module: nn.Module, bits: int) -> int:
        """Recursively replace Linear layers with TritonQuantizedLinear."""
        from ..kernels.triton import TritonQuantizedLinear

        count = 0
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if self.verbose:
                    print_info(f"Quantizing {name}...")

                quantized = TritonQuantizedLinear(
                    child.in_features,
                    child.out_features,
                    bits=bits,
                    bias=child.bias is not None,
                    group_size=128,
                )
                quantized.to(child.weight.device)
                quantized.quantize_from(child)

                setattr(module, name, quantized)
                count += 1
            else:
                count += self._replace_with_triton(child, bits)
        return count

    def __repr__(self) -> str:
        try:
            params = self.model.num_parameters() / 1e9
            params_str = f"{params:.2f}B"
        except Exception:
            params_str = "?"
        model_id = getattr(
            getattr(self.model, "config", None), "_name_or_path", "?"
        )
        return (
            "TurboModel(\n"
            f"  model={model_id},\n"
            f"  params={params_str},\n"
            f"  bits={self.config.bits},\n"
            f"  quantized={self.is_quantized},\n"
            f"  finetuned={self._is_finetuned}\n"
            ")"
        )


def register_architecture(
    architecture: str,
    *,
    base_model_type: Optional[str] = None,
    model_class: Optional[Type[PreTrainedModel]] = None,
) -> None:
    """Register a new architecture alias and optional explicit model class.

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
    """Load and quantize any LLM in one line.

    This is the simplest way to use QuantLLM. Everything is
    automatically configured based on your hardware.

    Args:
        model: HuggingFace model name or local path.
        bits: Override quantization bits (default: auto).
        max_length: Override max sequence length (default: auto).
        device: Override device (default: best GPU).
        dtype: Override dtype (default: bf16/fp16).
        base_model_fallback: Retry with resolved base model config.
        config: Shared export/push config.
        **kwargs: Additional options passed to from_pretrained.

    Returns:
        TurboModel ready for use.
    """
    return TurboModel.from_pretrained(
        model,
        bits=bits,
        max_length=max_length,
        device=device,
        dtype=dtype,
        base_model_fallback=base_model_fallback,
        config=config,
        trust_remote_code=True,
        **kwargs,
    )
