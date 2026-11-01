"""
TurboModel - The Ultra-Simple QuantLLM API.

Load, quantize, fine-tune, and export LLMs with one line each.

This module composes domain-specific mixins into the unified
:class:`TurboModel` class exposed to users.
"""

from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..utils.hf_compat import dtype_kwargs
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
        self.export_push_config = ModelLoaderMixin._build_export_push_config(export_push_config)
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

        model_to_save.save_pretrained(output_dir, safe_serialization=safe_serialization)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

        if self.verbose:
            print_success(f"Model saved to {output_dir}")

        return output_dir

    def _is_bnb_quantized(self) -> bool:
        """Return True if the loaded model is BitsAndBytes-quantized."""
        if hasattr(self.model, "config"):
            quant_config = getattr(self.model.config, "quantization_config", None)
            if quant_config:
                quant_method = getattr(quant_config, "quant_method", None)
                if quant_method in ("bitsandbytes", "bnb"):
                    return True
                if getattr(quant_config, "load_in_4bit", False):
                    return True
                if getattr(quant_config, "load_in_8bit", False):
                    return True

        # Check for bnb modules (handle from_config_only mock models)
        if hasattr(self.model, "modules"):
            try:
                import bitsandbytes as bnb

                for module in self.model.modules():
                    if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                        return True
            except ImportError:
                pass

        return False

    def _has_runtime_quantization(self) -> bool:
        """Return True if the loaded model carries any quantization."""
        if self._is_bnb_quantized():
            return True

        if hasattr(self.model, "config"):
            quant_config = getattr(self.model.config, "quantization_config", None)
            if quant_config:
                quant_method = getattr(quant_config, "quant_method", None)
                if quant_method:
                    return True
                if isinstance(quant_config, dict) and quant_config.get("quant_method"):
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
        self._is_quantized_override = None if value is None else bool(value)

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
            quant_config = getattr(self.model.config, "quantization_config", None)
            if quant_config is not None:
                quant_method = getattr(quant_config, "quant_method", None) or (
                    quant_config.get("quant_method") if isinstance(quant_config, dict) else None
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
            "effective_loading_bits": getattr(self.config, "effective_loading_bits", None),
            "is_quantized": self.is_quantized,
            "quant_method": quant_method,
            "device": str(device) if device is not None else None,
            "dtype": str(dtype) if dtype is not None else None,
            "finetuned": self._is_finetuned,
            "lora_applied": self._lora_applied,
        }

    def _dequantize_model(self) -> nn.Module:
        """Dequantize a BitsAndBytes model to full precision for GGUF export.

        Works on the in-memory model so fine-tuned weights are preserved:
        LoRA adapters are merged first, then ``model.dequantize()``
        (transformers) is used, then a manual bnb dequantization. Reloading
        the base weights from the Hub is only a last resort and only when the
        model was never fine-tuned.
        """
        import gc

        target_dtype = self.config.dtype if self.config.dtype in [torch.float16, torch.bfloat16] else torch.float16
        finetuned = bool(self._is_finetuned or self._lora_applied)

        # 1. Merge LoRA adapters into the (quantized) base weights.
        if self._lora_applied or hasattr(self.model, "merge_and_unload"):
            try:
                if self.verbose:
                    print_info("Merging LoRA weights before dequantization...")
                self.model = self.model.merge_and_unload()
                self._lora_applied = False
            except Exception as e:
                raise RuntimeError(f"Failed to merge LoRA adapters before dequantization: {e}") from e

        model = self.model
        errors: List[str] = []

        # 2. Native transformers dequantization (in place).
        if hasattr(model, "dequantize"):
            try:
                try:
                    dequant_model = model.dequantize(dtype=target_dtype)  # transformers >= 5
                except TypeError:
                    dequant_model = model.dequantize()  # transformers 4.x
                if dequant_model is None:
                    dequant_model = model
                dequant_model = dequant_model.to(target_dtype)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return dequant_model
            except Exception as e:
                errors.append(f"model.dequantize(): {e}")
                if self.verbose:
                    print_warning(f"model.dequantize() failed: {e}")
                    print_info("Attempting manual in-place dequantization...")

        # 3. Manual dequantization of bnb layers + copy of every other tensor.
        try:
            dequant_model = self._manual_bnb_dequantize(model, target_dtype)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return dequant_model
        except Exception as e:
            errors.append(f"manual dequantization: {e}")

        # 4. Last resort: reload base weights from the Hub (never after fine-tuning).
        model_name = getattr(model.config, "_name_or_path", None)
        if model_name and not finetuned:
            print_warning(
                f"In-memory dequantization failed ({'; '.join(errors)}). "
                f"Reloading base weights of {model_name} in full precision."
            )
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **dtype_kwargs(target_dtype),
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )
            except Exception as e:
                errors.append(f"reload from {model_name}: {e}")

        hint = (
            "Fine-tuned weights cannot be recovered by reloading from the Hub; "
            "save the adapter/model and re-export from a full-precision load."
            if finetuned
            else "To export to GGUF, reload your model without quantization:\n"
            "  model = TurboModel.from_pretrained('your-model', quantize=False)\n"
            "  model.export('gguf', quantization='Q4_K_M')"
        )
        raise RuntimeError(f"Failed to dequantize BitsAndBytes model: {'; '.join(errors)}\n\n{hint}")

    @staticmethod
    def _manual_bnb_dequantize(model: nn.Module, target_dtype: torch.dtype) -> nn.Module:
        """Rebuild ``model`` in ``target_dtype`` with bnb layers dequantized."""
        import copy

        import bitsandbytes as bnb

        new_state: Dict[str, torch.Tensor] = {}
        bnb_modules = set()

        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, bnb.nn.Linear4bit):
                    weight = module.weight
                    dequant = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)  # type: ignore[attr-defined]
                elif isinstance(module, bnb.nn.Linear8bitLt):
                    weight = module.weight
                    scb = getattr(weight, "SCB", None)
                    if scb is None:
                        scb = getattr(getattr(module, "state", None), "SCB", None)
                    if scb is None:
                        if weight.dtype == torch.int8:
                            raise RuntimeError(f"Missing SCB scales for 8-bit layer '{name}'")
                        dequant = weight.data
                    else:
                        dequant = weight.data.float() * scb.float()[:, None] / 127.0
                else:
                    continue
                bnb_modules.add(name)
                new_state[f"{name}.weight"] = dequant.to("cpu", target_dtype)
                if module.bias is not None:
                    new_state[f"{name}.bias"] = module.bias.data.to("cpu", target_dtype)

            def _owned_by_bnb(key: str) -> bool:
                parts = key.split(".")
                return any(".".join(parts[:i]) in bnb_modules for i in range(1, len(parts)))

            for key, tensor in model.state_dict().items():
                if key in new_state or _owned_by_bnb(key):
                    continue
                tensor = tensor.detach().to("cpu")
                new_state[key] = tensor.to(target_dtype) if tensor.is_floating_point() else tensor

        config = copy.deepcopy(model.config)
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        dequant_model = AutoModelForCausalLM.from_config(config, **dtype_kwargs(target_dtype))
        result = dequant_model.load_state_dict(new_state, strict=False)
        missing = [k for k in result.missing_keys if not k.endswith("lm_head.weight")]
        if missing:
            raise RuntimeError(f"Dequantized state dict is missing {len(missing)} tensors (e.g. {missing[:3]})")
        if hasattr(dequant_model, "tie_weights"):
            dequant_model.tie_weights()
        return dequant_model

    def optimize_inference(self, backend: str = "triton", bits: int = 4):
        """Optimize model for inference using high-performance kernels.

        No-op for BitsAndBytes-quantized models and when CUDA is unavailable.
        """
        if not torch.cuda.is_available():
            print_warning("optimize_inference requires a CUDA GPU; skipping.")
            return
        if self._is_bnb_quantized():
            print_warning("optimize_inference is a no-op for BitsAndBytes-quantized models; skipping.")
            return

        if backend == "triton":
            from ..kernels.triton import (
                TritonQuantizedLinear,
                is_triton_available,
            )

            if not is_triton_available():
                print_warning("Triton is not available or no GPU detected. Skipping optimization.")
                return

            if self.verbose:
                print_header("Optimizing with Triton Kernels")

            count = self._replace_with_triton(self.model, bits)

            if self.verbose:
                print_success(f"Optimized {count} layers with Triton fused kernels!")

    def _replace_with_triton(self, module: nn.Module, bits: int) -> int:
        """Recursively replace Linear layers with TritonQuantizedLinear."""
        from ..kernels.triton import TritonQuantizedLinear

        count = 0
        for name, child in module.named_children():
            # bnb layers subclass nn.Linear but hold packed weights; keep lm_head in full precision.
            if name == "lm_head" or type(child).__name__ in ("Linear4bit", "Linear8bitLt"):
                continue
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
        trust_remote_code=kwargs.pop("trust_remote_code", False),
        **kwargs,
    )
