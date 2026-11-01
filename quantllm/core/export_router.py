"""Export routing for QuantLLM — GGUF, ONNX, MLX, SafeTensors.

Provides the :class:`ExportRouterMixin` used by :class:`TurboModel` for
the ``export`` method and its format-specific helpers.
"""

import os
import shutil
import tempfile
import time
from typing import Any, Optional

from .model_loader import DEFAULT_EXPORT_PUSH_CONFIG
from ..utils import (
    logger,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
    QuantLLMProgress,
    format_size,
    format_time,
)


class ExportRouterMixin:
    """Export methods mixed into :class:`TurboModel`."""

    # -- declared by concrete class / other mixins (for mypy) --
    model: Any
    tokenizer: Any
    config: Any
    verbose: bool
    export_push_config: Any
    _lora_applied: bool
    _is_bnb_quantized: Any  # method on TurboModel
    _dequantize_model: Any

    @staticmethod
    def list_available_formats() -> list:
        """Return formats that can be exported with the current environment.

        Checks for availability of optional dependencies (optimum, mlx-lm)
        without importing them eagerly.
        """
        available = ["safetensors"]
        try:
            import gguf

            available.append("gguf")
        except ImportError:
            pass
        try:
            import optimum.onnxruntime

            available.append("onnx")
        except ImportError:
            pass
        try:
            import mlx_lm

            available.append("mlx")
        except ImportError:
            pass
        return available

    def export(
        self,
        format: Optional[str] = None,
        output_path: Optional[str] = None,
        *,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Export model to various formats.

        Supported formats: gguf, safetensors, onnx, mlx.
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

        if self._lora_applied:
            if self.verbose:
                print_info("Merging LoRA weights before export...")
            self.model = self.model.merge_and_unload()
            self._lora_applied = False
            self._lora_merged = True

        if output_path is None:
            model_name = self.model.config._name_or_path.split("/")[-1]
            if format == "gguf":
                quant = effective_quantization or "Q4_K_M"
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
        imatrix: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Export to GGUF format using llama.cpp converter.

        ``imatrix`` is the path to a llama-imatrix file, required for IQ1/IQ2/IQ3_XXS types.
        """
        from ..quant import (
            convert_to_gguf,
            quantize_gguf,
            ensure_llama_cpp_installed,
            gguf_outtype,
        )
        from .memory import memory_optimized_tensor_order

        DEFAULT_CHUNKED_SHARD_SIZE = "2GB"

        start_time = time.time()

        effective_shard_size = max_shard_size or (DEFAULT_CHUNKED_SHARD_SIZE if chunked_conversion else None)

        quant_type = quantization or self.config.quant_type or "q4_k_m"
        quant_type_upper = quant_type.upper()
        quant_type_lower = quant_type.lower()

        # Full-precision targets (f16/bf16/f32) are written directly by the converter
        passthrough_outtype = gguf_outtype(quant_type_lower)
        needs_quantization = passthrough_outtype is None
        if passthrough_outtype is not None:
            quant_type_lower = passthrough_outtype
            quant_type_upper = passthrough_outtype.upper()

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

        if self.verbose:
            print_info("Checking llama.cpp installation...")
        ensure_llama_cpp_installed()

        model_to_save = self.model
        is_bnb_quantized = self._is_bnb_quantized()

        if is_bnb_quantized:
            if self.verbose:
                print_warning("Model is BitsAndBytes quantized. Dequantizing for GGUF export...")
                print_info("This may use significant memory. For large models, consider loading with quantize=False.")
            model_to_save = self._dequantize_model()
            if self.verbose:
                print_success("Model dequantized successfully!")

        # Converter --outtype: the target itself for passthrough types, F16 intermediate otherwise
        model_dtype = passthrough_outtype or "f16"
        model_name = self.model.config._name_or_path.split("/")[-1]
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        temp_parent = disk_offload_dir if disk_offloading else None
        if temp_parent:
            os.makedirs(temp_parent, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=temp_parent) as temp_dir:
            if self.verbose:
                print_header("Step 1/3: Saving Model", icon="[save]")
                print_info(f"Staging model to {temp_dir}...")

            with QuantLLMProgress() as progress:
                task = progress.add_task("Saving model weights...", total=None)
                save_kwargs: dict[str, Any] = {"safe_serialization": True}
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

            if self.verbose:
                print_header("Step 2/3: Converting to GGUF", icon="[convert]")

            # The converter writes into the staging dir, so the intermediate F16 file is
            # removed with it on both success and failure (never left in the CWD).
            output_files, _ = convert_to_gguf(
                model_name=model_name,
                input_folder=temp_dir,
                model_dtype=model_dtype,
                quantization_type=model_dtype,
                print_output=self.verbose,
                output_dir=temp_dir,
            )

            if not output_files or not os.path.exists(output_files[0]):
                raise RuntimeError("GGUF conversion failed to produce output file.")

            f16_gguf_file = output_files[0]

            if self.verbose:
                print_success(f"{model_dtype.upper()} GGUF created: {f16_gguf_file}")

            if needs_quantization:
                if self.verbose:
                    print_header(
                        f"Step 3/3: Quantizing to {quant_type_upper}",
                        icon="[quant]",
                    )
                    print_info(f"Applying {quant_type_upper} quantization...")

                quantized_file = os.path.join(temp_dir, f"{model_name}.{quant_type_upper}.gguf")

                quantize_gguf(
                    input_gguf=f16_gguf_file,
                    output_gguf=quantized_file,
                    quant_type=quant_type_upper,
                    print_output=self.verbose,
                    imatrix=imatrix,
                )

                final_file = quantized_file

                # Free the intermediate F16 file as soon as it is no longer needed
                if os.path.exists(f16_gguf_file) and f16_gguf_file != quantized_file:
                    os.remove(f16_gguf_file)

                if self.verbose:
                    print_success(f"Quantization complete: {quantized_file}")
            else:
                final_file = f16_gguf_file
                if self.verbose:
                    print_info("No quantization needed (already in target format)")

            if os.path.abspath(final_file) != os.path.abspath(output_path):
                if self.verbose:
                    print_info(f"Moving {final_file} -> {output_path}")
                shutil.move(final_file, output_path)

        if is_bnb_quantized and model_to_save is not self.model:
            del model_to_save
            if torch_available():
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        if self.verbose:
            file_size_bytes = os.path.getsize(output_path)
            print_header("Export Complete!", icon="[done]")
            print_info(f"Output: {output_path}")
            print_info(f"Format: GGUF {quant_type_upper}")
            print_info(f"Size: {format_size(file_size_bytes)}")
            print_info(f"Time: {format_time(elapsed)}")

        return output_path

    def _export_safetensors(self, output_path: str, **kwargs) -> str:
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
        **kwargs,
    ) -> str:
        """Export to ONNX format using Optimum."""
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
        except ImportError as e:
            import transformers

            error_msg = (
                "ONNX export requires optimum-onnx: pip install 'quantllm[onnx]'\n"
                "(or: pip install 'optimum-onnx[onnxruntime]' onnx onnxscript)"
            )
            if int(transformers.__version__.split(".")[0]) >= 5:
                error_msg += (
                    "\nNote: optimum-onnx currently supports transformers<4.58, but transformers "
                    f"{transformers.__version__} is installed. Export ONNX from a separate environment "
                    "with 'transformers<4.58'."
                )
            print_error(error_msg)
            raise ImportError(error_msg) from e

        os.makedirs(output_path, exist_ok=True)
        model_name = self.model.config._name_or_path

        if self.verbose:
            print_info("Using Optimum for ONNX export...")

        with QuantLLMProgress() as progress:
            task = progress.add_task("Exporting to ONNX...", total=None)

            try:
                if self._is_bnb_quantized() and not self._has_in_memory_changes():
                    # Unmodified BNB model: the original Hub weights are more accurate than a dequantized copy
                    if self.verbose:
                        print_info("BNB quantized model detected. Exporting from original HuggingFace model...")
                    ort_model = ORTModelForCausalLM.from_pretrained(
                        model_name,
                        export=True,
                        trust_remote_code=True,
                    )
                else:
                    # Export the in-memory weights (fine-tuned / LoRA-merged changes are preserved)
                    temp_path = self._stage_in_memory_model()
                    try:
                        ort_model = ORTModelForCausalLM.from_pretrained(
                            temp_path,
                            export=True,
                            trust_remote_code=True,
                        )
                    finally:
                        shutil.rmtree(temp_path, ignore_errors=True)

                ort_model.save_pretrained(output_path)
                self.tokenizer.save_pretrained(output_path)

            except Exception as e:
                progress.update(task, completed=100)
                error_str = str(e)
                if "onnxscript" in error_str.lower():
                    raise ImportError("ONNX export requires onnxscript: pip install onnxscript") from e
                raise

            progress.update(task, completed=100)

        if quantization:
            if self.verbose:
                print_info(f"Applying {quantization} ONNX quantization...")
            quantized_name = self._quantize_onnx_model(output_path, quantization)
            if quantized_name:
                # ORTModel loads model.onnx by default; the INT8 graph lives in a separate file
                print_info(f"Quantized ONNX model: {os.path.join(output_path, quantized_name)}")
                print_info(
                    f"Load it with ORTModelForCausalLM.from_pretrained('{output_path}', file_name='{quantized_name}')"
                )

        if self.verbose:
            print_success(f"ONNX model exported to {output_path}")

        return output_path

    def _has_in_memory_changes(self) -> bool:
        """True when the in-memory weights differ from the Hub checkpoint (fine-tuned or LoRA merged)."""
        return bool(
            getattr(self, "_is_finetuned", False)
            or getattr(self, "_lora_applied", False)
            or getattr(self, "_lora_merged", False)
        )

    def _stage_in_memory_model(self) -> str:
        """Save the in-memory model (dequantized if BNB) + tokenizer to a new temp dir and return it.

        The caller owns the directory and must remove it (``shutil.rmtree``).
        """
        model_to_save = self._dequantize_model() if self._is_bnb_quantized() else self.model
        staging_dir = tempfile.mkdtemp(prefix="quantllm_export_")
        try:
            model_to_save.save_pretrained(staging_dir, safe_serialization=True)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(staging_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        return staging_dir

    def _quantize_onnx_model(self, model_path: str, quant_type: str) -> Optional[str]:
        """Apply ONNX INT8 quantization.

        Returns the file name of the quantized model inside ``model_path``
        (e.g. ``model_quantized.onnx``), or ``None`` if quantization was skipped.
        """
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            quantizer = ORTQuantizer.from_pretrained(model_path)
            quant_lower = quant_type.lower().replace("_", "").replace("-", "")

            bit_request = None
            for bit_pattern in [
                "2bit",
                "3bit",
                "4bit",
                "5bit",
                "6bit",
                "q2",
                "q3",
                "q4",
                "q5",
                "q6",
            ]:
                if bit_pattern in quant_lower:
                    bit_request = bit_pattern
                    break

            if bit_request:
                print_warning(f"ONNX only supports INT8 (8-bit) quantization, not {quant_type}.")
                print_info("For lower bit quantization, use GGUF format instead.")
                print_info("Proceeding with INT8 quantization...")

            if "avx512" in quant_lower or "vnni" in quant_lower:
                qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
            elif "arm64" in quant_lower or "arm" in quant_lower:
                qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=True)
            elif "static" in quant_lower:
                qconfig = AutoQuantizationConfig.avx2(is_static=True, per_channel=True)
            else:
                qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)

            file_suffix = "quantized"
            quantizer.quantize(save_dir=model_path, quantization_config=qconfig, file_suffix=file_suffix)

            source_onnx = getattr(quantizer, "onnx_model_path", None)
            source_stem = os.path.splitext(os.path.basename(str(source_onnx)))[0] if source_onnx else "model"
            quantized_name = f"{source_stem}_{file_suffix}.onnx"

            if self.verbose:
                print_success(f"ONNX INT8 quantization applied successfully ({quantized_name})")
            return quantized_name

        except ImportError:
            print_warning("Optimum quantization not available. Skipping ONNX quantization.")
            print_info("Install with: pip install optimum[onnxruntime]")
        except Exception as e:
            print_warning(f"ONNX quantization failed: {e}")
            print_info("The unquantized ONNX model is still available.")
        return None

    @staticmethod
    def _mlx_quant_bits(quantization: Optional[str]) -> Optional[int]:
        """Map a quantization name to MLX ``q_bits`` (4 or 8), or ``None`` for no quantization.

        Full-precision names (f16/bf16/fp16/f32/none) are checked first so that e.g. the
        "16" in "bf16" is not mistaken for a 6-bit request.
        """
        if not quantization:
            return None
        quant_lower = quantization.lower().replace("_", "").replace("-", "")
        if quant_lower in {"f16", "bf16", "fp16", "f32", "fp32", "float16", "bfloat16", "float32", "none"}:
            return None

        if any(x in quant_lower for x in ["2", "3"]):
            print_warning(f"MLX only supports 4-bit and 8-bit, not {quantization}.")
            print_info("Using 4-bit quantization (smallest available).")
            return 4
        if any(x in quant_lower for x in ["5", "6", "7"]):
            print_warning(f"MLX only supports 4-bit and 8-bit, not {quantization}.")
            print_info("Using 8-bit quantization (closest available).")
            return 8
        if "8" in quant_lower:
            return 8
        return 4

    def _export_mlx(
        self,
        output_path: str,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Export to MLX format for Apple Silicon."""
        import platform

        if platform.system() != "Darwin" or platform.machine() != "arm64":
            print_warning("MLX export is optimized for Apple Silicon Macs.")
            print_info("The model will be saved but may not run efficiently on this system.")

        try:
            import mlx

            HAS_MLX = True
        except ImportError:
            HAS_MLX = False

        # Don't pre-create output_path: mlx_lm.convert refuses to write into an existing directory
        model_name = self.model.config._name_or_path

        if HAS_MLX:
            try:
                from mlx_lm import convert

                if self.verbose:
                    print_info("Using mlx-lm for conversion...")

                with QuantLLMProgress() as progress:
                    task = progress.add_task("Converting to MLX...", total=None)

                    staging_dir = None
                    try:
                        if self._is_bnb_quantized() and not self._has_in_memory_changes():
                            # Unmodified BNB model: convert from the original Hub weights
                            source_path = model_name
                        else:
                            # Convert the in-memory weights (fine-tuned / LoRA-merged changes are preserved)
                            staging_dir = self._stage_in_memory_model()
                            source_path = staging_dir

                        convert_args: dict[str, Any] = {
                            "hf_path": source_path,
                            "mlx_path": output_path,
                        }
                        q_bits = self._mlx_quant_bits(quantization)
                        if q_bits is not None:
                            convert_args["quantize"] = True
                            convert_args["q_bits"] = q_bits

                        convert(**convert_args)
                    finally:
                        if staging_dir is not None:
                            shutil.rmtree(staging_dir, ignore_errors=True)

                    progress.update(task, completed=100)

            except Exception as e:
                print_error(f"MLX conversion failed: {e}")
                raise
        else:
            if self.verbose:
                print_warning("mlx-lm not installed. Saving as HuggingFace format.")
                print_info("To convert to MLX: pip install mlx-lm && python -m mlx_lm.convert ...")

            os.makedirs(output_path, exist_ok=True)
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)

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


def torch_available() -> bool:
    try:
        import torch

        return True
    except ImportError:
        return False
