"""HuggingFace Hub publishing support for QuantLLM.

Provides the :class:`PushToHubMixin` used by :class:`TurboModel` for
the ``push_to_hub`` method.
"""

import os
from typing import Any, Optional

from .model_loader import DEFAULT_EXPORT_PUSH_CONFIG
from ..utils import (
    logger,
    print_header,
    print_info,
)


class PushToHubMixin:
    """Hub-push methods mixed into :class:`TurboModel`."""

    # -- declared by concrete class / other mixins (for mypy) --
    model: Any
    export_push_config: Any
    export: Any
    _export_onnx: Any
    _export_mlx: Any

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        format: Optional[str] = None,
        quantization: Optional[str] = None,
        commit_message: str = "Upload model via QuantLLM",
        license: str = "apache-2.0",
        private: bool = False,
        **kwargs,
    ):
        """Push model to HuggingFace Hub with proper model card.

        Args:
            repo_id: Repository ID (e.g. "username/model").
            token: HF Token.
            format: "safetensors", "gguf", "onnx", or "mlx".
            quantization: Quantization type (for gguf/onnx/mlx).
            commit_message: Commit message.
            license: License type (default: apache-2.0).
        """
        from ..hub import QuantLLMHubManager

        format_lower = (
            format
            if format is not None
            else self.export_push_config.get("push_format", DEFAULT_EXPORT_PUSH_CONFIG["push_format"])
        ).lower()
        push_quantization = quantization or self.export_push_config.get(
            "push_quantization",
            DEFAULT_EXPORT_PUSH_CONFIG["push_quantization"],
        )

        base_model_full = self.model.config._name_or_path
        model_name = base_model_full.split("/")[-1]

        print_header(f"Pushing to {repo_id}")
        print_info(f"Format: {format_lower.upper()}")
        print_info(f"Base model: {base_model_full}")

        manager = QuantLLMHubManager(repo_id=repo_id, hf_token=token)

        if format_lower == "gguf":
            quant_label = push_quantization or self.export_push_config.get(
                "quantization", DEFAULT_EXPORT_PUSH_CONFIG["quantization"]
            )
            filename = f"{model_name}.{quant_label.upper()}.gguf"
            save_path = os.path.join(manager.staging_dir, filename)

            self.export(
                format="gguf",
                output_path=save_path,
                quantization=quant_label,
                **kwargs,
            )

            manager.track_hyperparameters(
                {
                    "format": "gguf",
                    "quantization": quant_label.upper(),
                    "base_model": base_model_full,
                    "license": license,
                }
            )
            manager._generate_model_card(format="gguf")

        elif format_lower == "onnx":
            save_path = manager.staging_dir
            self._export_onnx(save_path, quantization=push_quantization, **kwargs)

            manager.track_hyperparameters(
                {
                    "format": "onnx",
                    "quantization": push_quantization,
                    "base_model": base_model_full,
                    "license": license,
                }
            )
            manager._generate_model_card(format="onnx")

        elif format_lower == "mlx":
            save_path = manager.staging_dir
            self._export_mlx(save_path, quantization=push_quantization, **kwargs)

            manager.track_hyperparameters(
                {
                    "format": "mlx",
                    "quantization": push_quantization,
                    "base_model": base_model_full,
                    "license": license,
                }
            )
            manager._generate_model_card(format="mlx")

        else:
            manager.track_hyperparameters(
                {
                    "format": format_lower,
                    "base_model": base_model_full,
                    "license": license,
                }
            )
            manager.save_final_model(self, format=format_lower)
            manager._generate_model_card(format=format_lower)

        manager.push(commit_message=commit_message)

    push = push_to_hub
