"""Fine-tuning support for QuantLLM via LoRA.

Provides the :class:`FinetuneMixin` used by :class:`TurboModel` for
the ``finetune`` method.
"""

import inspect
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import torch

from ..utils import (
    logger,
    print_error,
    print_header,
    print_info,
    print_success,
)


class FinetuneMixin:
    """Fine-tuning methods mixed into :class:`TurboModel`."""

    # -- declared by concrete class (for mypy) --
    model: Any
    tokenizer: Any
    config: Any
    _is_quantized: bool
    _is_finetuned: bool
    _lora_applied: bool
    _is_bnb_quantized: Any  # method on TurboModel

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
        """Fine-tune the model with LoRA.

        Args:
            data: Training data - file path, list of dicts, or HF dataset.
            epochs: Number of training epochs (default: auto).
            learning_rate: Learning rate (default: auto based on model size).
            batch_size: Batch size (default: auto based on GPU memory).
            lora_r: LoRA rank (default: auto).
            lora_alpha: LoRA alpha (default: 2x lora_r).
            output_dir: Where to save (default: ./output/{model_name}).
            hub_manager: QuantLLMHubManager instance for auto-tracking.
            **kwargs: Additional training arguments.

        Returns:
            Training results dictionary.
        """
        print_header("Starting Fine-tuning")

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if "WANDB_DISABLED" not in os.environ:
            os.environ["WANDB_DISABLED"] = "true"

        warnings.filterwarnings("ignore", module="peft")
        warnings.filterwarnings("ignore", category=FutureWarning)

        try:
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            from transformers import (
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling,
            )
        except ImportError:
            raise ImportError("peft is required for fine-tuning. Install with: pip install peft")

        self._ensure_distinct_pad_token()

        if not self._lora_applied:
            if self._is_quantized:
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=bool(self.config.gradient_checkpointing),
                )
        else:
            # Second finetune(): the model is already a PeftModel. Don't re-run
            # prepare_model_for_kbit_training (it would freeze the adapters);
            # just make sure the LoRA weights are trainable again.
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        r = lora_r or (16 if self.model.num_parameters() < 10e9 else 64)
        alpha = lora_alpha or (r * 2)

        target_modules = self._get_lora_target_modules()

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if not self._lora_applied:
            self.model = get_peft_model(self.model, lora_config)
            self._lora_applied = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print_info(f"LoRA applied: {trainable:,} trainable params ({100 * trainable / total:.2f}%)")

        train_dataset = self._prepare_dataset(data)

        epochs = epochs or 3
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or 2e-4
        output_dir = output_dir or (f"./output/{self.model.config._name_or_path.split('/')[-1]}")

        if hub_manager:
            hub_manager.track_hyperparameters(
                {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "lora_r": r,
                    "lora_alpha": alpha,
                    "base_model": getattr(self.config, "model_name", "unknown"),
                    "output_dir": output_dir,
                }
            )

        # torch_compile is not supported with quantized models + PEFT
        # Check effective_loading_bits from SmartConfig (4 or 8 bit = quantized)
        torch_compile = self.config.compile_model and self.config.effective_loading_bits >= 16

        try:
            # Defaults first; user kwargs override them instead of raising "multiple values".
            args_dict: Dict[str, Any] = dict(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=learning_rate,
                fp16=self.config.dtype == torch.float16,
                bf16=self.config.dtype == torch.bfloat16,
                logging_steps=10,
                save_strategy="epoch",
                # warmup_ratio was removed in transformers 5.x; warmup_steps works everywhere.
                warmup_steps=max(
                    1,
                    int(
                        0.03
                        * epochs
                        * math.ceil(len(train_dataset) / (batch_size * self.config.gradient_accumulation_steps))
                    ),
                ),
                lr_scheduler_type="cosine",
                gradient_checkpointing=self.config.gradient_checkpointing,
                optim=("paged_adamw_32bit" if self._is_bnb_quantized() else "adamw_torch"),
                torch_compile=torch_compile,
            )
            args_dict.update(kwargs)
            training_args = TrainingArguments(**args_dict)

            trainer_kwargs: Dict[str, Any] = {
                "model": self.model,
                "args": training_args,
                "train_dataset": train_dataset,
                "data_collator": DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            }
            # ``processing_class`` replaced ``tokenizer`` in transformers 4.46.
            if "processing_class" in inspect.signature(Trainer.__init__).parameters:
                trainer_kwargs["processing_class"] = self.tokenizer
            else:
                trainer_kwargs["tokenizer"] = self.tokenizer

            trainer = Trainer(**trainer_kwargs)

            result = trainer.train()
            self._is_finetuned = True

            trainer.save_model(output_dir)
            if hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(output_dir)

            print_success(f"Training complete! Model saved to {output_dir}")

            step_losses = [h["loss"] for h in (getattr(trainer.state, "log_history", None) or []) if "loss" in h]
            return {
                # train_loss is the mean over all steps, so it is inflated by early steps that
                # fp16 loss scaling skips; final_loss is the last logged step.
                "train_loss": result.training_loss,
                "final_loss": step_losses[-1] if step_losses else result.training_loss,
                "epochs": epochs,
                "output_dir": output_dir,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "lora_r": r,
                "log_history": list(getattr(trainer.state, "log_history", None) or []),
            }

        except Exception as e:
            print_error(f"Training failed: {e}")
            if "out of memory" in str(e).lower():
                print_info("Tip: Try reducing batch_size or enabling gradient_checkpointing in config.")
            raise
        finally:
            self._restore_inference_mode()

    def _restore_inference_mode(self) -> None:
        """Switch the model back to inference mode after training."""
        model = self.model
        try:
            if getattr(model, "is_gradient_checkpointing", False) is True:
                model.gradient_checkpointing_disable()
            if callable(getattr(model, "eval", None)):
                model.eval()
            cfg = getattr(model, "config", None)
            if cfg is not None and hasattr(cfg, "use_cache"):
                cfg.use_cache = True
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Could not restore inference mode: {e}")

    def _ensure_distinct_pad_token(self) -> None:
        """Make pad != eos so the LM collator doesn't mask EOS labels."""
        tok = self.tokenizer
        eos = getattr(tok, "eos_token", None)
        pad = getattr(tok, "pad_token", None)
        if pad is not None and pad != eos:
            return
        unk = getattr(tok, "unk_token", None)
        if unk is not None and unk != eos:
            # Prefer unk: avoids resizing embeddings (problematic on quantized models).
            tok.pad_token = unk
            return
        tok.add_special_tokens({"pad_token": "<|pad|>"})
        try:
            self.model.resize_token_embeddings(len(tok))
        except Exception as e:
            logger.warning(f"Could not resize token embeddings for new pad token: {e}")

    def _get_lora_target_modules(self) -> Union[List[str], str]:
        """Get appropriate LoRA target modules for the model.

        Uses the architecture table in :class:`LoRAAutoConfig`; falls back to
        peft's ``"all-linear"`` for unknown architectures.
        """
        from .training import LoRAAutoConfig

        try:
            model_type = LoRAAutoConfig._detect_model_type(self.model)
        except Exception:
            model_type = (getattr(getattr(self.model, "config", None), "model_type", "") or "").lower()

        modules = LoRAAutoConfig.TARGET_MODULES.get(model_type) if model_type != "default" else None
        if not modules:
            return "all-linear"

        try:
            present = {name.split(".")[-1] for name, _ in self.model.named_modules()}
        except Exception:
            return list(modules)

        valid = [m for m in modules if m in present]
        return valid if valid else "all-linear"

    def _prepare_dataset(self, data: Union[str, List[Dict], Any]) -> Any:
        """Prepare dataset for training."""
        from datasets import Dataset, load_dataset

        if isinstance(data, str):
            if data.endswith(".json") or data.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=data)["train"]
            else:
                dataset = load_dataset(data)["train"]
        elif isinstance(data, list):
            dataset = Dataset.from_list(data)
        else:
            dataset = data

        from .training import format_training_texts

        tokenizer = self.tokenizer

        def tokenize_function(examples):
            texts, templated = format_training_texts(examples, tokenizer)

            return tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
                add_special_tokens=not templated,
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc="Tokenizing dataset",
        )

        return tokenized
