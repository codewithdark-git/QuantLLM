"""Fine-tuning support for QuantLLM via LoRA.

Provides the :class:`FinetuneMixin` used by :class:`TurboModel` for
the ``finetune`` method.
"""

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

        if self._is_quantized:
            self.model = prepare_model_for_kbit_training(self.model)

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
                optim=("paged_adamw_32bit" if self.config.bits <= 4 else "adamw_torch"),
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
                "lora_r": r,
            }

        except Exception as e:
            print_error(f"Training failed: {e}")
            if "out of memory" in str(e).lower():
                print_info("Tip: Try reducing batch_size or enabling gradient_checkpointing in config.")
            raise

    def _get_lora_target_modules(self) -> List[str]:
        """Get appropriate LoRA target modules for the model."""
        model_type = getattr(self.model.config, "model_type", "").lower()

        target_patterns = {
            "llama": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "mistral": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "qwen2": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            "gemma": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        }

        return target_patterns.get(model_type, ["q_proj", "v_proj"])

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

        def tokenize_function(examples):
            if "text" in examples:
                texts = examples["text"]
            elif "instruction" in examples and "output" in examples:
                texts = [
                    f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                    for inst, out in zip(examples["instruction"], examples["output"])
                ]
            else:
                keys = [k for k in examples.keys() if isinstance(examples[k][0], str)]
                texts = [" ".join([examples[k][i] for k in keys]) for i in range(len(examples[keys[0]]))]

            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
            desc="Tokenizing dataset",
        )

        return tokenized
