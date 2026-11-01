"""Regression tests for fine-tuning / training-data fixes (CPU only, tiny mocks)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from quantllm.core.training import (
    _find_text_column,
    format_training_texts,
    load_training_data,
)
from quantllm.core.turbo_model import TurboModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal stand-in for ``datasets.Dataset`` (column_names, [0], map)."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0].keys())

    def __getitem__(self, idx):
        return self.rows[idx]

    def map(self, fn, batched=True, remove_columns=None, **kwargs):
        batch = {k: [r[k] for r in self.rows] for k in self.column_names}
        return fn(batch)


class _CharTokenizer:
    eos_token = "</s>"
    pad_token = "<pad>"
    eos_token_id = 2
    pad_token_id = 0
    chat_template = None

    def __init__(self):
        self.texts = None

    def __call__(self, texts, truncation=True, max_length=8, padding=False, add_special_tokens=True, **kw):
        self.texts = list(texts)
        input_ids, attention_mask = [], []
        for t in texts:
            ids = [10 + (ord(c) % 50) for c in t][:max_length]
            mask = [1] * len(ids)
            if padding == "max_length":
                pad = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad
                mask = mask + [0] * pad
            input_ids.append(ids)
            attention_mask.append(mask)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


ALPACA_ROWS = [
    {"instruction": "Translate", "input": "bonjour", "output": "hello"},
    {"instruction": "Say hi", "input": "", "output": "hi"},
]


# ---------------------------------------------------------------------------
# training.py: column detection + label masking
# ---------------------------------------------------------------------------


def test_alpaca_prefers_instruction_output_over_input():
    ds = _FakeDataset(ALPACA_ROWS)
    assert _find_text_column(ds) == "__instruction_output__"


def test_messages_column_detected_before_text():
    ds = _FakeDataset([{"messages": [{"role": "user", "content": "x"}], "text": "x"}])
    assert _find_text_column(ds) == "__messages__:messages"


def test_load_training_data_includes_input_and_masks_padding():
    tok = _CharTokenizer()
    result = load_training_data(_FakeDataset(ALPACA_ROWS), tok, max_length=200)

    assert "### Input:\nbonjour" in tok.texts[0]
    assert "### Input:" not in tok.texts[1]  # empty input is omitted

    for ids, mask, labels in zip(result["input_ids"], result["attention_mask"], result["labels"]):
        for i, m, lab in zip(ids, mask, labels):
            assert lab == (i if m else -100)
    assert -100 in result["labels"][1]
    # labels must not alias input_ids
    assert result["labels"][1] is not result["input_ids"][1]
    assert tok.pad_token_id in result["input_ids"][1]


def test_format_training_texts_appends_eos_and_uses_chat_template():
    tok = _CharTokenizer()
    texts, templated = format_training_texts({k: [r[k] for r in ALPACA_ROWS] for k in ALPACA_ROWS[0]}, tok)
    assert templated is False
    assert all(t.endswith("</s>") for t in texts)
    assert "bonjour" in texts[0]

    tok.chat_template = "{{ bos_token }}..."
    tok.apply_chat_template = Mock(return_value="<s><user>Translate\n\nbonjour<asst>hello</s>")
    texts, templated = format_training_texts({k: [r[k] for r in ALPACA_ROWS] for k in ALPACA_ROWS[0]}, tok)
    assert templated is True
    assert texts[0].count("</s>") == 1  # not doubled
    msgs = tok.apply_chat_template.call_args_list[0][0][0]
    assert msgs[0] == {"role": "user", "content": "Translate\n\nbonjour"}
    assert msgs[1] == {"role": "assistant", "content": "hello"}


def test_format_training_texts_sharegpt_conversations():
    tok = _CharTokenizer()
    batch = {"conversations": [[{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]]}
    texts, templated = format_training_texts(batch, tok)
    assert templated is False
    assert texts == ["User: Q\nAssistant: A\n</s>"]


# ---------------------------------------------------------------------------
# finetune_mixin.py
# ---------------------------------------------------------------------------


def _instance(monkeypatch, model, tok=None, quantized=False):
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})
    inst = TurboModel.__new__(TurboModel)
    inst.model = model
    inst.tokenizer = tok or _CharTokenizer()
    inst.config = SimpleNamespace(
        bits=4,
        effective_loading_bits=16,
        dtype=torch.float32,
        gradient_checkpointing=False,
        compile_model=False,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_length=32,
    )
    inst._is_quantized_override = quantized
    inst._is_finetuned = False
    inst._lora_applied = False
    inst.verbose = False
    return inst


class _Proj(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            setattr(self, name, nn.Linear(2, 2))


class _Phi3Like(nn.Module):
    def __init__(self, model_type="phi3"):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type, _name_or_path="org/tiny", use_cache=False)
        self.layer = _Proj()


def test_target_modules_phi3(monkeypatch):
    inst = _instance(monkeypatch, _Phi3Like("phi3"))
    assert inst._get_lora_target_modules() == ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


def test_target_modules_unknown_arch_uses_all_linear(monkeypatch):
    inst = _instance(monkeypatch, SimpleNamespace(config=SimpleNamespace(model_type="brand_new_arch")))
    assert inst._get_lora_target_modules() == "all-linear"


def test_target_modules_known_type_but_missing_modules_uses_all_linear(monkeypatch):
    # falcon table but the model actually has phi3-style names
    inst = _instance(monkeypatch, _Phi3Like("falcon"))
    assert inst._get_lora_target_modules() == "all-linear"


def test_optimizer_choice_follows_bnb_quantization(monkeypatch):
    bnb = SimpleNamespace(config=SimpleNamespace(quantization_config=SimpleNamespace(quant_method="bitsandbytes")))
    inst = _instance(monkeypatch, bnb)
    assert inst._is_bnb_quantized() is True
    hqq = SimpleNamespace(config=SimpleNamespace(quantization_config=SimpleNamespace(quant_method="hqq")))
    inst.model = hqq
    assert inst._is_bnb_quantized() is False


def test_pad_token_made_distinct_from_eos(monkeypatch):
    tok = SimpleNamespace(eos_token="</s>", pad_token="</s>", unk_token="<unk>")
    inst = _instance(monkeypatch, SimpleNamespace(), tok=tok)
    inst._ensure_distinct_pad_token()
    assert tok.pad_token == "<unk>"


def test_pad_token_added_when_no_unk(monkeypatch):
    tok = Mock(eos_token="<eot>", pad_token=None, unk_token=None)
    tok.__len__ = Mock(return_value=11)
    model = SimpleNamespace(resize_token_embeddings=Mock())
    inst = _instance(monkeypatch, model, tok=tok)
    inst._ensure_distinct_pad_token()
    tok.add_special_tokens.assert_called_once_with({"pad_token": "<|pad|>"})
    model.resize_token_embeddings.assert_called_once_with(11)


class _LoraModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="phi3", _name_or_path="org/tiny", use_cache=False)
        self.base = nn.Linear(2, 2)
        self.lora_A = nn.Linear(2, 1, bias=False)
        self.is_gradient_checkpointing = False

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def test_second_finetune_keeps_adapters_trainable_and_saves(monkeypatch):
    pytest.importorskip("peft")
    import peft
    import transformers

    prepare = Mock(side_effect=lambda m, **kw: m)
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", prepare)
    monkeypatch.setattr(peft, "get_peft_model", Mock(side_effect=lambda m, cfg: m))
    monkeypatch.setattr(peft, "LoraConfig", Mock())

    captured = {}

    class FakeArgs:
        def __init__(self, **kwargs):
            captured["args"] = kwargs

    class FakeTrainer:
        def __init__(self, model=None, args=None, train_dataset=None, data_collator=None, processing_class=None):
            captured["trainer_kwargs"] = {"processing_class": processing_class}
            self.state = SimpleNamespace(log_history=[{"loss": 1.0}])
            self.save_model = Mock()
            captured["trainer"] = self

        def train(self):
            return SimpleNamespace(training_loss=0.5)

    monkeypatch.setattr(transformers, "TrainingArguments", FakeArgs)
    monkeypatch.setattr(transformers, "Trainer", FakeTrainer)
    monkeypatch.setattr(transformers, "DataCollatorForLanguageModeling", Mock())

    model = _LoraModel()
    tok = _CharTokenizer()
    tok.save_pretrained = Mock()
    inst = _instance(monkeypatch, model, tok=tok, quantized=True)
    inst._prepare_dataset = Mock(return_value=[])

    out1 = inst.finetune([], output_dir="out_dir", epochs=1)
    assert prepare.call_count == 1
    assert out1["log_history"] == [{"loss": 1.0}]
    captured["trainer"].save_model.assert_called_once_with("out_dir")
    tok.save_pretrained.assert_called_once_with("out_dir")
    assert captured["trainer_kwargs"]["processing_class"] is tok
    # not bnb-quantized -> no paged optimizer
    assert captured["args"]["optim"] == "adamw_torch"
    # back in inference mode
    assert model.training is False
    assert model.config.use_cache is True

    # Simulate prepare_model_for_kbit_training freezing everything
    for p in model.parameters():
        p.requires_grad_(False)

    inst.finetune([], output_dir="out_dir", epochs=1)
    assert prepare.call_count == 1  # not re-run on the PeftModel
    assert model.lora_A.weight.requires_grad is True
    assert model.base.weight.requires_grad is False
