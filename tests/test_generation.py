"""Tests for text generation and chat functionality."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from quantllm.core.turbo_model import TurboModel
import quantllm.core.turbo_model as turbo_model_module


def _smart_config():
    return SimpleNamespace(
        bits=16,
        effective_loading_bits=16,
        dtype="float16",
        cpu_offload=False,
        device=torch.device("cpu"),
        gradient_checkpointing=False,
        use_flash_attention=False,
        compile_model=False,
        max_seq_length=4096,
        quant_type="Q4_K_M",
    )


def _tokenizer():
    """Returns a mock tokenizer compatible with and GenerationMixin.generate()."""

    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 2
            self.bos_token_id = 1
            self.pad_token_id = 0

        def __call__(self, model_inputs=None, return_tensors="pt", **kwargs):
            inputs = model_inputs or kwargs.get("text", "")
            if isinstance(inputs, str):
                inputs = [inputs]
            input_ids = torch.tensor([[1, 2, 3] for _ in inputs], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        def decode(self, token_ids, skip_special_tokens=False):
            return " ".join(["word"] * len(token_ids))

        def apply_chat_template(self, messages, add_generation_prompt=False):
            return "Hello, how can I help?"

    return MockTokenizer()


def _fake_model():
    """Returns a mock model with a working generate method."""
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="llama",
            _name_or_path="test/model",
            quantization_config=None,
        ),
        device=torch.device("cpu"),
    )

    class FakeOutput(list):
        pass

    def fake_generate(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        extra = min(kwargs.get("max_new_tokens", 10), 10)
        full_len = seq_len + extra
        return FakeOutput([torch.ones(full_len, dtype=torch.long)])

    model.generate = Mock(side_effect=fake_generate)
    model.num_parameters = Mock(return_value=7_000_000_000)
    model.parameters = Mock(return_value=iter([]))
    return model


def test_generate_returns_string(monkeypatch):
    """generate() should return a non-empty string."""
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})

    instance = TurboModel.__new__(TurboModel)
    instance.model = _fake_model()
    instance.tokenizer = _tokenizer()
    instance.config = _smart_config()
    instance._is_quantized_override = False
    instance._is_finetuned = False
    instance._lora_applied = False
    instance.verbose = False

    result = instance.generate("Hello", max_new_tokens=5)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_respects_max_new_tokens(monkeypatch):
    """generate() should pass max_new_tokens to model.generate."""
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})

    instance = TurboModel.__new__(TurboModel)
    instance.model = _fake_model()
    instance.tokenizer = _tokenizer()
    instance.config = _smart_config()
    instance._is_quantized_override = False
    instance._is_finetuned = False
    instance._lora_applied = False
    instance.verbose = False

    instance.generate("Hello", max_new_tokens=100)
    call_kwargs = instance.model.generate.call_args[1]
    assert call_kwargs["max_new_tokens"] == 100


def test_generate_defaults_to_sampling(monkeypatch):
    """generate() should enable do_sample by default."""
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})

    instance = TurboModel.__new__(TurboModel)
    instance.model = _fake_model()
    instance.tokenizer = _tokenizer()
    instance.config = _smart_config()
    instance._is_quantized_override = False
    instance._is_finetuned = False
    instance._lora_applied = False
    instance.verbose = False

    instance.generate("Hello")
    call_kwargs = instance.model.generate.call_args[1]
    assert call_kwargs["do_sample"] is True


def test_generate_greedy_when_do_sample_false(monkeypatch):
    """generate(do_sample=False) should use greedy decoding."""
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})

    instance = TurboModel.__new__(TurboModel)
    instance.model = _fake_model()
    instance.tokenizer = _tokenizer()
    instance.config = _smart_config()
    instance._is_quantized_override = False
    instance._is_finetuned = False
    instance._lora_applied = False
    instance.verbose = False

    instance.generate("Hello", do_sample=False)
    call_kwargs = instance.model.generate.call_args[1]
    assert call_kwargs["do_sample"] is False
    assert call_kwargs["temperature"] == 1.0


def test_chat_applies_chat_template(monkeypatch):
    """chat() should call apply_chat_template when available."""
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})

    instance = TurboModel.__new__(TurboModel)
    instance.model = _fake_model()
    tok = _tokenizer()
    tok.chat_template = "<|user|>\n{content}<|end|>\n<|assistant|>\n"
    tok.apply_chat_template = Mock(return_value="<|user|>\nHello<|end|>\n<|assistant|>\n")
    instance.tokenizer = tok
    instance.config = _smart_config()
    instance._is_quantized_override = False
    instance._is_finetuned = False
    instance._lora_applied = False
    instance.verbose = False

    result = instance.chat([{"role": "user", "content": "Hello"}])
    assert isinstance(result, str)
    tok.apply_chat_template.assert_called_once()


def test_chat_fallback_no_template(monkeypatch):
    """chat() should use default formatting when no chat_template."""
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})

    instance = TurboModel.__new__(TurboModel)
    instance.model = _fake_model()
    tok = _tokenizer()
    tok.chat_template = None
    instance.tokenizer = tok
    instance.config = _smart_config()
    instance._is_quantized_override = False
    instance._is_finetuned = False
    instance._lora_applied = False
    instance.verbose = False

    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]
    instance.generate = Mock(return_value="Hello there!")
    result = instance.chat(messages)
    assert result == "Hello there!"
    call_prompt = instance.generate.call_args[0][0]
    assert "System:" in call_prompt
    assert "User:" in call_prompt


def test_generate_handles_stop_strings(monkeypatch):
    """generate() should truncate at stop strings."""
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})

    instance = TurboModel.__new__(TurboModel)
    instance.model = _fake_model()
    tok = Mock()
    tok.pad_token = "<pad>"
    tok.pad_token_id = 0
    tok.eos_token = "</s>"
    tok.eos_token_id = 2
    tok.chat_template = None
    tok.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tok.decode = Mock(return_value="Some output before END rest ignored")
    instance.tokenizer = tok
    instance.config = _smart_config()
    instance._is_quantized_override = False
    instance._is_finetuned = False
    instance._lora_applied = False
    instance.verbose = False

    result = instance.generate("Hello", stop_strings=["END"])
    assert "END" not in result
