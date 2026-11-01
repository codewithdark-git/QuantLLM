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


# ---------------------------------------------------------------------------
# Regression tests for generation fixes
# ---------------------------------------------------------------------------


def _instance(monkeypatch, model=None, tok=None, config=None):
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})
    instance = TurboModel.__new__(TurboModel)
    instance.model = model or _fake_model()
    instance.tokenizer = tok or _tokenizer()
    instance.config = config or _smart_config()
    instance._is_quantized_override = False
    instance._is_finetuned = False
    instance._lora_applied = False
    instance.verbose = False
    return instance


class _BosTokenizer:
    """Tokenizer that maps the literal "<s>" to BOS and prepends BOS when asked."""

    bos_token = "<s>"
    eos_token = "</s>"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "{{ bos_token }}..."
    truncation_side = "right"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize is False
        return "<s>[INST] " + messages[-1]["content"] + " [/INST]"

    def __call__(self, text, return_tensors="pt", add_special_tokens=True, **kwargs):
        self.calls.append({"add_special_tokens": add_special_tokens, "truncation_side": self.truncation_side, **kwargs})
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        rest = text
        if rest.startswith("<s>"):
            ids.append(self.bos_token_id)
            rest = rest[3:]
        ids.extend(10 + (ord(c) % 50) for c in rest)
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, token_ids, skip_special_tokens=False):
        return "ok"


def test_chat_does_not_double_bos(monkeypatch):
    tok = _BosTokenizer()
    instance = _instance(monkeypatch, tok=tok)
    instance.chat([{"role": "user", "content": "Hi"}])
    input_ids = instance.model.generate.call_args[1]["input_ids"][0].tolist()
    assert input_ids.count(tok.bos_token_id) == 1
    assert tok.calls[-1]["add_special_tokens"] is False


def test_generate_does_not_override_eos_token_id(monkeypatch):
    instance = _instance(monkeypatch)
    instance.generate("Hello")
    call_kwargs = instance.model.generate.call_args[1]
    assert "eos_token_id" not in call_kwargs
    # generation_config has no pad -> tokenizer pad is filled in
    assert call_kwargs["pad_token_id"] == 0
    assert "attention_mask" in call_kwargs


def test_generate_keeps_generation_config_pad(monkeypatch):
    model = _fake_model()
    model.generation_config = SimpleNamespace(pad_token_id=5, eos_token_id=[2, 107])
    instance = _instance(monkeypatch, model=model)
    instance.generate("Hello")
    call_kwargs = instance.model.generate.call_args[1]
    assert "pad_token_id" not in call_kwargs
    assert "eos_token_id" not in call_kwargs


def test_generate_clamps_max_length_and_truncates_left(monkeypatch):
    tok = _BosTokenizer()
    config = _smart_config()
    config.max_seq_length = 10
    instance = _instance(monkeypatch, tok=tok, config=config)
    instance.generate("Hello", max_new_tokens=50)
    call = tok.calls[-1]
    assert call["max_length"] == 1
    assert call["truncation"] is True
    assert call["truncation_side"] == "left"
    assert tok.truncation_side == "right"  # restored


def test_generate_uses_input_embedding_device_and_eval(monkeypatch):
    model = _fake_model()
    model.eval = Mock()
    model.get_input_embeddings = Mock(return_value=SimpleNamespace(weight=torch.zeros(1, device="cpu")))
    instance = _instance(monkeypatch, model=model)
    instance.generate("Hello")
    model.eval.assert_called_once()
    model.get_input_embeddings.assert_called()


def test_generate_trims_at_earliest_stop_string(monkeypatch):
    tok = _tokenizer()
    tok.decode = lambda ids, skip_special_tokens=False: "answer B-STOP more A-STOP tail"
    instance = _instance(monkeypatch, tok=tok)
    result = instance.generate("Hello", stop_strings=["A-STOP", "B-STOP"])
    assert result == "answer"
    call_kwargs = instance.model.generate.call_args[1]
    assert call_kwargs["stop_strings"] == ["A-STOP", "B-STOP"]
    assert call_kwargs["tokenizer"] is tok


class _StreamTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    VOCAB = {5: "Hello ", 6: "world ", 7: "ST", 8: "OP ", 9: "leak "}

    def __call__(self, text, return_tensors="pt", **kwargs):
        input_ids = torch.tensor([[1, 3]], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(self.VOCAB.get(int(t), "") for t in token_ids)


def test_streaming_does_not_leak_stop_string(monkeypatch):
    model = _fake_model()

    def streaming_generate(**kwargs):
        streamer = kwargs["streamer"]
        criteria = kwargs["stopping_criteria"]
        ids = kwargs["input_ids"]
        streamer.put(ids)  # prompt (skipped)
        for t in [5, 6, 7, 8, 9, 9]:
            ids = torch.cat([ids, torch.tensor([[t]])], dim=1)
            streamer.put(torch.tensor([t]))
            if bool(criteria(ids, None).all()):
                break
        streamer.end()

    model.generate = Mock(side_effect=streaming_generate)
    instance = _instance(monkeypatch, model=model, tok=_StreamTokenizer())
    chunks = list(instance.generate("Hi", stream=True, stop_strings=["STOP"]))
    text = "".join(chunks)
    assert text == "Hello world "


def test_streaming_propagates_thread_exception(monkeypatch):
    import pytest

    model = _fake_model()
    model.generate = Mock(side_effect=RuntimeError("CUDA exploded"))
    instance = _instance(monkeypatch, model=model, tok=_StreamTokenizer())
    gen = instance.generate("Hi", stream=True, stream_timeout=5)
    with pytest.raises(RuntimeError, match="CUDA exploded"):
        list(gen)


def test_enable_flash_attention_noop_without_setter():
    from quantllm.core.generation import GenerationMixin

    cfg = SimpleNamespace(_attn_implementation="eager")
    GenerationMixin._enable_flash_attention(SimpleNamespace(config=cfg), verbose=False)
    assert cfg._attn_implementation == "eager"

    model = SimpleNamespace(set_attn_implementation=Mock(side_effect=ValueError("nope")))
    GenerationMixin._enable_flash_attention(model, verbose=False)  # must not raise
    model.set_attn_implementation.assert_called_once()
