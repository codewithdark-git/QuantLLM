"""CPU-only regression tests for model loading / configuration fixes."""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

import quantllm.core.model_loader as model_loader_module
import quantllm.core.smart_config as smart_config_module
import quantllm.core.turbo_model as turbo_model_module
from quantllm.core.flash_attention import _pytorch_attention
from quantllm.core.hardware import GPUInfo, HardwareProfile, HardwareProfiler
from quantllm.core.memory import DynamicOffloader
from quantllm.core.model_analyzer import ModelAnalyzer, ModelInfo
from quantllm.core.smart_config import SmartConfig
from quantllm.core.turbo_model import TurboModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _gpu(index, total_gb, cc, free_gb=None):
    return GPUInfo(
        index=index,
        name=f"gpu{index}",
        memory_total_gb=total_gb,
        memory_free_gb=total_gb if free_gb is None else free_gb,
        compute_capability=cc,
        supports_bf16=cc >= (8, 0),
        supports_flash_attention=cc >= (8, 0),
    )


def _profile(gpus, best_gpu_index=0):
    best = next((g for g in gpus if g.index == best_gpu_index), None)
    return HardwareProfile(
        cpu_count=4,
        cpu_name="cpu",
        system_memory_gb=32.0,
        num_gpus=len(gpus),
        gpus=gpus,
        cuda_available=bool(gpus),
        cuda_version="12.1" if gpus else None,
        supports_bf16=any(g.supports_bf16 for g in gpus),  # deliberately "any" to catch misuse
        supports_flash_attention=any(g.supports_flash_attention for g in gpus),
        best_gpu_index=best.index if best else 0,
        total_gpu_memory_gb=sum(g.memory_total_gb for g in gpus),
    )


def _small_model_info():
    return ModelInfo(
        model_name="org/tiny-0.5b",
        num_params=500_000_000,
        size_gb=500_000_000 * 4 / (1024**3),
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        num_kv_heads=16,
        intermediate_size=4096,
        vocab_size=32000,
        max_position_embeddings=4096,
        model_type="llama",
    )


def _patch_detect(monkeypatch, profile, info=None):
    monkeypatch.setattr(HardwareProfiler, "detect", classmethod(lambda cls, force_refresh=False: profile))
    monkeypatch.setattr(
        ModelAnalyzer,
        "analyze",
        classmethod(lambda cls, *a, **k: info or _small_model_info()),
    )


# ---------------------------------------------------------------------------
# model_analyzer
# ---------------------------------------------------------------------------


def test_moe_param_estimate_counts_experts():
    mixtral = SimpleNamespace(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=32000,
        max_position_embeddings=32768,
        model_type="mixtral",
        num_local_experts=8,
    )
    info = ModelAnalyzer._extract_info_from_config("mistralai/Mixtral-8x7B-v0.1", mixtral)
    assert 45e9 < info.num_params < 48e9  # Mixtral 8x7B is ~46.7B total

    dense = SimpleNamespace(**{k: v for k, v in vars(mixtral).items() if k != "num_local_experts"})
    dense_info = ModelAnalyzer._extract_info_from_config("dense", dense)
    assert dense_info.num_params < 8e9


def test_moe_param_estimate_uses_moe_intermediate_and_shared_experts():
    base = dict(
        hidden_size=2048,
        num_hidden_layers=10,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=10944,
        vocab_size=100000,
        model_type="deepseek_v2",
        n_routed_experts=64,
        moe_intermediate_size=1408,
    )
    no_shared = ModelAnalyzer._extract_info_from_config("a", SimpleNamespace(**base))
    shared = ModelAnalyzer._extract_info_from_config("b", SimpleNamespace(**base, n_shared_experts=2))
    expected_shared = 2 * 3 * 2048 * 1408 * 10
    assert shared.num_params - no_shared.num_params == expected_shared
    # routed experts use moe_intermediate_size, not intermediate_size
    assert no_shared.num_params < 10 * 64 * 3 * 2048 * 10944


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Qwen/Qwen2.5-0.5B-Instruct", 500_000_000),
        ("org/model-1.5b", 1_500_000_000),
        ("org/model-13B-chat", 13_000_000_000),
        ("org/unknown", 7_000_000_000),
    ],
)
def test_estimate_from_name_handles_decimals(name, expected):
    assert ModelAnalyzer._estimate_from_name(name).num_params == expected


def test_analyze_defaults_to_no_remote_code(monkeypatch):
    assert inspect.signature(ModelAnalyzer.analyze).parameters["trust_remote_code"].default is False

    import transformers

    captured = {}

    def _fake_from_pretrained(name, **kwargs):
        captured.update(kwargs)
        raise OSError("offline")

    monkeypatch.setattr(ModelAnalyzer, "_cache", {})
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _fake_from_pretrained)
    ModelAnalyzer.analyze("org/some-model-3b", token="tok")
    assert captured["trust_remote_code"] is False
    assert captured["token"] == "tok"


# ---------------------------------------------------------------------------
# smart_config / hardware
# ---------------------------------------------------------------------------


def test_detect_forwards_trust_remote_code_and_token(monkeypatch):
    calls = {}

    def _analyze(cls, name, **kwargs):
        calls.update(kwargs)
        return _small_model_info()

    monkeypatch.setattr(HardwareProfiler, "detect", classmethod(lambda cls, force_refresh=False: _profile([])))
    monkeypatch.setattr(ModelAnalyzer, "analyze", classmethod(_analyze))
    SmartConfig.detect("org/x", trust_remote_code=True, token="abc")
    assert calls == {"trust_remote_code": True, "token": "abc"}


def test_no_gpu_defaults_to_full_precision(monkeypatch):
    _patch_detect(monkeypatch, _profile([]))
    cfg = SmartConfig.detect("org/x")
    assert cfg.bits == 16
    assert cfg.effective_loading_bits == 16
    assert cfg.device.type == "cpu"
    assert cfg.dtype == torch.float32
    assert cfg.use_flash_attention is False


def test_bf16_comes_from_selected_gpu(monkeypatch):
    # GPU 0: biggest but pre-Ampere; GPU 1: Ampere. any() would wrongly pick bf16.
    gpus = [_gpu(0, 24.0, (7, 5)), _gpu(1, 16.0, (8, 0))]
    _patch_detect(monkeypatch, _profile(gpus, best_gpu_index=0))

    cfg = SmartConfig.detect("org/x")
    assert cfg.device == torch.device("cuda:0")
    assert cfg.dtype == torch.float16
    assert cfg.use_flash_attention is False

    cfg1 = SmartConfig.detect("org/x", device="cuda:1")
    assert cfg1.dtype == torch.bfloat16
    assert cfg1.use_flash_attention is True


def test_compile_model_not_auto_enabled(monkeypatch):
    _patch_detect(monkeypatch, _profile([_gpu(0, 80.0, (9, 0))]))
    assert SmartConfig.detect("org/x").compile_model is False


def test_size_estimates_use_effective_loading_bits(monkeypatch):
    _patch_detect(monkeypatch, _profile([_gpu(0, 80.0, (9, 0))]))
    cfg = SmartConfig.detect("org/x", bits=3)
    info = _small_model_info()
    assert cfg.stats["quant_size"] == pytest.approx(info.estimated_size_at_bits(4))


def test_best_gpu_index_is_cuda_ordinal(monkeypatch):
    # GPU 0 fails to report info, so the list has only the device with ordinal 1.
    monkeypatch.setattr(HardwareProfiler, "_cached_profile", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        HardwareProfiler,
        "_get_gpu_info",
        staticmethod(lambda i: None if i == 0 else _gpu(1, 16.0, (8, 6))),
    )
    try:
        profile = HardwareProfiler.detect(force_refresh=True)
        assert profile.best_gpu_index == 1
        assert profile.best_gpu.index == 1
        assert profile.device == torch.device("cuda:1")
        assert profile.supports_bf16 is True
    finally:
        HardwareProfiler._cached_profile = None


def test_gpu_info_uses_mem_get_info_without_set_device(monkeypatch):
    props = SimpleNamespace(total_memory=16 * 1024**3, major=8, minor=0, name="fake", multi_processor_count=10)
    set_device = Mock()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (4 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    info = HardwareProfiler._get_gpu_info(0)
    assert info.memory_free_gb == pytest.approx(4.0)
    set_device.assert_not_called()


# ---------------------------------------------------------------------------
# turbo() / model_loader
# ---------------------------------------------------------------------------


def test_turbo_pops_trust_remote_code(monkeypatch):
    captured = []
    monkeypatch.setattr(
        TurboModel,
        "from_pretrained",
        classmethod(lambda cls, model, **kwargs: captured.append(kwargs)),
    )
    turbo_model_module.turbo("org/x")
    turbo_model_module.turbo("org/x", trust_remote_code=True)  # must not raise TypeError
    assert captured[0]["trust_remote_code"] is False
    assert captured[1]["trust_remote_code"] is True


def _call_fallback(monkeypatch, fake_auto, registered=None):
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {"llama": registered} if registered else {})
    monkeypatch.setattr(model_loader_module, "AutoModelForCausalLM", fake_auto)
    return TurboModel._load_model_with_fallback(
        "org/llama-7b",
        {"torch_dtype": torch.float32},
        trust_remote_code=False,
        hf_config=SimpleNamespace(model_type="llama"),
        model_type_override=None,
        base_model_fallback=True,
        from_config_only=False,
    )


@pytest.mark.parametrize(
    "error",
    [
        OSError("org/llama-7b is not a valid model identifier (401 gated)"),
        ImportError("needs package foo"),
        RuntimeError("something unrelated"),
    ],
)
def test_non_architecture_errors_are_reraised(monkeypatch, error):
    registered = Mock()

    class _FakeAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise error

    with pytest.raises(type(error)) as excinfo:
        _call_fallback(monkeypatch, _FakeAuto, registered=registered)
    assert excinfo.value is error
    registered.from_pretrained.assert_not_called()


def test_architecture_error_message_includes_primary(monkeypatch):
    primary = ValueError("Unrecognized configuration class FooConfig")

    class _FakeAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            if "config" in kwargs:
                raise ValueError("still bad")
            raise primary

    with pytest.raises(RuntimeError) as excinfo:
        _call_fallback(monkeypatch, _FakeAuto)
    assert "Unrecognized configuration class FooConfig" in str(excinfo.value)
    assert excinfo.value.__cause__ is primary


def test_auto_attn_implementation_retries_without_it(monkeypatch):
    calls = []

    class _FakeAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(dict(kwargs))
            if "attn_implementation" in kwargs:
                raise ValueError(
                    "Model does not support an attention implementation through scaled_dot_product_attention"
                )
            return "model"

    monkeypatch.setattr(model_loader_module, "AutoModelForCausalLM", _FakeAuto)
    result = TurboModel._load_model_with_fallback(
        "org/x",
        {"attn_implementation": "sdpa"},
        trust_remote_code=False,
        hf_config=None,
        model_type_override=None,
        base_model_fallback=True,
        from_config_only=False,
        auto_attn_implementation=True,
    )
    assert result == "model"
    assert "attn_implementation" not in calls[-1]


def test_choose_attn_implementation_cpu_is_sdpa():
    cfg = SimpleNamespace(dtype=torch.float32, device=torch.device("cpu"), use_flash_attention=True)
    assert TurboModel._choose_attn_implementation(cfg) == "sdpa"
    cfg = SimpleNamespace(dtype="float16", device="cpu", use_flash_attention=False)
    assert TurboModel._choose_attn_implementation(cfg) == "sdpa"


def test_quantization_kwargs_empty_without_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    cfg = SimpleNamespace(bits=4, dtype=torch.float16, cpu_offload=False)
    assert TurboModel._get_quantization_kwargs(cfg) == {}


def test_quantization_kwargs_empty_without_bitsandbytes(monkeypatch):
    real_find_spec = model_loader_module.importlib.util.find_spec
    monkeypatch.setattr(
        model_loader_module.importlib.util,
        "find_spec",
        lambda name, *a: None if name == "bitsandbytes" else real_find_spec(name, *a),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cfg = SimpleNamespace(bits=4, dtype=torch.float16, cpu_offload=False)
    assert TurboModel._get_quantization_kwargs(cfg) == {}


def test_from_pretrained_passes_attn_implementation_and_skips_compile(monkeypatch):
    monkeypatch.setattr(TurboModel, "_architecture_registry", {})
    monkeypatch.setattr(TurboModel, "_model_class_registry", {})
    cfg = SimpleNamespace(
        bits=16,
        effective_loading_bits=16,
        dtype=torch.float32,
        cpu_offload=False,
        device=torch.device("cpu"),
        gradient_checkpointing=False,
        use_flash_attention=False,
        compile_model=False,
        print_summary=lambda: None,
    )
    monkeypatch.setattr(model_loader_module.SmartConfig, "detect", lambda *a, **k: cfg)
    monkeypatch.setattr(
        model_loader_module.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(pad_token="x", eos_token="x", eos_token_id=0),
    )
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(model_type="llama", quantization_config=None),
    )
    seen = {}

    class _FakeAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            seen.update(kwargs)
            return nn.Linear(2, 2)

    monkeypatch.setattr(model_loader_module, "AutoModelForCausalLM", _FakeAuto)
    loaded = TurboModel.from_pretrained("org/llama-1b", quantize=False, verbose=False)
    assert seen["attn_implementation"] == "sdpa"
    assert isinstance(loaded.model, nn.Linear)  # not wrapped by torch.compile

    TurboModel.from_pretrained("org/llama-1b", quantize=False, verbose=False, attn_implementation="eager")
    assert seen["attn_implementation"] == "eager"


# ---------------------------------------------------------------------------
# turbo_model: dequantize / optimize_inference
# ---------------------------------------------------------------------------


def _bare_turbo(model, finetuned=False, lora=False):
    tm = TurboModel.__new__(TurboModel)
    tm.model = model
    tm.tokenizer = None
    tm.config = SimpleNamespace(dtype=torch.float16, bits=4)
    tm._is_quantized_override = None
    tm._is_finetuned = finetuned
    tm._lora_applied = lora
    tm.verbose = False
    return tm


def test_dequantize_merges_lora_and_uses_in_memory_model(monkeypatch):
    inner = nn.Linear(2, 2)
    inner.config = SimpleNamespace(_name_or_path="org/base")
    inner.dequantize = Mock(return_value=inner)

    peft_like = SimpleNamespace(merge_and_unload=Mock(return_value=inner), config=inner.config)
    reload = Mock()
    monkeypatch.setattr(turbo_model_module.AutoModelForCausalLM, "from_pretrained", reload)

    tm = _bare_turbo(peft_like, finetuned=True, lora=True)
    result = tm._dequantize_model()

    peft_like.merge_and_unload.assert_called_once()
    inner.dequantize.assert_called_once()
    reload.assert_not_called()
    assert result is inner
    assert result.weight.dtype == torch.float16
    assert tm._lora_applied is False


def test_dequantize_never_reloads_hub_after_finetuning(monkeypatch):
    model = nn.Linear(2, 2)
    model.config = SimpleNamespace(_name_or_path="org/base")
    model.dequantize = Mock(side_effect=ValueError("no quantizer"))
    monkeypatch.setattr(
        TurboModel, "_manual_bnb_dequantize", staticmethod(Mock(side_effect=RuntimeError("manual failed")))
    )
    reload = Mock()
    monkeypatch.setattr(turbo_model_module.AutoModelForCausalLM, "from_pretrained", reload)

    tm = _bare_turbo(model, finetuned=True)
    with pytest.raises(RuntimeError, match="Fine-tuned weights"):
        tm._dequantize_model()
    reload.assert_not_called()


def test_optimize_inference_noop_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tm = _bare_turbo(nn.Linear(2, 2))
    replace = Mock()
    monkeypatch.setattr(TurboModel, "_replace_with_triton", replace)
    tm.optimize_inference()
    replace.assert_not_called()


# ---------------------------------------------------------------------------
# flash_attention fallback
# ---------------------------------------------------------------------------


def _reference_attention(q, k, v, causal):
    # q/k/v: [b, s, h, d] with equal head counts
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    if causal:
        sq, sk = q.shape[2], k.shape[2]
        mask = torch.ones(sq, sk, dtype=torch.bool).tril(diagonal=sk - sq)
        scores = scores.masked_fill(~mask, float("-inf"))
    return (scores.softmax(-1) @ v).transpose(1, 2)


def test_pytorch_attention_gqa_and_bottom_right_causal():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 5, 2, 8)
    v = torch.randn(2, 5, 2, 8)
    out = _pytorch_attention(q, k, v, causal=True)
    expected = _reference_attention(q, k.repeat_interleave(2, dim=2), v.repeat_interleave(2, dim=2), causal=True)
    assert out.shape == (2, 3, 4, 8)
    assert torch.allclose(out, expected, atol=1e-5)

    # single decode token attends to the whole cache
    out1 = _pytorch_attention(q[:, :1], k, v, causal=True)
    expected1 = _reference_attention(
        q[:, :1], k.repeat_interleave(2, dim=2), v.repeat_interleave(2, dim=2), causal=False
    )
    assert torch.allclose(out1, expected1, atol=1e-5)


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


class _TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.lm_head = nn.Linear(4, 4)


def test_dynamic_offloader_never_selects_container():
    offloader = DynamicOffloader(_TinyDecoder(), max_gpu_layers=2, device="cpu")
    assert offloader.offloadable_layers == ["model.layers.0", "model.layers.1", "model.layers.2"]
    offloader.move_layer_to_gpu("model.layers.2")  # must not hardcode "cuda"
    assert offloader._layer_locations["model.layers.2"] == "gpu"
