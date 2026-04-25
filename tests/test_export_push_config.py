from types import SimpleNamespace

from quantllm.core.turbo_model import TurboModel


def _stub_model(name: str = "org/test-model"):
    return SimpleNamespace(config=SimpleNamespace(_name_or_path=name))


def _stub_turbo(export_push_config):
    model = TurboModel.__new__(TurboModel)
    model.model = _stub_model()
    model.tokenizer = None
    smart_config = SimpleNamespace(quant_type="Q8_0")
    model.config = smart_config
    model._lora_applied = False
    model.verbose = False
    model.export_push_config = export_push_config
    return model


def test_build_export_push_config_uses_deterministic_defaults():
    resolved = TurboModel._build_export_push_config(None)
    assert resolved["format"] == "safetensors"
    assert resolved["push_format"] == "safetensors"
    assert resolved["quantization"] == "Q4_K_M"
    assert resolved["push_quantization"] is None


def test_build_export_push_config_aligns_push_values_with_export_values():
    resolved = TurboModel._build_export_push_config(
        {"format": "gguf", "quantization": "Q5_K_M"}
    )
    assert resolved["format"] == "gguf"
    assert resolved["push_format"] == "gguf"
    assert resolved["quantization"] == "Q5_K_M"
    assert resolved["push_quantization"] == "Q5_K_M"


def test_build_export_push_config_allows_nullable_push_quantization_override():
    resolved = TurboModel._build_export_push_config(
        {"format": "gguf", "quantization": "Q5_K_M", "push_quantization": None}
    )
    assert resolved["quantization"] == "Q5_K_M"
    assert resolved["push_quantization"] is None


def test_export_prefers_shared_quantization_over_smart_config_quant_type():
    model = _stub_turbo(
        {
            "format": "gguf",
            "push_format": "gguf",
            "quantization": "Q4_K_M",
            "push_quantization": "Q4_K_M",
        }
    )

    captured = {}

    def fake_export_gguf(output_path, quantization=None, **kwargs):
        captured["output_path"] = output_path
        captured["quantization"] = quantization
        return output_path

    model._export_gguf = fake_export_gguf
    model._export_safetensors = lambda *args, **kwargs: ""
    model._export_onnx = lambda *args, **kwargs: ""
    model._export_mlx = lambda *args, **kwargs: ""

    output = model.export()

    assert model.config.quant_type == "Q8_0"
    assert output.endswith(".Q4_K_M.gguf")
    assert captured["quantization"] == "Q4_K_M"


def test_gguf_push_uses_shared_config_when_omitted(monkeypatch, tmp_path):
    model = _stub_turbo({
        "format": "gguf",
        "push_format": "gguf",
        "quantization": "Q4_K_M",
        "push_quantization": "Q4_K_M",
    })

    calls = {}

    def fake_export(*, format, output_path, quantization=None, **kwargs):
        calls["export"] = {
            "format": format,
            "output_path": output_path,
            "quantization": quantization,
        }
        return output_path

    model.export = fake_export

    class FakeManager:
        def __init__(self, repo_id, hf_token=None):
            self.staging_dir = str(tmp_path / "quantllm-test-staging")

        def track_hyperparameters(self, params):
            calls["tracked"] = params

        def _generate_model_card(self, format):
            calls["card_format"] = format

        def push(self, commit_message):
            calls["pushed"] = commit_message

        def save_final_model(self, *args, **kwargs):
            raise AssertionError(
                "save_final_model should not be called for GGUF push"
            )

    import quantllm.hub as hub_module

    monkeypatch.setattr(hub_module, "QuantLLMHubManager", FakeManager)

    model.push("user/repo")

    assert calls["export"]["format"] == "gguf"
    assert calls["export"]["quantization"] == "Q4_K_M"
    assert calls["tracked"]["quantization"] == "Q4_K_M"


def test_onnx_push_does_not_force_quantization(monkeypatch, tmp_path):
    model = _stub_turbo(
        TurboModel._build_export_push_config({"push_format": "onnx"})
    )

    calls = {}

    class FakeManager:
        def __init__(self, repo_id, hf_token=None):
            self.staging_dir = str(tmp_path / "quantllm-test-staging")

        def track_hyperparameters(self, params):
            calls["tracked"] = params

        def _generate_model_card(self, format):
            calls["card_format"] = format

        def push(self, commit_message):
            calls["pushed"] = commit_message

        def save_final_model(self, *args, **kwargs):
            raise AssertionError(
                "save_final_model should not be called for ONNX push"
            )

    def fake_export_onnx(output_path, quantization=None, **kwargs):
        calls["onnx_quantization"] = quantization
        return output_path

    model._export_onnx = fake_export_onnx

    import quantllm.hub as hub_module

    monkeypatch.setattr(hub_module, "QuantLLMHubManager", FakeManager)

    model.push("user/repo")

    assert calls["onnx_quantization"] is None
    assert calls["tracked"]["quantization"] is None
