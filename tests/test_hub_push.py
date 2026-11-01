"""Tests for HF Hub push: repo privacy, base_model sanitization, staging dirs."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest

import quantllm.hub.hub_manager as hub_manager_mod
from quantllm.hub.hub_manager import QuantLLMHubManager
from quantllm.hub.model_card import ModelCardGenerator, sanitize_base_model


@pytest.fixture
def fake_api(monkeypatch):
    api = mock.MagicMock()
    api.repo_exists.return_value = False
    monkeypatch.setattr(hub_manager_mod, "HfApi", mock.MagicMock(return_value=api))
    return api


class TestPrivateForwarded:
    def test_push_private_true(self, fake_api):
        manager = QuantLLMHubManager("me/repo")
        manager.push(private=True)
        fake_api.create_repo.assert_called_once()
        kwargs = fake_api.create_repo.call_args.kwargs
        assert kwargs["private"] is True
        assert kwargs["exist_ok"] is True

    def test_push_uses_init_default(self, fake_api):
        manager = QuantLLMHubManager("me/repo", private=True)
        manager.push()
        assert fake_api.create_repo.call_args.kwargs["private"] is True

    def test_push_default_public(self, fake_api):
        QuantLLMHubManager("me/repo").push()
        assert fake_api.create_repo.call_args.kwargs["private"] is False


class TestStagingDir:
    def test_unique_per_manager_and_keeps_owner(self, fake_api):
        a = QuantLLMHubManager("alice/model")
        b = QuantLLMHubManager("bob/model")
        try:
            assert a.staging_dir != b.staging_dir
            assert "alice" in os.path.basename(a.staging_dir)
            assert "bob" in os.path.basename(b.staging_dir)
            assert os.path.isdir(a.staging_dir) and os.listdir(a.staging_dir) == []
        finally:
            a.cleanup()
            b.cleanup()

    def test_cleaned_after_successful_push(self, fake_api):
        manager = QuantLLMHubManager("me/repo")
        with open(os.path.join(manager.staging_dir, "stale.bin"), "wb") as f:
            f.write(b"x")
        manager.push()
        assert not os.path.exists(manager.staging_dir)

    def test_kept_after_failed_push(self, fake_api):
        fake_api.upload_folder.side_effect = RuntimeError("boom")
        manager = QuantLLMHubManager("me/repo")
        try:
            with pytest.raises(RuntimeError):
                manager.push()
            assert os.path.isdir(manager.staging_dir)
        finally:
            manager.cleanup()
        assert not os.path.exists(manager.staging_dir)


class TestBaseModelSanitization:
    @pytest.mark.parametrize("value", ["meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-0.5B", "org/model.v1_x"])
    def test_accepts_repo_ids(self, value):
        assert sanitize_base_model(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "/home/user/models/llama",
            "C:\\Users\\me\\models\\llama",
            "./checkpoints/run1/final",
            "model-without-owner",
            "",
            None,
        ],
    )
    def test_rejects_paths(self, value):
        assert sanitize_base_model(value) is None

    def test_existing_local_dir_rejected(self, tmp_path, monkeypatch):
        (tmp_path / "org" / "model").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        assert sanitize_base_model("org/model") is None

    def test_card_omits_local_path(self):
        card = ModelCardGenerator(
            repo_id="me/repo", base_model="/home/user/secret/llama-ft", format="safetensors"
        ).generate()
        header = card.split("---", 2)[1]
        assert "base_model:" not in header
        assert "/home/user" not in card
        assert "llama-ft" in card

    def test_card_keeps_hub_id(self):
        card = ModelCardGenerator(repo_id="me/repo", base_model="Qwen/Qwen2.5-0.5B", format="gguf").generate()
        assert "base_model: Qwen/Qwen2.5-0.5B" in card

    def test_card_uses_actual_gguf_filename(self):
        card = ModelCardGenerator(
            repo_id="me/other-name",
            base_model="Qwen/Qwen2.5-0.5B",
            format="gguf",
            quantization="Q4_K_M",
            gguf_filename="Qwen2.5-0.5B.Q4_K_M.gguf",
        ).generate()
        assert 'filename="Qwen2.5-0.5B.Q4_K_M.gguf"' in card
        assert "other-name.Q4_K_M.gguf" not in card


class TestPushToHubMixin:
    def _model(self, name_or_path):
        from quantllm.core.push import PushToHubMixin

        class Stub(PushToHubMixin):
            pass

        stub = Stub()
        stub.model = SimpleNamespace(config=SimpleNamespace(_name_or_path=name_or_path))
        stub.export_push_config = {}

        def fake_export(*, format, output_path, quantization=None, **kwargs):
            with open(output_path, "wb") as f:
                f.write(b"GGUF")
            return output_path

        stub.export = fake_export
        return stub

    def test_gguf_push_private_filename_and_sanitized_base(self, fake_api):
        uploaded = {}

        def capture(**kwargs):
            folder = kwargs["folder_path"]
            uploaded["files"] = sorted(os.listdir(folder))
            with open(os.path.join(folder, "README.md"), encoding="utf-8") as f:
                uploaded["readme"] = f.read()

        fake_api.upload_folder.side_effect = capture
        model = self._model(os.path.join(os.sep, "home", "user", "ckpt", "my-llama"))
        model.push_to_hub("me/published", format="gguf", quantization="Q4_K_M", private=True)

        assert fake_api.create_repo.call_args.kwargs["private"] is True
        assert "my-llama.Q4_K_M.gguf" in uploaded["files"]
        readme = uploaded["readme"]
        assert 'filename="my-llama.Q4_K_M.gguf"' in readme
        assert "base_model:" not in readme.split("---", 2)[1]
        assert "ckpt" not in readme
