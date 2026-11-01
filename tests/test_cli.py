"""Tests for the expanded CLI."""

import pytest

from quantllm.cli import main


class TestCLICommands:
    def test_version_command(self):
        import sys
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        sys.argv = ["quantllm", "version"]
        sys.stdout = StringIO()

        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code is None or exc.value.code == 0
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout

    def test_info_command(self):
        import sys
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        sys.argv = ["quantllm", "info"]
        sys.stdout = StringIO()

        try:
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code is None or exc.value.code == 0
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout


def _run_cli(argv, monkeypatch, capsys=None):
    import sys

    monkeypatch.setattr(sys, "argv", ["quantllm", *argv])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


class TestCLIExitCodes:
    def test_version_matches_package(self, monkeypatch, capsys):
        import quantllm

        _run_cli(["version"], monkeypatch)
        assert quantllm.__version__ in capsys.readouterr().out

    def test_models_without_subcommand_exits_2(self, monkeypatch, capsys):
        assert _run_cli(["models"], monkeypatch) == 2

    def test_register_failure_exits_1(self, monkeypatch, capsys):
        import quantllm.registry as registry_mod

        monkeypatch.setattr(registry_mod, "register_model", lambda *a, **k: False)
        assert _run_cli(["register", "me/model"], monkeypatch) == 1

    def test_register_success_exits_0(self, monkeypatch, capsys):
        import sys

        import quantllm.registry as registry_mod

        monkeypatch.setattr(registry_mod, "register_model", lambda *a, **k: True)
        monkeypatch.setattr(sys, "argv", ["quantllm", "register", "me/model"])
        main()  # no SystemExit on success
        assert "registered" in capsys.readouterr().out


class TestCLIFlagWiring:
    def _patch_loader(self, monkeypatch):
        from unittest import mock

        import quantllm.core as core

        fake_model = mock.MagicMock()
        loader = mock.MagicMock(return_value=fake_model)
        monkeypatch.setattr(core.TurboModel, "from_pretrained", loader)
        return loader, fake_model

    @pytest.mark.parametrize(
        "argv",
        [
            ["convert", "org/m", "-o", "out.gguf"],
            ["export", "org/m", "-f", "gguf"],
            ["finetune", "org/m", "--dataset", "d"],
        ],
    )
    def test_trust_remote_code_defaults_false(self, argv, monkeypatch, capsys):
        import sys

        loader, _ = self._patch_loader(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["quantllm", *argv])
        main()
        assert loader.call_args.kwargs["trust_remote_code"] is False

    @pytest.mark.parametrize(
        "argv",
        [
            ["convert", "org/m", "-o", "out.gguf", "--trust-remote-code"],
            ["export", "org/m", "--trust-remote-code"],
            ["finetune", "org/m", "--dataset", "d", "--trust-remote-code"],
        ],
    )
    def test_trust_remote_code_passed_through(self, argv, monkeypatch, capsys):
        import sys

        loader, _ = self._patch_loader(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["quantllm", *argv])
        main()
        assert loader.call_args.kwargs["trust_remote_code"] is True

    def test_convert_wires_output_and_quant(self, monkeypatch, capsys):
        import sys

        _, fake_model = self._patch_loader(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["quantllm", "convert", "org/m", "-o", "x.gguf", "-q", "q5_k_m"])
        main()
        fake_model.export.assert_called_once_with("gguf", output_path="x.gguf", quantization="q5_k_m")

    @pytest.mark.parametrize("flag,expected", [([], False), (["--trust-remote-code"], True)])
    def test_serve_trust_remote_code(self, flag, expected, monkeypatch):
        import sys

        pytest.importorskip("fastapi")
        import quantllm.server.app as app_mod

        captured = {}
        monkeypatch.setattr(app_mod, "run_server", lambda cfg: captured.setdefault("cfg", cfg))
        monkeypatch.setattr(sys, "argv", ["quantllm", "serve", "org/m", "--port", "9999", *flag])
        main()
        assert captured["cfg"].trust_remote_code is expected
        assert captured["cfg"].port == 9999
