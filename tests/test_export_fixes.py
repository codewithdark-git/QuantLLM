"""CPU-only regression tests for export / quantization fixes."""

import os
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from quantllm.core.export_router import ExportRouterMixin
from quantllm.kernels.cpu import CPUInt4Linear, CPUInt8Linear
from quantllm.kernels.triton import is_triton_available
from quantllm.kernels.triton.quantized_linear import _pytorch_dequant_matmul
from quantllm.quant import llama_cpp
from quantllm.quant.hqq import HQQLinear


def _outlier_matrix(out_features=256, in_features=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=g)
    mask = torch.rand(out_features, in_features, generator=g) < 0.01
    w[mask] *= 10.0
    return w


# --------------------------------------------------------------------------- HQQ


class TestHQQFixes:
    def test_non_divisible_in_features(self):
        layer = HQQLinear(576, 64, bits=4, group_size=128)
        w = torch.randn(64, 576)
        layer.quantize(w)
        deq = layer.dequantize(torch.float32)
        assert deq.shape == (64, 576)
        assert ((deq - w) ** 2).mean() < 0.05
        out = layer(torch.randn(3, 576))
        assert out.shape == (3, 64)

    def test_group_larger_than_in_features(self):
        layer = HQQLinear(48, 32, bits=4, group_size=128)
        w = torch.randn(32, 48)
        layer.quantize(w)
        assert layer.dequantize().shape == (32, 48)

    @pytest.mark.parametrize("bits", [2, 4])
    def test_optimize_lowers_reconstruction_error(self, bits):
        w = _outlier_matrix()
        errors = {}
        for optimize in (False, True):
            layer = HQQLinear(512, 256, bits=bits, group_size=64, optimize=optimize)
            layer.quantize(w)
            deq = layer.dequantize(torch.float32)
            errors[optimize] = ((deq - w) ** 2).mean().item()
        assert errors[True] < errors[False]

    def test_bf16_forward_dtype(self):
        layer = HQQLinear(64, 32, bits=4, group_size=32, bias=True)
        layer.quantize(torch.randn(32, 64))
        layer.bias = nn.Parameter(torch.randn(32, dtype=torch.float16))
        out = layer(torch.randn(2, 64, dtype=torch.bfloat16))
        assert out.dtype == torch.bfloat16


# --------------------------------------------------------------------------- CPU / Triton kernels


class TestKernelFixes:
    def test_cpu_int4_group_ge_in_features(self):
        layer = CPUInt4Linear(48, 32, bits=4, group_size=128)
        assert layer.qweight.shape == (32, 1, 64)
        w = torch.randn(32, 48)
        layer.quantize(w)
        deq = layer._dequantize_weights()
        assert deq.shape == (32, 48)
        assert ((deq - w) ** 2).mean() < 0.05

    def test_cpu_int4_non_divisible(self):
        layer = CPUInt4Linear(576, 16, bits=4, group_size=128)
        layer.quantize(torch.randn(16, 576))
        assert layer._dequantize_weights().shape == (16, 576)

    def test_cpu_int4_bf16_forward(self):
        layer = CPUInt4Linear(64, 32, bits=4, group_size=32, bias=True)
        layer.quantize(torch.randn(32, 64))
        layer.bias = nn.Parameter(torch.randn(32))
        out = layer(torch.randn(2, 64, dtype=torch.bfloat16))
        assert out.dtype == torch.bfloat16

    def test_cpu_int8_bf16_forward(self):
        layer = CPUInt8Linear(64, 32, bias=True)
        layer.quantize(torch.randn(32, 64))
        layer.bias = nn.Parameter(torch.randn(32))
        out = layer(torch.randn(2, 64, dtype=torch.bfloat16))
        assert out.dtype == torch.bfloat16

    def test_pytorch_fallback_casts_to_input_dtype(self):
        qweight = torch.randint(-8, 7, (64, 16), dtype=torch.int8)
        scales = torch.rand(2, 16)
        zeros = torch.zeros(2, 16)
        bias = torch.randn(16)
        out = _pytorch_dequant_matmul(torch.randn(3, 64, dtype=torch.bfloat16), qweight, scales, zeros, bias, 32)
        assert out.dtype == torch.bfloat16

    def test_triton_requires_cuda(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            assert is_triton_available() is False


# --------------------------------------------------------------------------- Export routing


class TestMLXQuantParsing:
    @pytest.mark.parametrize("name", [None, "", "f16", "bf16", "fp16", "f32", "none", "BF16"])
    def test_full_precision_means_no_quantization(self, name):
        assert ExportRouterMixin._mlx_quant_bits(name) is None

    @pytest.mark.parametrize(
        "name,bits",
        [("q4_k_m", 4), ("4bit", 4), ("q8_0", 8), ("8bit", 8), ("q2_k", 4), ("q3_k_m", 4), ("q5_k_m", 8), ("q6_k", 8)],
    )
    def test_bit_mapping(self, name, bits):
        assert ExportRouterMixin._mlx_quant_bits(name) == bits


class TestGGUFOuttype:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("f16", "f16"),
            ("F16", "f16"),
            ("float16", "f16"),
            ("bf16", "bf16"),
            ("bfloat16", "bf16"),
            ("torch.bfloat16", "bf16"),
            ("f32", "f32"),
            ("float32", "f32"),
            ("q4_k_m", None),
            ("Q8_0", None),
        ],
    )
    def test_mapping(self, name, expected):
        assert llama_cpp.gguf_outtype(name) == expected


def _make_converter_env(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("")
    return model_dir, converter


def _fake_popen(lines, returncode=0, captured=None):
    def factory(cmd, **kwargs):
        if captured is not None:
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
        proc = mock.MagicMock()
        text = "".join(lines)
        proc.stdout = mock.MagicMock()
        proc.stdout.readline.side_effect = list(lines) + [""]
        proc.communicate.return_value = (text, None)
        proc.returncode = returncode
        proc.wait.return_value = returncode
        return proc

    return factory


class TestConvertToGGUF:
    @pytest.mark.parametrize("print_output", [False, True])
    def test_merged_stream_and_output_dir(self, tmp_path, print_output):
        model_dir, converter = _make_converter_env(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        captured = {}
        with mock.patch.object(subprocess, "Popen", side_effect=_fake_popen(["INFO: Loading model\n"], 0, captured)):
            files, _ = llama_cpp.convert_to_gguf(
                model_name="m",
                input_folder=str(model_dir),
                model_dtype="bfloat16",
                quantization_type="bf16",
                converter_location=str(converter),
                print_output=print_output,
                output_dir=str(out_dir),
            )
        assert captured["kwargs"]["stderr"] is subprocess.STDOUT
        cmd = captured["cmd"]
        assert cmd[cmd.index("--outtype") + 1] == "bf16"
        outfile = cmd[cmd.index("--outfile") + 1]
        assert os.path.isabs(outfile)
        assert os.path.dirname(outfile) == os.path.abspath(str(out_dir))
        assert files == [outfile]

    def test_failure_reports_merged_output(self, tmp_path):
        model_dir, converter = _make_converter_env(tmp_path)
        with mock.patch.object(subprocess, "Popen", side_effect=_fake_popen(["Traceback: boom from stderr\n"], 1)):
            with pytest.raises(RuntimeError, match="boom from stderr"):
                llama_cpp.convert_to_gguf(
                    model_name="m",
                    input_folder=str(model_dir),
                    model_dtype="f16",
                    quantization_type="f16",
                    converter_location=str(converter),
                    print_output=False,
                )


class TestQuantizeGGUFImatrix:
    @pytest.mark.parametrize("qtype", ["iq3_xxs", "iq2_xxs", "iq1_s"])
    def test_requires_imatrix(self, tmp_path, qtype):
        with pytest.raises(ValueError, match="importance matrix"):
            llama_cpp.quantize_gguf(str(tmp_path / "in.gguf"), str(tmp_path / "out.gguf"), qtype)

    def test_imatrix_flag_passed(self, tmp_path):
        inp = tmp_path / "in.gguf"
        inp.write_bytes(b"x" * 16)
        imatrix = tmp_path / "imatrix.dat"
        imatrix.write_bytes(b"x")
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            proc = mock.MagicMock()
            proc.stdout = iter([])
            proc.returncode = 0
            return proc

        with mock.patch.object(subprocess, "Popen", side_effect=fake_popen):
            llama_cpp.quantize_gguf(
                str(inp),
                str(tmp_path / "out.gguf"),
                "iq2_xxs",
                quantizer_location="llama-quantize",
                print_output=False,
                imatrix=str(imatrix),
            )
        cmd = captured["cmd"]
        assert cmd[1:3] == ["--imatrix", str(imatrix)]
        assert "IQ2_XXS" in cmd


class TestWindowsBinaryResolution:
    def _setup(self, tmp_path, rel_parts):
        llama_dir = tmp_path / "llama.cpp"
        exe = llama_dir.joinpath(*rel_parts)
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("")
        (llama_dir / "convert_hf_to_gguf.py").write_text("")
        return llama_dir, exe

    def test_exe_name(self):
        with mock.patch.object(llama_cpp.os, "name", "nt"):
            assert llama_cpp._exe_name("llama-quantize") == "llama-quantize.exe"
        with mock.patch.object(llama_cpp.os, "name", "posix"):
            assert llama_cpp._exe_name("llama-quantize") == "llama-quantize"

    @pytest.mark.parametrize(
        "rel_parts",
        [
            ("build", "bin", "Release", "llama-quantize.exe"),
            ("build", "bin", "llama-quantize.exe"),
            ("llama-quantize.exe",),
        ],
    )
    def test_finds_msvc_and_root_exe(self, tmp_path, rel_parts):
        llama_dir, exe = self._setup(tmp_path, rel_parts)
        with (
            mock.patch.object(llama_cpp, "get_llama_cpp_dir", return_value=str(llama_dir)),
            mock.patch.object(llama_cpp.os, "name", "nt"),
            mock.patch.object(llama_cpp.os, "access", return_value=True),
        ):
            quantizer, converter = llama_cpp.check_llama_cpp()
        assert os.path.normpath(quantizer) == os.path.normpath(str(exe))
        assert converter.endswith("convert_hf_to_gguf.py")

    def test_cmake_failure_shows_output_tail(self, tmp_path):
        result = SimpleNamespace(returncode=2, stdout="line1\nCMake Error: no compiler found\n")
        with mock.patch.object(subprocess, "run", return_value=result):
            with pytest.raises(RuntimeError, match="no compiler found"):
                llama_cpp._run_build_step(["cmake", "."], print_output=False, step="CMake configure")


class TestStaging:
    def test_stage_in_memory_model_dequantizes_bnb(self, tmp_path):
        saved = {}

        class FakeModel:
            def save_pretrained(self, path, **kwargs):
                saved["path"] = path
                open(os.path.join(path, "config.json"), "w").close()

        dequantized = FakeModel()
        router = ExportRouterMixin.__new__(ExportRouterMixin)
        router.model = object()
        router.tokenizer = None
        router._is_bnb_quantized = lambda: True  # type: ignore[assignment]
        router._dequantize_model = lambda: dequantized  # type: ignore[assignment]

        staging = router._stage_in_memory_model()
        try:
            assert saved["path"] == staging
            assert os.path.exists(os.path.join(staging, "config.json"))
        finally:
            import shutil

            shutil.rmtree(staging, ignore_errors=True)

    def test_has_in_memory_changes(self):
        router = ExportRouterMixin.__new__(ExportRouterMixin)
        assert router._has_in_memory_changes() is False
        router._lora_merged = True  # type: ignore[attr-defined]
        assert router._has_in_memory_changes() is True
