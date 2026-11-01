"""End-to-end GPU tests against a real (small) HuggingFace model.

These check behaviour, not just "it didn't crash": that quantization really
replaced layers, that greedy decoding is deterministic, that fine-tuning lowers
the loss, and that exported artifacts load back and produce the same output.
"""

import gc
import importlib.util
import shutil

import pytest
import torch


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


PROMPT = "The capital of France is"


def _bnb_layer_count(model, cls_name):
    return sum(1 for m in model.modules() if type(m).__name__ == cls_name)


# --------------------------------------------------------------------------- #
# Loading / quantization
# --------------------------------------------------------------------------- #


class TestLoading:
    def test_4bit_replaces_linear_layers(self, model_4bit):
        assert model_4bit.is_quantized
        assert _bnb_layer_count(model_4bit.model, "Linear4bit") > 0
        assert next(model_4bit.model.parameters()).device.type == "cuda"

    def test_4bit_uses_less_memory_than_16bit(self, model_4bit, model_16bit):
        q = model_4bit.model.get_memory_footprint()
        f = model_16bit.model.get_memory_footprint()
        assert q < 0.6 * f, f"4-bit footprint {q / 1e9:.2f}GB not much smaller than 16-bit {f / 1e9:.2f}GB"

    def test_8bit_loads(self, load_model):
        m = load_model(bits=8)
        assert m.is_quantized
        assert _bnb_layer_count(m.model, "Linear8bitLt") > 0
        assert isinstance(m.generate(PROMPT, max_new_tokens=8, do_sample=False), str)

    def test_16bit_dtype_matches_hardware(self, model_16bit):
        dtype = next(model_16bit.model.parameters()).dtype
        # is_bf16_supported() is True on pre-Ampere GPUs via emulation; native bf16 needs sm_80+.
        expected = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        assert dtype == expected

    def test_report_is_consistent(self, model_4bit):
        r = model_4bit.report()
        assert r["is_quantized"] is True
        assert r["effective_loading_bits"] == 4
        assert "cuda" in str(r["device"])


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


class TestGeneration:
    @pytest.mark.parametrize("fixture", ["model_4bit", "model_16bit"])
    def test_greedy_is_deterministic_and_excludes_prompt(self, fixture, request):
        m = request.getfixturevalue(fixture)
        a = m.generate(PROMPT, max_new_tokens=16, do_sample=False)
        b = m.generate(PROMPT, max_new_tokens=16, do_sample=False)
        assert a == b
        assert a.strip()
        assert not a.startswith(PROMPT), "generate() should return only the completion"

    def test_greedy_answer_is_sensible(self, model_16bit):
        out = model_16bit.generate(PROMPT, max_new_tokens=8, do_sample=False)
        assert "paris" in out.lower()

    def test_max_new_tokens_is_respected(self, model_4bit):
        out = model_4bit.generate("Write a long story about a dragon.", max_new_tokens=10, do_sample=False)
        n = len(model_4bit.tokenizer(out, add_special_tokens=False)["input_ids"])
        assert n <= 12  # small slack for re-tokenization

    def test_stream_matches_non_stream(self, model_16bit):
        full = model_16bit.generate(PROMPT, max_new_tokens=16, do_sample=False)
        streamed = "".join(model_16bit.generate(PROMPT, max_new_tokens=16, do_sample=False, stream=True))
        assert streamed.strip() == full.strip()

    def test_chat_uses_template(self, model_16bit):
        out = model_16bit.chat(
            [{"role": "user", "content": "Reply with exactly one word: what color is the sky?"}],
            max_new_tokens=10,
            do_sample=False,
        )
        assert out.strip()
        assert "<|im_start|>" not in out and "assistant" not in out.lower()[:12]

    def test_stop_strings(self, model_16bit):
        out = model_16bit.generate("1, 2, 3, 4,", max_new_tokens=30, do_sample=False, stop_strings=["7"])
        assert "8" not in out

    def test_batch_of_prompts_sequential_no_leak(self, model_4bit):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(5):
            model_4bit.generate(PROMPT, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() - before < 64 * 1024**2


# --------------------------------------------------------------------------- #
# HQQ native quantization on CUDA
# --------------------------------------------------------------------------- #


class TestHQQ:
    @pytest.mark.parametrize("bits,max_rel_err", [(8, 0.02), (4, 0.15), (2, 0.6)])
    def test_linear_reconstruction(self, bits, max_rel_err):
        from quantllm.quant.hqq import HQQLinear

        torch.manual_seed(0)
        w = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
        q = HQQLinear(1024, 512, bits=bits, group_size=64).cuda()
        q.quantize(w)
        rel = (q.dequantize().float() - w.float()).norm() / w.float().norm()
        assert rel < max_rel_err, f"{bits}-bit relative error {rel:.3f}"

        x = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
        y = q(x)
        assert y.shape == (4, 512) and y.dtype == x.dtype and torch.isfinite(y).all()

    def test_quantize_real_model_still_generates(self, load_model):
        from quantllm.quant.hqq import HQQQuantizer

        m = load_model(bits=16)
        m.model = HQQQuantizer().quantize_model(m.model, bits=8, group_size=64)
        out = m.generate(PROMPT, max_new_tokens=8, do_sample=False)
        assert "paris" in out.lower()


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #


TRAIN_DATA = [
    {"instruction": "What is the secret word?", "output": "The secret word is pineapple."},
] * 32


class TestFinetune:
    def test_qlora_lowers_loss(self, load_model, tmp_path):
        m = load_model(bits=4)
        result = m.finetune(
            TRAIN_DATA,
            epochs=3,
            learning_rate=2e-4,
            batch_size=4,
            lora_r=8,
            output_dir=str(tmp_path / "ft"),
            logging_steps=1,
        )
        assert result.get("final_loss") is not None, f"finetune failed: {result}"
        losses = [h["loss"] for h in result["log_history"] if "loss" in h]
        # Mean train_loss is noisy on fp16-only GPUs (early steps skipped by the loss scaler),
        # so check the trajectory and that the model actually learned the answer.
        assert result["final_loss"] < 0.5 * losses[0], f"loss did not fall: {losses}"
        out = m.chat([{"role": "user", "content": "What is the secret word?"}], max_new_tokens=12, do_sample=False)
        assert "pineapple" in out.lower(), out


# --------------------------------------------------------------------------- #
# Export round-trips
# --------------------------------------------------------------------------- #


class TestExport:
    def test_safetensors_roundtrip_16bit(self, model_16bit, tmp_path):
        from transformers import AutoModelForCausalLM

        from quantllm.utils.hf_compat import dtype_kwargs

        ref = model_16bit.generate(PROMPT, max_new_tokens=8, do_sample=False)
        out_dir = model_16bit.export("safetensors", str(tmp_path / "st"))
        reloaded = AutoModelForCausalLM.from_pretrained(out_dir, **dtype_kwargs(torch.float16)).cuda()
        tok = model_16bit.tokenizer
        ids = tok(PROMPT, return_tensors="pt").to("cuda")
        gen = reloaded.generate(**ids, max_new_tokens=8, do_sample=False)
        text = tok.decode(gen[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
        assert text.strip().split()[:1] == ref.strip().split()[:1]
        del reloaded
        free_cuda()

    def test_safetensors_from_4bit_is_loadable(self, load_model, tmp_path):
        from transformers import AutoModelForCausalLM

        m = load_model(bits=4)
        out_dir = m.export("safetensors", str(tmp_path / "st4"))
        reloaded = AutoModelForCausalLM.from_pretrained(out_dir, device_map="cuda")
        assert reloaded is not None
        del reloaded
        free_cuda()

    @pytest.mark.slow
    @pytest.mark.skipif(importlib.util.find_spec("gguf") is None, reason="gguf not installed")
    def test_gguf_q4_k_m_loads_in_llama_cpp(self, load_model, tmp_path):
        llama_cpp = pytest.importorskip("llama_cpp")
        m = load_model(bits=16)
        path = m.export("gguf", str(tmp_path / "model.Q4_K_M.gguf"), quantization="Q4_K_M")
        llm = llama_cpp.Llama(model_path=str(path), n_ctx=256, verbose=False)
        out = llm(PROMPT, max_tokens=8, temperature=0)["choices"][0]["text"]
        assert out.strip()

    @pytest.mark.slow
    def test_onnx_export_produces_model(self, load_model, tmp_path):
        pytest.importorskip("optimum.onnxruntime", reason="optimum-onnx not installed (needs transformers<4.58)")
        m = load_model(bits=16)
        out = m.export("onnx", str(tmp_path / "onnx"))
        assert any(p.suffix == ".onnx" for p in __import__("pathlib").Path(out).rglob("*"))
        shutil.rmtree(out, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Serving and benchmarking
# --------------------------------------------------------------------------- #


class TestServing:
    def test_openai_chat_endpoint(self, model_id):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from quantllm.server.app import _build_app
        from quantllm.server.engine import InferenceEngine, ServerConfig

        engine = InferenceEngine(ServerConfig(model=model_id, bits=4, trust_remote_code=False))
        engine.load()
        with TestClient(_build_app(engine)) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "max_tokens": 8,
                    "temperature": 0,
                },
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["choices"][0]["message"]["content"].strip()
            assert body["usage"]["completion_tokens"] > 0
        free_cuda()


class TestBenchmark:
    def test_benchmark_reports_real_numbers(self, model_4bit):
        from quantllm.benchmark.runner import BenchmarkRunner

        r = BenchmarkRunner(verbose=False).benchmark_model(
            model_4bit.model, model_4bit.tokenizer, model_4bit.config, max_new_tokens=32, num_warmup=1, num_runs=2
        )
        assert r.decode_tokens_per_sec > 0
        assert r.peak_vram_gb > 0
