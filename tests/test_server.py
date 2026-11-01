"""Tests for the QuantLLM inference server."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestServerConfig:
    def test_default_config(self):
        from quantllm.server import ServerConfig

        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080
        assert cfg.max_seq_length == 4096

    def test_from_cli(self):
        from quantllm.server import ServerConfig

        cfg = ServerConfig.from_cli(model="test-model", port=9090)
        assert cfg.model == "test-model"
        assert cfg.port == 9090

    def test_from_cli_env(self, monkeypatch):
        from quantllm.server import ServerConfig

        monkeypatch.setenv("QUANTLLM_MODEL", "env-model")
        cfg = ServerConfig.from_cli()
        assert cfg.model == "env-model"


class TestInferenceEngine:
    def test_init(self):
        from quantllm.server import InferenceEngine, ServerConfig

        cfg = ServerConfig(model="test-model")
        engine = InferenceEngine(cfg)
        assert engine.config.model == "test-model"
        assert engine.backend == "transformers"

    def test_detect_gguf_backend(self, monkeypatch):
        from quantllm.server import InferenceEngine, ServerConfig

        cfg = ServerConfig(model="/path/to/model.gguf")
        engine = InferenceEngine(cfg)

        def mock_import(name, *args, **kwargs):
            if name == "llama_cpp":
                raise ImportError
            raise ImportError

        monkeypatch.setattr("builtins.__import__", mock_import)
        assert not engine._detect_gguf()

    def test_load_transformers_mocked(self, monkeypatch):
        from quantllm.server import InferenceEngine, ServerConfig

        cfg = ServerConfig(model="test-model")
        engine = InferenceEngine(cfg)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model = MagicMock()

        def mock_from_pretrained(*args, **kwargs):
            if "tokenizer" in str(kwargs):
                return mock_tokenizer
            return mock_model

        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", mock_from_pretrained)
        monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", mock_from_pretrained)
        engine._load_transformers()
        assert engine.tokenizer is not None

    def test_build_chat_prompt(self):
        from quantllm.server.engine import _build_chat_prompt
        from quantllm.server import InferenceEngine, ServerConfig

        cfg = ServerConfig(model="test")
        engine = InferenceEngine(cfg)
        engine.tokenizer = MagicMock()
        engine.tokenizer.chat_template = None
        engine.tokenizer.apply_chat_template.side_effect = Exception("no template")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        prompt = _build_chat_prompt(engine, messages)
        assert "System:" in prompt
        assert "User:" in prompt
        assert "Assistant:" in prompt

    def test_chat_completion_format(self):
        from quantllm.server.engine import create_chat_completion
        from quantllm.server import InferenceEngine, ServerConfig

        cfg = ServerConfig(model="test-model")
        engine = InferenceEngine(cfg)
        engine.tokenizer = MagicMock()
        engine.tokenizer.chat_template = None
        engine.generate = MagicMock(return_value="Hello, world!")
        engine.get_token_count = MagicMock(return_value=10)

        messages = [{"role": "user", "content": "Hello!"}]
        result = create_chat_completion(engine, messages)
        assert result["object"] == "chat.completion"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Hello, world!"
        assert "usage" in result


class TestServerCLI:
    def test_serve_imports(self):
        from quantllm.server.app import run_server

        assert callable(run_server)


# ---------------------------------------------------------------------------
# Fake transformers-style model/tokenizer (CPU, no downloads)
# ---------------------------------------------------------------------------

_VOCAB = {0: "<pad>", 1: "Hello", 2: " world", 3: "STOP", 4: " after", 5: " more"}


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    chat_template = None

    def __init__(self):
        self.calls = []

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        import torch

        self.calls.append(add_special_tokens)
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def encode(self, text, add_special_tokens=True):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return "".join(_VOCAB[int(i)] for i in ids)

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("no chat template")


class _FakeModel:
    device = "cpu"

    def __init__(self, ids=(1, 2), chunks=None, delay=0.0, error=None):
        self.ids = list(ids)
        self.chunks = chunks
        self.delay = delay
        self.error = error
        self.calls = []

    def generate(self, input_ids=None, streamer=None, **kwargs):
        import time as _time

        import torch

        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if streamer is not None:
            for chunk in self.chunks or [_VOCAB[i] for i in self.ids]:
                _time.sleep(self.delay)
                streamer.on_finalized_text(chunk)
            return None
        _time.sleep(self.delay)
        return torch.cat([input_ids, torch.tensor([self.ids])], dim=1)


def _fake_engine(**model_kwargs):
    from quantllm.server import InferenceEngine, ServerConfig

    engine = InferenceEngine(ServerConfig(model="fake-model"))
    engine.tokenizer = _FakeTokenizer()
    engine.model = _FakeModel(**model_kwargs)
    return engine


def _sse_events(body: str):
    events = [line[len("data: ") :] for line in body.split("\n\n") if line.startswith("data: ")]
    assert body.endswith("data: [DONE]\n\n")
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


class TestServerFixes:
    def test_trust_remote_code_default_off(self):
        from quantllm.server import ServerConfig

        assert ServerConfig().trust_remote_code is False
        assert ServerConfig().gguf_file is None

    def test_gguf_detection(self, tmp_path):
        from quantllm.server import InferenceEngine, ServerConfig

        local = tmp_path / "m.gguf"
        local.write_bytes(b"")
        assert InferenceEngine(ServerConfig(model=str(local)))._is_local_gguf()
        hub = InferenceEngine(ServerConfig(model="org/Model-7B-GGUF"))
        assert hub._is_hub_gguf() and not hub._is_local_gguf()
        assert not InferenceEngine(ServerConfig(model="/missing/model.gguf"))._is_local_gguf()
        assert not InferenceEngine(ServerConfig(model="org/plain-model"))._is_hub_gguf()

    def test_no_eos_override_and_stop_strings(self):
        engine = _fake_engine(ids=[1, 3, 4])
        text = engine.generate("hi", max_tokens=8, stop=["STOP"])
        assert text == "Hello"
        kwargs = engine.model.calls[0]
        assert "eos_token_id" not in kwargs
        assert kwargs.get("stop_strings") == ["STOP"]

    def test_chat_template_prompt_skips_special_tokens(self):
        from quantllm.server.engine import create_chat_completion

        engine = _fake_engine()
        engine.tokenizer.apply_chat_template = lambda *a, **k: "<bos><user>hi<assistant>"
        create_chat_completion(engine, [{"role": "user", "content": "hi"}], max_tokens=8)
        assert engine.tokenizer.calls == [False]

    def test_filter_stops_emits_prefix_across_chunks(self):
        from quantllm.server.engine import _filter_stops

        assert "".join(_filter_stops(iter(["Hel", "loST", "OP after"]), ["STOP"])) == "Hello"
        assert "".join(_filter_stops(iter(["abc STOP"]), ["STOP"])) == "abc "
        assert "".join(_filter_stops(iter(["no", " stop"]), ["STOP"])) == "no stop"

    def test_stream_generation_error_does_not_hang(self):
        import threading

        engine = _fake_engine(error=RuntimeError("boom"))
        outcome = {}

        def consume():
            try:
                list(engine.generate("hi", max_tokens=4, stream=True))
            except RuntimeError as e:
                outcome["error"] = str(e)

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        t.join(timeout=15)
        assert not t.is_alive(), "stream hung after generate() raised"
        assert outcome.get("error") == "boom"
        # The model lock must be released so the next request can run.
        assert engine._lock.acquire(timeout=1)
        engine._lock.release()


class TestServerApp:
    @pytest.fixture(autouse=True)
    def _require_fastapi(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

    def _client(self, engine):
        from fastapi.testclient import TestClient

        from quantllm.server.app import _build_app

        return TestClient(_build_app(engine))

    def test_stop_as_string_and_content_parts(self):
        engine = _fake_engine(ids=[1, 3, 4])
        body = {
            "messages": [
                {"role": "system", "content": None},
                {"role": "user", "content": [{"type": "text", "text": "Hi "}, {"type": "text", "text": "there"}]},
            ],
            "stop": "STOP",
            "max_tokens": 16,
        }
        resp = self._client(engine).post("/v1/chat/completions", json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert engine.model.calls[0]["stop_strings"] == ["STOP"]

    def test_finish_reason_length(self):
        engine = _fake_engine(ids=[1, 2])
        resp = self._client(engine).post("/v1/completions", json={"prompt": "x", "max_tokens": 2})
        data = resp.json()
        assert data["choices"][0]["text"] == "Hello world"
        assert data["choices"][0]["finish_reason"] == "length"
        assert data["usage"]["completion_tokens"] == 2

    def test_chat_stream_format(self):
        engine = _fake_engine(chunks=["Hel", "loST", "OP after"])
        body = {"messages": [{"role": "user", "content": "hi"}], "stream": True, "stop": ["STOP"]}
        resp = self._client(engine).post("/v1/chat/completions", json=body)
        events = _sse_events(resp.text)
        assert all(e["object"] == "chat.completion.chunk" for e in events)
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        assert text == "Hello"
        assert events[-1]["choices"][0]["finish_reason"] == "stop"

    def test_completion_stream_format(self):
        engine = _fake_engine(chunks=["a", " b", " c"])
        resp = self._client(engine).post("/v1/completions", json={"prompt": "x", "stream": True, "max_tokens": 3})
        events = _sse_events(resp.text)
        for e in events:
            assert e["object"] == "text_completion"
            assert "text" in e["choices"][0] and "delta" not in e["choices"][0]
        assert "".join(e["choices"][0]["text"] for e in events) == "a b c"
        assert events[-1]["choices"][0]["finish_reason"] == "length"

    def test_stream_error_terminates(self):
        engine = _fake_engine(error=RuntimeError("boom"))
        resp = self._client(engine).post("/v1/completions", json={"prompt": "x", "stream": True})
        assert resp.text.endswith("data: [DONE]\n\n")
        assert "boom" in resp.text

    @pytest.mark.parametrize("stream", [False, True])
    def test_health_responsive_during_generation(self, stream):
        import asyncio
        import time as _time

        import httpx

        from quantllm.server.app import _build_app

        engine = _fake_engine(chunks=["a", "b", "c", "d"], delay=0.4)
        app = _build_app(engine)

        async def run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                done = {}

                async def gen():
                    await client.post("/v1/completions", json={"prompt": "x", "stream": stream})
                    done["gen"] = _time.perf_counter()

                async def health():
                    await asyncio.sleep(0.2)
                    r = await client.get("/health")
                    assert r.status_code == 200
                    done["health"] = _time.perf_counter()

                await asyncio.wait_for(asyncio.gather(gen(), health()), timeout=30)
                return done

        done = asyncio.run(run())
        assert done["health"] < done["gen"], "/health blocked while generating"
