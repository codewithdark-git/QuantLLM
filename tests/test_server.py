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
