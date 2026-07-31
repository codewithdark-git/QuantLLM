"""
QuantLLM Server -- inference engine abstraction and server configuration.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    model: str = ""
    max_batch_size: int = 1
    max_seq_length: int = 4096
    dtype: str = "auto"
    quant_method: Optional[str] = None
    bits: Optional[int] = None
    trust_remote_code: bool = True

    @classmethod
    def from_cli(cls, **kwargs) -> "ServerConfig":
        cfg = cls()
        for k, v in kwargs.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
        if not cfg.model:
            cfg.model = os.environ.get("QUANTLLM_MODEL", "")
        return cfg


class InferenceEngine:
    """Abstraction over transformers and llama-cpp-python backends."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None
        self.backend = "transformers"

    def load(self) -> None:
        if self._detect_gguf():
            self._load_llama_cpp()
        else:
            self._load_transformers()

    def _detect_gguf(self) -> bool:
        name = self.config.model.lower()
        if name.endswith(".gguf") or "gguf" in name:
            try:
                import llama_cpp

                return True
            except ImportError:
                pass
        return False

    def _load_transformers(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=self.config.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16}.get(self.config.dtype, "auto")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.eval()

    def _load_llama_cpp(self) -> None:
        from llama_cpp import Llama

        self.model = Llama(model_path=self.config.model, n_ctx=self.config.max_seq_length, verbose=False)
        self.backend = "llama-cpp-python"

    def warmup(self) -> None:
        if self.backend == "transformers" and self.model is not None:
            dummy = self.tokenizer("warmup", return_tensors="pt")
            dummy = {k: v.to(self.model.device) for k, v in dummy.items()}
            with torch.inference_mode():
                self.model.generate(**dummy, max_new_tokens=1)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ):
        if self.backend == "llama-cpp-python":
            return self._gen_llama_cpp(prompt, max_tokens, temperature, top_p, stop, stream)
        return self._gen_transformers(prompt, max_tokens, temperature, top_p, stop, stream)

    def _gen_transformers(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]], stream: bool
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if stream:
            return self._stream_transformers(inputs, max_tokens, temperature, top_p, stop)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return self._apply_stops(text, stop)

    async def _stream_transformers(
        self, inputs: Dict, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]
    ):
        from threading import Thread
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        collected = []
        for text in streamer:
            collected.append(text)
            if any(s in "".join(collected) for s in (stop or [])):
                break
            yield text
        thread.join()

    def _gen_llama_cpp(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]], stream: bool
    ):
        if stream:
            return self._stream_llama_cpp(prompt, max_tokens, temperature, top_p, stop)
        return self.model(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop or [])[
            "choices"
        ][0]["text"]

    async def _stream_llama_cpp(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]
    ):
        for chunk in self.model(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop or [], stream=True
        ):
            yield chunk["choices"][0]["text"]

    @staticmethod
    def _apply_stops(text: str, stop: Optional[List[str]]) -> str:
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return text.strip()

    def get_token_count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        return len(text) // 4


def _build_chat_prompt(engine: InferenceEngine, messages: List[Dict[str, str]]) -> str:
    if engine.tokenizer is not None and hasattr(engine.tokenizer, "chat_template"):
        try:
            return engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    parts = []
    for m in messages:
        r, c = m.get("role", "user"), m.get("content", "")
        parts.append(f"{r.capitalize()}: {c}\n")
    parts.append("Assistant:")
    return "".join(parts)


def create_chat_completion(
    engine: InferenceEngine,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
    stop: Optional[List[str]] = None,
) -> Dict:
    prompt = _build_chat_prompt(engine, messages)
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if stream:
        return {
            "type": "stream",
            "id": cid,
            "created": created,
            "model": engine.config.model,
            "engine": engine,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }

    text = engine.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop, stream=False)
    ct, pt = engine.get_token_count(text), engine.get_token_count(prompt)
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": engine.config.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


def create_completion(
    engine: InferenceEngine,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
    stop: Optional[List[str]] = None,
) -> Dict:
    cid = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if stream:
        return {
            "type": "stream",
            "id": cid,
            "created": created,
            "model": engine.config.model,
            "engine": engine,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "is_chat": False,
        }

    text = engine.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop, stream=False)
    ct, pt = engine.get_token_count(text), engine.get_token_count(prompt)
    return {
        "id": cid,
        "object": "text_completion",
        "created": created,
        "model": engine.config.model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }
