"""
QuantLLM Server -- inference engine abstraction and server configuration.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, AsyncGenerator, Dict, Generator, Iterator, List, Optional, Tuple

import torch

from ..utils.hf_compat import dtype_kwargs


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
    trust_remote_code: bool = False
    gguf_file: Optional[str] = None  # filename glob inside a hub GGUF repo, e.g. "*Q4_K_M.gguf"

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
        # Serializes model access: one generation (including a full stream) at a time.
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._detect_gguf():
            self._load_llama_cpp()
        else:
            self._load_transformers()

    def _is_local_gguf(self) -> bool:
        path = self.config.model
        return path.lower().endswith(".gguf") and os.path.isfile(path)

    def _is_hub_gguf(self) -> bool:
        name = self.config.model
        if not name or os.path.exists(name) or name.lower().endswith(".gguf"):
            return False
        return bool(self.config.gguf_file) or "gguf" in name.lower()

    def _detect_gguf(self) -> bool:
        if self._is_local_gguf() or self._is_hub_gguf():
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
            **dtype_kwargs(dtype),
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.eval()

    def _load_llama_cpp(self) -> None:
        from llama_cpp import Llama

        if self._is_local_gguf():
            self.model = Llama(model_path=self.config.model, n_ctx=self.config.max_seq_length, verbose=False)
        else:
            self.model = Llama.from_pretrained(
                repo_id=self.config.model,
                filename=self.config.gguf_file or "*Q4_K_M.gguf",
                n_ctx=self.config.max_seq_length,
                verbose=False,
            )
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
        messages: Optional[List[Dict[str, Any]]] = None,
        add_special_tokens: bool = True,
    ):
        """Generate text; ``stream=True`` returns a blocking iterator of text chunks.

        ``messages`` (chat requests) lets llama-cpp apply the GGUF-embedded chat template.
        Pass ``add_special_tokens=False`` when ``prompt`` came from a chat template (it already has BOS).
        """
        if stream:
            return self._stream(prompt, max_tokens, temperature, top_p, stop, messages, add_special_tokens)
        with self._lock:
            if self.backend == "llama-cpp-python":
                return self._gen_llama_cpp(prompt, max_tokens, temperature, top_p, stop, messages)
            return self._gen_transformers(prompt, max_tokens, temperature, top_p, stop, add_special_tokens)

    def _stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        messages: Optional[List[Dict[str, Any]]],
        add_special_tokens: bool,
    ) -> Iterator[str]:
        # The lock is held for the whole stream so two requests never generate on one model concurrently.
        with self._lock:
            if self.backend == "llama-cpp-python":
                chunks = self._stream_llama_cpp(prompt, max_tokens, temperature, top_p, stop, messages)
            else:
                inputs = self._encode(prompt, add_special_tokens)
                chunks = self._stream_transformers(inputs, max_tokens, temperature, top_p, stop)
            try:
                yield from _filter_stops(chunks, stop)
            finally:
                chunks.close()

    def _encode(self, prompt: str, add_special_tokens: bool = True) -> Dict:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _gen_kwargs(self, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]) -> Dict:
        # eos_token_id is intentionally not passed: generation_config may define several EOS ids.
        kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if stop and _supports_stop_strings():
            kwargs["stop_strings"] = stop
            kwargs["tokenizer"] = self.tokenizer
        return kwargs

    def _gen_transformers(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        add_special_tokens: bool = True,
    ):
        inputs = self._encode(prompt, add_special_tokens)
        outputs = self.model.generate(**inputs, **self._gen_kwargs(max_tokens, temperature, top_p, stop))
        text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return self._apply_stops(text, stop)

    def _stream_transformers(
        self, inputs: Dict, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]
    ) -> Generator[str, None, None]:
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300)
        cancel = threading.Event()

        class _Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), cancel.is_set(), dtype=torch.bool, device=input_ids.device)

        kwargs = {
            **inputs,
            **self._gen_kwargs(max_tokens, temperature, top_p, stop),
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([_Cancelled()]),
        }
        errors: List[BaseException] = []

        def _worker():
            try:
                with torch.inference_mode():
                    self.model.generate(**kwargs)
            except BaseException as e:  # re-raised in the consumer after the stream ends
                errors.append(e)
            finally:
                streamer.end()  # always unblock the consumer, even if generate() raised

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        try:
            for text in streamer:
                yield text
        finally:
            cancel.set()  # halts generation if the consumer stopped early (stop string / disconnect)
            thread.join()
        if errors:
            raise errors[0]

    def _gen_llama_cpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        messages: Optional[List[Dict[str, Any]]] = None,
    ):
        if messages is not None:
            # create_chat_completion applies the chat template embedded in the GGUF metadata.
            out = self.model.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop or []
            )
            return out["choices"][0]["message"]["content"] or ""
        return self.model(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop or [])[
            "choices"
        ][0]["text"]

    def _stream_llama_cpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        if messages is not None:
            for chunk in self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                stream=True,
            ):
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content:
                    yield content
            return
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
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        if self.backend == "llama-cpp-python" and self.model is not None:
            return len(self.model.tokenize(text.encode("utf-8"), add_bos=False))
        return len(text) // 4


def _supports_stop_strings() -> bool:
    """``generate(stop_strings=..., tokenizer=...)`` exists since transformers 4.39."""
    try:
        import transformers
        from packaging.version import Version

        return Version(transformers.__version__) >= Version("4.39.0")
    except Exception:
        return False


def _filter_stops(chunks: Iterator[str], stop: Optional[List[str]]) -> Iterator[str]:
    """Yield streamed text up to (not including) the first stop string, even if it spans chunks."""
    if not stop:
        yield from chunks
        return
    hold = max(len(s) for s in stop) - 1  # chars held back in case a stop string is split across chunks
    text, emitted = "", 0
    for chunk in chunks:
        text += chunk
        hits = [i for i in (text.find(s, max(0, emitted - hold)) for s in stop) if i != -1]
        if hits:
            idx = min(hits)
            if idx > emitted:
                yield text[emitted:idx]
            return
        safe = len(text) - hold
        if safe > emitted:
            yield text[emitted:safe]
            emitted = safe
    if len(text) > emitted:
        yield text[emitted:]


def finish_reason(completion_tokens: int, max_tokens: int) -> str:
    return "length" if completion_tokens >= max_tokens else "stop"


def _chat_prompt(engine: InferenceEngine, messages: List[Dict[str, str]]) -> Tuple[str, bool]:
    """Return ``(prompt, from_template)``; template prompts already contain special tokens (BOS)."""
    if engine.tokenizer is not None and hasattr(engine.tokenizer, "chat_template"):
        try:
            return engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), True
        except Exception:
            pass
    parts = []
    for m in messages:
        r, c = m.get("role", "user"), m.get("content", "")
        parts.append(f"{r.capitalize()}: {c}\n")
    parts.append("Assistant:")
    return "".join(parts), False


def _build_chat_prompt(engine: InferenceEngine, messages: List[Dict[str, str]]) -> str:
    return _chat_prompt(engine, messages)[0]


def create_chat_completion(
    engine: InferenceEngine,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
    stop: Optional[List[str]] = None,
) -> Dict:
    prompt, from_template = _chat_prompt(engine, messages)
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
            "messages": messages,
            "add_special_tokens": not from_template,
        }

    text = engine.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        stream=False,
        messages=messages,
        add_special_tokens=not from_template,
    )
    ct, pt = engine.get_token_count(text), engine.get_token_count(prompt)
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": engine.config.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason(ct, max_tokens),
            }
        ],
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
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason(ct, max_tokens)}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }
