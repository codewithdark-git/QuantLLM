"""
QuantLLM Server -- FastAPI application, OpenAI-compatible endpoints.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator, List, Optional, Union

logger = logging.getLogger("quantllm.server")

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

    class _ChatMessage(BaseModel):
        role: str
        # OpenAI allows a string, null, or a list of content parts ({"type": "text", "text": ...}).
        content: Optional[Union[str, List[Any]]] = None

        def text(self) -> str:
            if self.content is None:
                return ""
            if isinstance(self.content, str):
                return self.content
            return "".join(p.get("text", "") for p in self.content if isinstance(p, dict) and p.get("type") == "text")

    class _ChatRequest(BaseModel):
        model: str = ""
        messages: List[_ChatMessage]
        max_tokens: int = 512
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False
        stop: Optional[Union[str, List[str]]] = None

    class _CompletionRequest(BaseModel):
        model: str = ""
        prompt: str
        max_tokens: int = 512
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False
        stop: Optional[Union[str, List[str]]] = None

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def _normalize_stop(stop) -> Optional[List[str]]:
    if stop is None:
        return None
    if isinstance(stop, str):
        stop = [stop]
    return [s for s in stop if s] or None


def _build_app(engine) -> FastAPI:
    app = FastAPI(title="QuantLLM Server", version="2.2.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": engine.config.model}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.config.model or "default",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "quantllm",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(req: _ChatRequest):
        if engine.model is None:
            raise HTTPException(503, "Model not loaded")
        from .engine import create_chat_completion

        # Blocking generation runs in a worker thread so the event loop (and /health) stays responsive.
        result = await run_in_threadpool(
            create_chat_completion,
            engine=engine,
            messages=[{"role": m.role, "content": m.text()} for m in req.messages],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=req.stream,
            stop=_normalize_stop(req.stop),
        )
        if req.stream and isinstance(result, dict) and result.get("type") == "stream":
            return StreamingResponse(
                _stream_chunks(result, is_chat=True),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return result

    @app.post("/v1/completions")
    async def completions(req: _CompletionRequest):
        if engine.model is None:
            raise HTTPException(503, "Model not loaded")
        from .engine import create_completion

        result = await run_in_threadpool(
            create_completion,
            engine=engine,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=req.stream,
            stop=_normalize_stop(req.stop),
        )
        if req.stream and isinstance(result, dict) and result.get("type") == "stream":
            return StreamingResponse(
                _stream_chunks(result, is_chat=False),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return result

    return app


async def _stream_chunks(result: dict, is_chat: bool) -> AsyncGenerator[str, None]:
    engine = result["engine"]
    prompt = result["prompt"]
    cid = result["id"]
    created = result["created"]

    from .engine import finish_reason

    obj = "chat.completion.chunk" if is_chat else "text_completion"

    def _event(choice: dict) -> str:
        data = {"id": cid, "object": obj, "created": created, "model": engine.config.model, "choices": [choice]}
        return f"data: {json.dumps(data)}\n\n"

    def _choice(text: Optional[str], reason: Optional[str], role: bool = False) -> dict:
        if is_chat:
            delta: dict = {"role": "assistant", "content": ""} if role else ({"content": text} if text else {})
            return {"index": 0, "delta": delta, "finish_reason": reason}
        return {"index": 0, "text": text or "", "logprobs": None, "finish_reason": reason}

    if is_chat:
        yield _event(_choice(None, None, role=True))

    kwargs = {}
    if "messages" in result:
        kwargs = {"messages": result["messages"], "add_special_tokens": result["add_special_tokens"]}
    collected = []
    try:
        # engine.generate(stream=True) is a blocking iterator; pull each chunk from a worker thread.
        chunks = await run_in_threadpool(
            engine.generate,
            prompt,
            max_tokens=result["max_tokens"],
            temperature=result["temperature"],
            top_p=result["top_p"],
            stop=result["stop"],
            stream=True,
            **kwargs,
        )
        async for chunk in iterate_in_threadpool(chunks):
            if chunk:
                collected.append(chunk)
                yield _event(_choice(chunk, None))
    except Exception as e:
        logger.exception("Generation failed during streaming")
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': type(e).__name__}})}\n\n"
        yield "data: [DONE]\n\n"
        return

    ct = engine.get_token_count("".join(collected))
    yield _event(_choice(None, finish_reason(ct, result["max_tokens"])))
    yield "data: [DONE]\n\n"


def run_server(config):
    if not HAS_FASTAPI:
        print("FastAPI required: pip install fastapi uvicorn")
        return
    import uvicorn
    from .engine import InferenceEngine

    engine = InferenceEngine(config)
    print(f"Loading model: {config.model}...")
    engine.load()
    print("Warming up...")
    engine.warmup()
    print(f"Server ready at http://{config.host}:{config.port}")
    uvicorn.run(_build_app(engine), host=config.host, port=config.port, log_level="warning")
