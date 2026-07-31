"""
QuantLLM Server -- FastAPI application, OpenAI-compatible endpoints.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator

logger = logging.getLogger("quantllm.server")

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    class _ChatMessage(BaseModel):
        role: str
        content: str

    class _ChatRequest(BaseModel):
        model: str = ""
        messages: list[_ChatMessage]
        max_tokens: int = 512
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False
        stop: list[str] | None = None

    class _CompletionRequest(BaseModel):
        model: str = ""
        prompt: str
        max_tokens: int = 512
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False
        stop: list[str] | None = None

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


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

        result = create_chat_completion(
            engine=engine,
            messages=[m.model_dump() for m in req.messages],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=req.stream,
            stop=req.stop,
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

        result = create_completion(
            engine=engine,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=req.stream,
            stop=req.stop,
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

    async for chunk in engine.generate(
        prompt,
        max_tokens=result["max_tokens"],
        temperature=result["temperature"],
        top_p=result["top_p"],
        stop=result["stop"],
        stream=True,
    ):
        delta = {"content": chunk} if is_chat else {"text": chunk}
        data = {
            "id": cid,
            "object": "chat.completion.chunk" if is_chat else "text_completion.chunk",
            "created": created,
            "model": engine.config.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        yield f"data: {json.dumps(data)}\n\n"

    data = {
        "id": cid,
        "object": "chat.completion.chunk" if is_chat else "text_completion.chunk",
        "created": created,
        "model": engine.config.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(data)}\n\n"
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
