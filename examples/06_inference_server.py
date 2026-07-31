"""
QuantLLM v2.2 -- Inference Server

Start an OpenAI-compatible inference server.
Run this script to start the server; it will block and serve requests.
"""

from quantllm.server import ServerConfig
from quantllm.server.app import run_server

config = ServerConfig(
    host="0.0.0.0",
    port=8080,
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_seq_length=4096,
    dtype="auto",
)

print(f"Starting server on {config.host}:{config.port}")
print(f"  model: {config.model}")
print(f"  max_seq_length: {config.max_seq_length}")
print(f"  dtype: {config.dtype}")
print()
print("API endpoints:")
print(f"  GET  http://{config.host}:{config.port}/health")
print(f"  POST http://{config.host}:{config.port}/v1/chat/completions")
print(f"  POST http://{config.host}:{config.port}/v1/completions")
print()
print("Example request:")
print(f"  curl -X POST http://localhost:{config.port}/v1/chat/completions \\")
print('    -H "Content-Type: application/json" \\')
print('    -d \'{"messages":[{"role":"user","content":"Hello"}]}\'')
print()

run_server(config)
