"""Run the GPU end-to-end suite (tests/gpu) on a real GPU via Modal.

No local GPU needed. The local checkout is mounted into the container, so this
always tests your working tree, uncommitted changes included.

    modal run scripts/modal_gpu_tests.py                       # T4, fast tests
    modal run scripts/modal_gpu_tests.py --gpu A10G --slow     # include GGUF/ONNX
    modal run scripts/modal_gpu_tests.py --gpu H100 -k Finetune

Exit code is pytest's, so CI can gate on it. Requires `modal token new` once
(or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment).
"""

import os
import pathlib
import subprocess
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/repo"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "accelerate>=0.34",
        "peft>=0.12",
        "bitsandbytes>=0.43",
        "datasets>=2.14",
        "sentencepiece",
        "protobuf",
        "einops",
        "psutil",
        "scipy",
        "scikit-learn",
        "huggingface-hub",
        "gguf>=0.10.0",
        "fastapi",
        "httpx",
        "uvicorn",
        "pydantic>=2",
        "pytest>=7.4",
    )
    .env({"QUANTLLM_BANNER": "0", "HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "quantllm", f"{REMOTE_ROOT}/quantllm")
    .add_local_dir(ROOT / "tests", f"{REMOTE_ROOT}/tests")
    .add_local_file(ROOT / "pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml")
)

# Heavier optional backends only for --slow runs (llama.cpp build takes minutes).
slow_image = image.pip_install("llama-cpp-python", "onnx", "onnxscript")

app = modal.App("quantllm-gpu-tests")
hf_cache = modal.Volume.from_name("quantllm-hf-cache", create_if_missing=True)

_secrets = []
try:
    _secrets.append(modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"]))
except Exception:  # older modal versions or no secret configured
    pass


def _pytest(args: list[str], model: str = "") -> int:
    if model:
        os.environ["QUANTLLM_GPU_TEST_MODEL"] = model
    cmd = [sys.executable, "-m", "pytest", "tests/gpu", "-v", "-rA", "--tb=short", *args]
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REMOTE_ROOT)
    subprocess.call(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv"])
    hf_cache.commit()
    return rc


_common = dict(volumes={"/cache": hf_cache}, timeout=60 * 60, secrets=_secrets)


@app.function(image=image, gpu="T4", **_common)
def run_t4(args: list[str], model: str = "") -> int:
    return _pytest(args, model)


@app.function(image=image, gpu="A10G", **_common)
def run_a10g(args: list[str], model: str = "") -> int:
    return _pytest(args, model)


@app.function(image=image, gpu="L4", **_common)
def run_l4(args: list[str], model: str = "") -> int:
    return _pytest(args, model)


@app.function(image=image, gpu="H100", **_common)
def run_h100(args: list[str], model: str = "") -> int:
    return _pytest(args, model)


@app.function(image=slow_image, gpu="A10G", **_common)
def run_slow_a10g(args: list[str], model: str = "") -> int:
    return _pytest(args, model)


@app.local_entrypoint()
def main(gpu: str = "T4", slow: bool = False, k: str = "", model: str = ""):
    args = [] if slow else ["-m", "not slow"]
    if k:
        args += ["-k", k]
    runners = {"T4": run_t4, "A10G": run_a10g, "L4": run_l4, "H100": run_h100}
    fn = run_slow_a10g if slow else runners[gpu.upper()]
    rc = fn.remote(args, model)
    print(f"\npytest exit code on {gpu if not slow else 'A10G (slow)'}: {rc}")
    sys.exit(rc)
