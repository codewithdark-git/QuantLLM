"""Run the GPU end-to-end suite (tests/gpu) on Kaggle's free GPUs from your terminal.

Packs the local working tree (uncommitted changes included) into a private
Kaggle script kernel, runs it on a free GPU ("GPU T4 x2" by default), waits, and downloads the log.

    python scripts/kaggle_gpu_tests.py                 # fast tests
    python scripts/kaggle_gpu_tests.py --slow          # + GGUF/ONNX export tests
    python scripts/kaggle_gpu_tests.py -k Finetune     # pytest -k filter
    python scripts/kaggle_gpu_tests.py --accelerator NvidiaL4 --out gpu-test-results-l4

One-time setup: verify your phone on kaggle.com (needed for GPU + internet),
then create an API token and save it to ~/.kaggle/access_token (or kaggle.json).
Results land in ./gpu-test-results/. Exit code is pytest's.
"""

import argparse
import base64
import io
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACK = ["quantllm", "tests", "pyproject.toml", "README.md", "LICENSE"]
SLUG = "quantllm-gpu-tests"

KERNEL_TEMPLATE = r"""
import base64, io, os, shlex, subprocess, sys, tarfile

PAYLOAD = "{payload}"
ARGS = {args!r}
SLOW = {slow!r}
OUT = "/kaggle/working"
REPO = "/kaggle/tmp/repo"

def sh(cmd):
    print("$", cmd, flush=True)
    return subprocess.call(cmd, shell=True, executable="/bin/bash")

os.makedirs(REPO, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz").extractall(REPO)
os.chdir(REPO)
os.environ["QUANTLLM_BANNER"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

sh("nvidia-smi")
sh(f"nvidia-smi --query-gpu=name --format=csv,noheader > {{OUT}}/gpu.txt")
sh("pip install -q -e '.[server]' pytest httpx 'bitsandbytes>=0.43'")
if SLOW:
    sh("pip install -q llama-cpp-python onnx onnxscript")
sh("python -c 'import torch,transformers,bitsandbytes as b;print(torch.__version__,transformers.__version__,b.__version__)'")

mark = [] if SLOW else ["-m", "'not slow'"]
cmd = " ".join(["python -m pytest tests/gpu -v -rA --tb=short", f"--junitxml={{OUT}}/junit.xml", *mark, *map(shlex.quote, ARGS)])
print("$", cmd, flush=True)
proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
with open(f"{{OUT}}/pytest.log", "w") as log:
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
rc = proc.wait()
open(f"{{OUT}}/exit_code.txt", "w").write(str(rc))
print("PYTEST_EXIT_CODE", rc)
"""


def _pack() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in PACK:
            path = ROOT / name
            if path.exists():
                tar.add(path, arcname=name, filter=lambda t: None if "__pycache__" in t.name else t)
    return base64.b64encode(buf.getvalue()).decode()


def _kaggle(*args: str) -> str:
    res = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{res.stdout}\n{res.stderr}")
    return res.stdout


def _username() -> str:
    out = _kaggle("config", "view")
    for line in out.splitlines():
        if "username" in line.lower():
            return line.split(":", 1)[1].strip()
    raise SystemExit("Could not determine Kaggle username; pass --user")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slow", action="store_true", help="include slow export tests")
    p.add_argument("-k", default="", help="pytest -k expression")
    p.add_argument(
        "--accelerator", default="NvidiaTeslaT4", help="Kaggle accelerator, e.g. NvidiaTeslaT4 (T4 x2), NvidiaL4"
    )
    p.add_argument("--user", default="", help="Kaggle username (default: from kaggle config)")
    p.add_argument("--timeout", type=int, default=3 * 3600, help="max seconds to wait")
    p.add_argument("--out", default=str(ROOT / "gpu-test-results"))
    a = p.parse_args()

    user = a.user or _username()
    ref = f"{user}/{SLUG}"
    args = ["-k", a.k] if a.k else []

    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / "run.py"
        src.write_text(KERNEL_TEMPLATE.format(payload=_pack(), args=args, slow=a.slow), encoding="utf-8")
        meta = {
            "id": ref,
            "title": SLUG,
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (pathlib.Path(d) / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        print(_kaggle("kernels", "push", "-p", d, "--accelerator", a.accelerator).strip())

    print(f"Running https://www.kaggle.com/code/{ref} ...", flush=True)
    start = time.time()
    status = ""
    while time.time() - start < a.timeout:
        time.sleep(60)
        status = _kaggle("kernels", "status", ref).lower()
        print(f"[{int(time.time() - start) // 60:3d} min] {status.strip()}", flush=True)
        if any(s in status for s in ("complete", "error", "cancel")):
            break

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "output", ref, "-p", str(out))
    log = out / "pytest.log"
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-60:]))
    gpu_file = out / "gpu.txt"
    gpus = gpu_file.read_text().strip().splitlines() if gpu_file.exists() else []
    print(f"\nRan on: {', '.join(gpus) or 'unknown'}")
    wanted = a.accelerator.lower().replace("nvidia", "").replace("tesla", "")
    if gpus and not any(wanted in g.lower().replace(" ", "") for g in gpus):
        print(
            f"WARNING: requested {a.accelerator} but Kaggle ran on {gpus[0]} (accelerator not available to this account)"
        )
    code_file = out / "exit_code.txt"
    rc = int(code_file.read_text().strip()) if code_file.exists() else 1
    print(f"\nKernel status: {status.strip()}\npytest exit code: {rc}\nFull output in {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
