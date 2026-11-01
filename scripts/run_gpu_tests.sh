#!/usr/bin/env bash
# Run the GPU end-to-end suite on any machine with an NVIDIA GPU:
# a rented box, Kaggle (free 2xT4), Colab (free T4), Lightning, RunPod...
#
#   Colab/Kaggle cell:
#     !git clone https://github.com/codewithdark-git/QuantLLM && cd QuantLLM && bash scripts/run_gpu_tests.sh
#
#   Extra pytest args pass through, e.g.:  bash scripts/run_gpu_tests.sh -k Finetune
#   Include slow export tests:              SLOW=1 bash scripts/run_gpu_tests.sh
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=name,memory.total --format=csv || { echo "No NVIDIA GPU found"; exit 1; }

pip install -q -e ".[server]" pytest httpx
if [ "${SLOW:-0}" = "1" ]; then
  pip install -q llama-cpp-python onnx onnxscript  # add optimum-onnx[onnxruntime] only with transformers<4.58
  MARK=()
else
  MARK=(-m "not slow")
fi

QUANTLLM_BANNER=0 python -m pytest tests/gpu -v -rA --tb=short "${MARK[@]}" "$@"
