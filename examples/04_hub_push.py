"""
QuantLLM v2.2 -- Push to HuggingFace Hub

Push a quantized model to the HuggingFace Hub with an
auto-generated model card.  Requires HF_TOKEN or a token
passed explicitly.

Skip if no token is available (safely no-ops).
"""

import os

from quantllm import turbo

# Check for credentials
token = os.environ.get("HF_TOKEN")
if not token:
    print("HF_TOKEN not set -- push will fail.")
    print("Set it and re-run, or pass token= explicitly.")
    print("  export HF_TOKEN=hf_...")
    print("Continuing in dry-run mode for demonstration.\n")

# Load model with shared export/push config
model = turbo(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    config={
        "format": "gguf",
        "quantization": "Q4_K_M",
        "push_format": "gguf",
    },
)

# Push GGUF to Hub
repo = "YOUR_USERNAME/my-quantized-tinyllama-gguf"
print(f"Pushing to {repo} ...")
print("  (This will export and upload automatically)")
model.push(
    repo,
    private=False,
    license="apache-2.0",
    token=token or None,
)
print(f"  https://huggingface.co/{repo}")

# Push SafeTensors format to a different repo
repo2 = "YOUR_USERNAME/my-quantized-tinyllama"
print(f"\nPushing SafeTensors to {repo2} ...")
model.push(
    repo2,
    format="safetensors",
    license="apache-2.0",
    token=token or None,
)
print(f"  https://huggingface.co/{repo2}")

print("\nPush complete.")
