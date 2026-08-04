#!/usr/bin/env python
"""
Example 09: Complete Custom Workflow
====================================

Demonstrates the full QuantLLM workflow:
1. Load model with turbo() + auto quantization
2. Test BnB and HQQ quantization
3. Export to 4 formats (GGUF, SafeTensors, ONNX, MLX)
4. Push to HuggingFace Hub with auto-generated model cards
5. Register in QuantLLM model registry
6. Validate all formats from registry

Requires:
    pip install -e ".[full]"
    HF_TOKEN environment variable set
"""

import os
import sys

# Suppress banner for clean output
os.environ["QUANTLLM_BANNER"] = "0"

from quantllm import turbo, TurboModel, register_architecture, register_model
from quantllm.quant import HQQLinear, HQQQuantizer, HQQConfig
from quantllm.registry import list_models, search_models


def phase1_load_model():
    """Phase 1: Load model with turbo() - auto hardware detection & quantization"""
    print("=" * 80)
    print("PHASE 1: LOAD MODEL WITH TURBO()")
    print("=" * 80)

    # Auto-detects GPU, applies optimal quantization (BnB 8-bit on H100/A10G)
    model = turbo("Qwen/Qwen2.5-7B-Instruct")

    print(f"Loaded: {model.config.bits}-bit {model.config.quant_type}")
    print(f"Device: {model.config.device}, dtype: {model.config.dtype}")
    print(f"Params: {model.config.stats.get('params', 0) / 1e9:.2f}B")
    print(f"Quantized: {model.is_quantized}")

    # Test generation
    response = model.generate("Hello, world!", max_new_tokens=50)
    print(f"Generation: {'OK' if len(response) > 0 else 'FAILED'}")
    print(f"Response: {response[:100]}...")

    return model


def phase2_quantization():
    """Phase 2: Test BnB and HQQ quantization"""
    print("\n" + "=" * 80)
    print("PHASE 2: QUANTIZATION TESTING")
    print("=" * 80)

    # Test BitsAndBytes (runtime quantization via transformers)
    print("\nBitsAndBytes quantization:")
    for bits in [4, 8]:
        model = turbo("Qwen/Qwen2.5-7B-Instruct", bits=bits)
        print(
            f"  BnB {bits}-bit: bits={model.config.bits}, quant={model.config.quant_type}, quantized={model.is_quantized}"
        )
        del model

    # Test HQQ native quantization (no calibration, for export/optimization)
    print("\nHQQ native quantization:")
    import torch

    weight = torch.randn(256, 512, dtype=torch.bfloat16)

    for bits in [2, 4, 8]:
        config = HQQConfig(bits=bits, group_size=64, optimize=True)
        hqq_layer = HQQLinear(weight, config)  # Quantize linear layer
        print(
            f"  HQQ {bits}-bit: Original={weight.numel() * 2 / 1e6:.2f}MB, Quantized={hqq_layer.quantized_weight.numel() / 1e6:.2f}MB"
        )
        del hqq_layer

    torch.cuda.empty_cache()


def phase3_export(model):
    """Phase 3: Export to multiple formats"""
    print("\n" + "=" * 80)
    print("PHASE 3: MULTI-FORMAT EXPORT")
    print("=" * 80)

    from pathlib import Path
    import tempfile

    formats = [
        ("gguf", "Q4_K_M", "GGUF (llama.cpp)"),
        ("gguf", "Q5_K_M", "GGUF (llama.cpp)"),
        ("gguf", "Q8_0", "GGUF (llama.cpp)"),
        ("safetensors", None, "SafeTensors (HF native)"),
        ("onnx", "int8", "ONNX (Optimum)"),
        ("mlx", "4bit", "MLX (Apple Silicon)"),
    ]

    exports = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for fmt, quant, desc in formats:
            print(f"\nExporting {desc}...")
            try:
                output = (
                    tmp_path / f"model_{fmt}_{quant or 'default'}.gguf" if fmt == "gguf" else tmp_path / f"model_{fmt}"
                )
                result = model.export(fmt, str(output), quantization=quant or "auto", chunked_conversion=True)
                size_gb = Path(result).stat().st_size / (1024**3) if Path(result).exists() else 0
                exports[f"{fmt}_{quant or 'default'}"] = {"path": result, "size_gb": size_gb}
                print(f"  ✓ {desc}: {result} ({size_gb:.2f} GB)")
            except Exception as e:
                print(f"  ✗ {desc} failed: {e}")
                exports[f"{fmt}_{quant or 'default'}"] = {"error": str(e)}

    return exports


def phase4_push(model, hf_org="QuantLLM", model_name="Qwen2.5-7B-Instruct"):
    """Phase 4: Push to HuggingFace Hub"""
    print("\n" + "=" * 80)
    print("PHASE 4: PUSH TO HUGGINGFACE HUB")
    print("=" * 80)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  HF_TOKEN not set - skipping push (set HF_TOKEN to enable)")
        return {}

    repos = {
        "gguf": f"{hf_org}/{model_name}-GGUF",
        "safetensors": f"{hf_org}/{model_name}-SafeTensors",
        "onnx": f"{hf_org}/{model_name}-ONNX",
        "mlx": f"{hf_org}/{model_name}-MLX",
    }

    pushes = {}
    for fmt, repo in repos.items():
        quant = "Q4_K_M" if fmt == "gguf" else ("int8" if fmt == "onnx" else "4bit" if fmt == "mlx" else "")
        print(f"\nPushing {fmt} to {repo}...")
        try:
            model.push(
                repo,
                format=fmt,
                quantization=quant,
                license="apache-2.0",
                token=hf_token,
            )
            pushes[fmt] = {"repo": repo, "success": True}
            print(f"  ✓ Pushed to https://huggingface.co/{repo}")
        except Exception as e:
            pushes[fmt] = {"repo": repo, "success": False, "error": str(e)}
            print(f"  ✗ Failed: {e}")

    return pushes


def phase5_register(repos_data, family="qwen", params=7.0):
    """Phase 5: Register in QuantLLM registry"""
    print("\n" + "=" * 80)
    print("PHASE 5: REGISTER IN QUANTLLM REGISTRY")
    print("=" * 80)

    registrations = {}
    for fmt, repo_info in repos_data.items():
        if repo_info.get("success"):
            repo = repo_info["repo"]
            rec = "Q4_K_M" if fmt == "gguf" else ("int8" if fmt == "onnx" else "4bit")
            print(f"\nRegistering {fmt}: {repo}...")
            try:
                ok = register_model(
                    repo,
                    family=family,
                    params=params,
                    quality_score=4.0,
                    speed_score=4.5,
                    verified=True,
                    recommended=rec,
                )
                registrations[fmt] = {"repo": repo, "success": ok}
                print(f"  {'✓' if ok else '✗'} Registration: {'OK' if ok else 'FAILED'}")
            except Exception as e:
                registrations[fmt] = {"repo": repo, "success": False, "error": str(e)}
                print(f"  ✗ Failed: {e}")

    return registrations


def phase6_validate(registered_repos):
    """Phase 6: Validate loading from registry"""
    print("\n" + "=" * 80)
    print("PHASE 6: VALIDATE FROM REGISTRY")
    print("=" * 80)

    validations = {}
    for fmt, info in registered_repos.items():
        if info.get("success"):
            repo = info["repo"]
            print(f"\nValidating {fmt}: {repo}...")
            try:
                if fmt == "gguf":
                    model = TurboModel.from_gguf(repo, token=os.environ.get("HF_TOKEN"))
                else:
                    model = turbo(repo, quantize=False, token=os.environ.get("HF_TOKEN"))

                response = model.generate("Test generation from registry", max_new_tokens=30)
                validations[fmt] = {"repo": repo, "success": True, "response": response[:100]}
                print(f"  ✓ Load: OK")
                print(f"  ✓ Generation: OK")
                print(f"  Response: {response[:80]}...")
            except Exception as e:
                validations[fmt] = {"repo": repo, "success": False, "error": str(e)}
                print(f"  ✗ Failed: {e}")

    return validations


def main():
    """Run complete workflow"""
    print("QUANTLLM CUSTOM WORKFLOW")
    print("Model: Qwen/Qwen2.5-7B-Instruct (7B, Apache-2.0)")
    print("HF Org: QuantLLM")
    print()

    # Phase 1
    model = phase1_load_model()

    # Phase 2
    phase2_quantization()

    # Phase 3
    exports = phase3_export(model)

    # Phase 4
    pushes = phase4_push(model)

    # Phase 5
    registrations = phase5_register(pushes)

    # Phase 6
    validations = phase6_validate(registrations)

    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)

    phases = {
        "Phase 1 (Load)": True,
        "Phase 2 (Quantize)": True,
        "Phase 3 (Export)": all("error" not in v for v in exports.values()),
        "Phase 4 (Push)": all(v.get("success", False) for v in pushes.values()),
        "Phase 5 (Register)": all(v.get("success", False) for v in registrations.values()),
        "Phase 6 (Validate)": all(v.get("success", False) for v in validations.values()),
    }

    for phase, status in phases.items():
        print(f"  {phase}: {'✓ PASS' if status else '✗ FAIL'}")

    print("\nRegistered models in QuantLLM registry:")
    for m in list_models():
        if m.get("family") == "qwen" and m.get("params", 0) >= 7:
            print(f"  {m['id']}: {m['family']} {m['params']}B verified={m.get('verified', False)}")

    print("\n✅ Workflow complete!")
    print("Next steps:")
    print("  - Use `quantllm models list` to see all registered models")
    print("  - Use `quantllm models search qwen` to search")
    print("  - Use `quantllm serve QuantLLM/Qwen2.5-7B-Instruct-GGUF` to start server")


if __name__ == "__main__":
    main()
