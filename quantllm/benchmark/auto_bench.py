"""
QuantLLM Hardware Auto-Benchmark -- detects hardware and recommends optimal config.

Detects CPU, GPU, RAM; runs micro-benchmarks; scores quantization methods
and export formats; caches results to ~/.cache/quantllm/hardware_profile.json.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class HWProfile:
    cpu_cores: int = 0
    cpu_name: str = ""
    ram_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_compute: str = ""
    has_cuda: bool = False
    has_bf16: bool = False


@dataclass
class QuantScore:
    method: str
    bits: int
    quality: float
    speed: float
    vram: float
    overall: float


class HardwareAutoBench:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.hw = self._detect()
        self.cache_dir = os.path.expanduser("~/.cache/quantllm")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _detect(self) -> HWProfile:
        hw = HWProfile(cpu_cores=os.cpu_count() or 1)
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            hw.cpu_name = line.split(":")[1].strip()
                            break
        except Exception:
            hw.cpu_name = platform.processor() or "Unknown"
        try:
            import psutil

            hw.ram_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            hw.ram_gb = 16.0
        if HAS_TORCH and torch.cuda.is_available():
            hw.has_cuda = True
            try:
                p = torch.cuda.get_device_properties(0)
                hw.gpu_name = p.name
                hw.gpu_vram_gb = p.total_memory / (1024**3)
                hw.gpu_compute = f"{p.major}.{p.minor}"
                hw.has_bf16 = (p.major, p.minor) >= (8, 0)
            except Exception:
                pass
        return hw

    def _micro_bench(self) -> Dict[str, float]:
        results = {}
        if HAS_TORCH and torch.cuda.is_available():
            times = []
            for n in [1024, 2048]:
                a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
                start = time.time()
                for _ in range(10):
                    a @ b
                torch.cuda.synchronize()
                times.append((time.time() - start) / 10)
            results["matmul_ms"] = sum(times) / len(times) * 1000

            x = torch.randn(1024, 1024, device="cuda")
            start = time.time()
            for _ in range(100):
                x.clone()
            torch.cuda.synchronize()
            results["mem_bw_gb_s"] = (1024 * 1024 * 4 * 2 * 100) / (time.time() - start) / (1024**3)
        return results

    def _score_quant(self, micro: Dict[str, float]) -> List[QuantScore]:
        scores = [
            QuantScore("none", 16, 10.0, 10.0, 2.0, 8.0),
            QuantScore("bitsandbytes", 8, 9.5, 9.0, 5.0, 8.5),
            QuantScore("bitsandbytes", 4, 8.5, 8.0, 8.0, 8.2),
            QuantScore("hqq", 4, 9.0, 7.5, 8.5, 8.5),
            QuantScore("hqq", 8, 9.5, 8.5, 5.5, 8.3),
        ]
        if self.hw.has_cuda and self.hw.gpu_vram_gb < 8:
            for s in scores:
                s.vram *= 1.5
        if self.hw.has_bf16:
            for s in scores:
                s.speed *= 1.1
        for s in scores:
            s.overall = s.quality * 0.3 + s.speed * 0.3 + s.vram * 0.4
        return sorted(scores, key=lambda x: x.overall, reverse=True)

    def get_recommendations(self) -> Dict[str, Any]:
        cache = os.path.join(self.cache_dir, "hardware_profile.json")
        if os.path.exists(cache):
            try:
                with open(cache) as f:
                    return json.load(f)
            except Exception:
                pass
        micro = self._micro_bench()
        quant = self._score_quant(micro)
        bq = quant[0] if quant else None

        formats: List[Dict[str, Any]] = [{"format": "safetensors", "score": 9.0, "reason": "Best HF compatibility"}]
        if self.hw.has_cuda:
            formats.append({"format": "gguf", "score": 8.5, "reason": "Optimized for llama.cpp/Ollama"})
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            formats.append({"format": "mlx", "score": 9.5, "reason": "Optimized for Apple Silicon"})
        formats.append({"format": "onnx", "score": 7.0, "reason": "Cross-platform deployment"})
        formats.sort(key=lambda x: x["score"], reverse=True)

        recs = {
            "hardware": asdict(self.hw),
            "micro": micro,
            "quant_scores": [{"method": q.method, "bits": q.bits, "overall": q.overall} for q in quant],
            "format_scores": formats,
            "recommended_quant": {"method": bq.method if bq else "bitsandbytes", "bits": bq.bits if bq else 4},
            "recommended_format": formats[0]["format"],
            "recommended_gguf_quant": "Q4_K_M",
        }
        with open(cache, "w") as f:
            json.dump(recs, f, indent=2)
        return recs

    def generate_report(self) -> str:
        r = self.get_recommendations()
        h = r["hardware"]
        lines = [
            "=" * 70,
            "  QuantLLM Hardware Auto-Benchmark Report",
            "=" * 70,
            f"\n  CPU: {h['cpu_name']} ({h['cpu_cores']} cores, {h['ram_gb']:.1f} GB RAM)",
        ]
        if h["has_cuda"]:
            lines.append(f"  GPU: {h['gpu_name']} ({h['gpu_vram_gb']:.1f} GB, SM{h['gpu_compute']})")
        else:
            lines.append(f"  GPU: None (CPU mode)")
        lines.append(f"\n  Quantization Rankings:")
        for qs in r["quant_scores"][:5]:
            lines.append(f"    {qs['method'] + ' ' + str(qs['bits']) + 'bit':<25} Score: {qs['overall']:.1f}")
        lines.append(f"\n  Recommended: {r['recommended_quant']['method']} {r['recommended_quant']['bits']}bit")
        lines.append(f"  Format: {r['recommended_format']}")
        lines.append(f"  GGUF: {r['recommended_gguf_quant']}")
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
