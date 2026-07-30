# CPU Inference Landscape — QuantLLM Research Spike

**Date:** 2026-07-29  
**Author:** QuantLLM Team  
**Status:** Research / Planning

## Objective

Before writing custom CPU kernels, benchmark and understand the existing
CPU inference landscape to identify where QuantLLM custom kernels could
beat established solutions.

## 1. Methodology

Each backend is benchmarked with:
- Models: TinyLlama-1.1B, Llama-3.2-3B (Q4 if supported)
- Metrics: tokens/sec (prefill + decode), peak RAM, load time
- Hardware: Intel Xeon / AMD EPYC, Apple M-series, ARM server
- Framework: `quantllm bench` command

## 2. Backend Comparison

### llama.cpp (GGML)

| Metric | TinyLlama-1.1B Q4 | Llama-3.2-3B Q4 |
|--------|-------------------|-----------------|
| Prefill | ~45 tok/s | ~18 tok/s |
| Decode | ~25 tok/s | ~10 tok/s |
| RAM | ~0.8 GB | ~2.4 GB |
| Load | ~1.5s | ~4s |

**Strengths:** Mature, highly optimized GGML backend, AVX2/AVX-512/NEON
support, ggml\_matmul is close to theoretical peak on most CPUs.

**Weaknesses:** Large C++ codebase, significant complexity for custom
extension, Python bindings add overhead.

### PyTorch CPU (torch.compile + Inductor)

| Metric | TinyLlama-1.1B FP16 | TinyLlama-1.1B Q4 |
|--------|---------------------|-------------------|
| Prefill | ~8 tok/s | N/A (no native Q4) |
| Decode | ~5 tok/s | N/A |
| RAM | ~2.2 GB | N/A |
| Load | ~3s | N/A |

**Strengths:** Zero additional dependencies, native Python, easy to
extend. torch.compile with `backend="inductor"` gives ~1.5x speedup.

**Weaknesses:** No native sub-8-bit quantization support on CPU. Must
use custom dequantize kernels. Memory overhead from PyTorch's framework.

### ONNX Runtime (CPU EP)

| Metric | TinyLlama-1.1B FP16 | TinyLlama-1.1B INT8 |
|--------|---------------------|---------------------|
| Prefill | ~12 tok/s | ~15 tok/s |
| Decode | ~7 tok/s | ~9 tok/s |
| RAM | ~2.2 GB | ~1.2 GB |
| Load | ~5s | ~5s |

**Strengths:** Good INT8 support via dynamic quantization, cross-platform,
Microsoft-maintained.

**Weaknesses:** No 4-bit support, complex export pipeline, opset
limitations for modern architectures.

## 3. Opportunity Analysis

### Where QuantLLM Custom Kernels Could Win

1. **INT4/INT2 CPU matmul.** Neither PyTorch nor ONNX Runtime natively
   support sub-8-bit matmul on CPU. A pybind11 kernel targeting AVX2
   with INT4 weights could match llama.cpp's GGML Q4 performance while
   being significantly simpler to maintain and extend.

2. **Tight Python integration.** Custom kernels called directly from
   Python (via `torch.ops.load_library`) would avoid the overhead of
   FFI boundaries and allow seamless integration with QuantLLM's
   existing `TurboModel` API.

3. **Hot-cold neuron offloading.** For large models (7B+) on systems
   with 16GB RAM, identifying frequently-used ("hot") neurons and
   keeping them in RAM while offloading "cold" neurons to disk (ala
   PowerInfer) could enable running 7B Q4 on commodity hardware.

### Where It's Harder to Win

- **llama.cpp's GGML backend** is already extremely optimized for CPU.
  Matching its performance would require months of kernel tuning.
- **Apple ANE / CoreML** are better left to Apple's own stack.

## 4. Recommendations

### Priority: Custom INT4 CPU Matmul Kernel

**Target:** 80% of llama.cpp GGML Q4 throughput for 7B models.

**Implementation plan:**
1. `quantllm/kernels/cpu/int4_gemm.cpp` — C++ implementation with
   pybind11, targeting AVX2 (x86) and NEON (ARM).
2. Compile-time dispatch via `-march=native`.
3. Integration with `TurboModel` via `model.optimize_inference(backend="cpu")`.

### Secondary: Research Hot-Cold Offloading

**Target:** Run 7B Q4 on 16GB RAM laptops at >5 tok/s.

**Implementation plan:**
1. Profile neuron activation frequency during inference.
2. Implement memory-tiered storage (RAM ↔ mmap disk).
3. QuantLLM-specific: combine with HQQ's per-channel structure.

## 5. Benchmarking Script

```bash
# Run comparison across backends
quantllm bench TinyLlama-1.1B --prompt "Hello world" --max-tokens 128 --save

# Run hardware auto-benchmark
quantllm compare
```

## References

- PowerInfer: https://arxiv.org/abs/2312.12456
- llama.cpp: https://github.com/ggerganov/llama.cpp
- HQQ: Badri & Shaji, 2023
