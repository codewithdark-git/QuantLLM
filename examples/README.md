# QuantLLM v2.2 Examples

These examples are **runnable** -- each one calls real package APIs
and prints live results.  No hardcoded data or fakes.

## Examples

| # | File | What it does |
|---|------|-------------|
| 01 | `01_quickstart.py` | Load model, generate, chat, stream, export |
| 02 | `02_gguf_export.py` | Export at multiple GGUF quant levels |
| 03 | `03_finetuning.py` | LoRA fine-tuning on sample data |
| 04 | `04_hub_push.py` | Push to HF Hub (requires `HF_TOKEN`) |
| 05 | `05_hqq_quantization.py` | HQQ 2-8 bit quantization on layers/models |
| 06 | `06_inference_server.py` | Start OpenAI-compatible server (blocking) |
| 07 | `07_benchmark.py` | Live benchmark with BenchmarkRunner + comparer |
| 08 | `08_full_pipeline.py` | Load -> Generate -> Chat -> Export -> Push -> Serve |

## Run

```bash
# Basic model loading and generation
python examples/01_quickstart.py

# HQQ quantisation (no external model needed)
python examples/05_hqq_quantization.py

# Full pipeline
python examples/08_full_pipeline.py
```

## Requirements

```bash
pip install quantllm torch transformers
pip install quantllm[server]   # for example 06
pip install quantllm[full]     # for all features
```
