# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QuantLLM** is a Python library for ultra-fast LLM quantization, fine-tuning, and deployment. The core philosophy is "one line of code" — load any HuggingFace model with automatic hardware detection, optimal quantization, Flash Attention, and memory configuration.

- **Main API**: `from quantllm import turbo` → `turbo("model-id")` returns a `TurboModel`
- **Key features**: Auto-quantization (BitsAndBytes 4/8-bit, HQQ 2-8-bit), GGUF/ONNX/MLX export, LoRA fine-tuning, OpenAI-compatible inference server, benchmarking, HF Hub push with auto-generated model cards
- **CLI**: `quantllm` command with 10 subcommands (`version`, `info`, `convert`, `finetune`, `export`, `serve`, `bench`, `compare`, `models`, `register`)

## Commands

### Development Setup
```bash
# Install in development mode with all optional dependencies
pip install -e ".[dev]"

# Or install minimal core
pip install -e .
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=quantllm tests/

# Run a single test file
pytest tests/test_cli.py

# Run a single test
pytest tests/test_cli.py::TestCLICommands::test_version_command -v
```

### Linting & Formatting
```bash
# Check linting (ruff - conservative ruleset)
ruff check .

# Check formatting (ruff format)
ruff format --check .

# Auto-fix linting issues
ruff check . --fix

# Auto-format code
ruff format .

# Traditional tools (also configured)
black .
isort .
flake8 .
```

### Type Checking
```bash
mypy quantllm/ --ignore-missing-imports
```

### Building
```bash
# Build distribution
python -m build

# Validate artifacts
twine check dist/*
```

### Running Examples
```bash
# Quick start
python examples/01_quickstart.py

# HQQ quantization
python examples/05_hqq_quantization.py

# Benchmark
python examples/07_benchmark.py

# Full pipeline (load → generate → chat → export → push → serve)
python examples/08_full_pipeline.py
```

## Architecture

### Core Module Structure (`quantllm/`)
```
quantllm/
├── __init__.py           # Public API exports, version, banner
├── cli/                  # CLI entry point (10 subcommands)
├── core/                 # Main logic
│   ├── smart_config.py   # Hardware-aware auto-configuration
│   ├── turbo_model.py    # TurboModel class + turbo() factory
│   ├── model_loader.py   # Model loading with quantization
│   ├── generation.py     # Text generation, chat, streaming
│   ├── finetune_mixin.py # LoRA fine-tuning
│   ├── export_router.py  # Export to GGUF/ONNX/MLX/SafeTensors
│   ├── model_gguf.py     # GGUF-specific export
│   ├── push.py           # HF Hub push with model card
│   ├── architecture.py   # Architecture detection/registration
│   ├── hardware.py       # Hardware profiling (GPU/CPU)
│   ├── model_analyzer.py # Model size/shape analysis
│   ├── flash_attention.py
│   ├── compilation.py    # torch.compile integration
│   ├── memory.py
│   └── training.py
├── quant/                # Quantization backends
│   ├── hqq.py            # Native HQQ (2-8 bit, no calibration)
│   └── llama_cpp.py      # llama-cpp-python GGUF quantization
├── kernels/
│   ├── cpu/              # CPU kernels
│   └── triton/           # Triton fused kernels (GPU)
├── benchmark/
│   ├── runner.py         # Perf benchmarks (throughput, latency, VRAM)
│   ├── compare.py        # Model/method comparison
│   ├── auto_bench.py     # Hardware auto-benchmark
│   └── report.py         # Results formatting
├── server/
│   ├── app.py            # FastAPI OpenAI-compatible server
│   └── engine.py         # Inference engine (transformers/llama.cpp)
├── hub/
│   ├── hub_manager.py    # HF Hub operations
│   └── model_card.py     # Auto-generated model cards
├── registry.py           # Dynamic model registry (QuantLLM HF org)
└── utils/
    ├── logging.py        # stdlib logging (no Rich dependency)
    ├── memory_tracker.py
    └── progress.py
```

### Key Design Patterns

1. **Mixin Composition**: `TurboModel` inherits from 7 mixins (`ArchitectureMixin`, `ModelLoaderMixin`, `ModelGGUFMixin`, `GenerationMixin`, `FinetuneMixin`, `ExportRouterMixin`, `PushToHubMixin`) — each handles a distinct domain.

2. **SmartConfig**: Single source of truth for all auto-detected settings (bits, quant_type, group_size, device, dtype, flash attention, CPU offload, batch sizes). `SmartConfig.detect(model_name, ...)` is the primary entry point.

3. **Quantization Dual-Track**:
   - **BitsAndBytes**: 4-bit/8-bit loading via transformers integration (runtime)
   - **HQQ**: 2-8 bit native quantization (no calibration data, for export/optimization)
   - GGUF export uses `llama-cpp-python` quantization types (Q4_K_M, Q5_K_M, etc.)

4. **Lazy Imports**: Heavy dependencies (torch, transformers, bitsandbytes, triton) are imported inside functions to keep startup fast.

5. **Hardware Abstraction**: `HardwareProfiler.detect()` → `HardwareProfile` with GPU memory, compute capability, bf16/fp16 support, Flash Attention eligibility.

## Configuration Files

- **pyproject.toml**: PEP 621 metadata, dependencies, optional groups (`full`, `server`, `bench`, `gguf`, `onnx`, `mlx`, `triton`, `flash`, `hub`, `dev`), tool configs (ruff, mypy, pytest)
- **.github/workflows/ci.yml**: CI pipeline (lint → typecheck → tests on 3.10/3.11/3.12 → build)
- **.pre-commit-config.yaml**: Local hooks (whitespace, EOF, large files, ruff autofix)

## Testing Notes

- Tests use CPU-only PyTorch (installed via `--index-url https://download.pytorch.org/whl/cpu` in CI)
- Test `QUANTLLM_BANNER=0` to suppress banner
- Test files mirror core modules: `test_hqq.py`, `test_cli.py`, `test_benchmark.py`, `test_server.py`, `test_registry.py`, `test_cpu_kernels.py`, `test_generation.py`, `test_quantization_state.py`, `test_resolve_model_type.py`, `test_export_push_config.py`, `test_architecture_fallback.py`

## Common Tasks

### Adding a New CLI Subcommand
1. Add parser in `quantllm/cli/__init__.py` (`_route()` function)
2. Import and call the core function
3. Add test in `tests/test_cli.py`

### Adding a New Quantization Method
1. Implement in `quantllm/quant/`
2. Export from `quantllm/quant/__init__.py`
3. Add to `TurboModel` via `ExportRouterMixin` or new mixin
4. Update `SmartConfig._choose_quant_type()` if GGUF-related

### Adding a New Export Format
1. Add method to `ExportRouterMixin` (`quantllm/core/export_router.py`)
2. Register in `TurboModel.export()` routing
3. Update CLI `export` subcommand choices

## Important Conventions

- **Version**: `__version__` in `quantllm/__init__.py` (also in `pyproject.toml`)
- **Logging**: Use `quantllm.utils.logging.configure_logging()` — stdlib only, no Rich
- **Banner**: Shown on import in TTY unless `QUANTLLM_BANNER=0`
- **Type hints**: mypy config is permissive (`ignore_missing_imports = true`, no `disallow_untyped_defs`)
- **Line length**: 120 (ruff), not strictly enforced (E501 ignored)