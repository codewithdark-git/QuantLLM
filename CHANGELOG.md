# Changelog

All notable changes to QuantLLM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — production hardening on top of v2.1.0rc1

### Fixed

- **`is_quantized` no longer lies about the loaded model state.** The
  attribute is now a derived property reading
  `model.config.quantization_config` (and BitsAndBytes layer types) at
  call time. This fixes three concrete bugs in v2.1.0rc1:
  * `from_config_only=True` previously left `_is_quantized=True` even
    though `AutoModelForCausalLM.from_config(...)` returns a random-
    weights model with no quantization. The flag is now `False` and a
    warning is emitted to make the random-weights nature explicit.
  * A missing `bitsandbytes` install used to silently fall through to
    full precision while keeping `_is_quantized=True`. We now log a
    descriptive warning and report `False`.
  * Pre-quantized HF repos that already shipped a `quantization_config`
    (GPTQ, AWQ, etc.) are now correctly reported as quantized regardless
    of the user's `quantize=False` flag.
- **`DEFAULT_ARCHITECTURE_FALLBACKS` is now actually consulted.** The
  fallback table introduced by PR #27 was dead code whenever HF returned
  a non-empty `model_type` (i.e. always). `resolve_model_type` now
  checks the table directly and recognises common version-suffix
  patterns (`qwen3` → `qwen2`, `llama4` → `llama`, `phi4` → `phi3`,
  `gemma3` → `gemma2`, etc.).
- **`register_architecture` class lookup now uses the natural API.**
  Calling `register_architecture("newmodel", base_model_type="llama",
  model_class=NewModel)` previously stored the class under `"newmodel"`
  but looked it up under `"llama"`, so the fallback path silently
  ignored it. The lookup now tries the original `config.model_type`
  first and falls back to the resolved base family.
- Removed an accidentally duplicated `if is_bnb and is_8bit ...` block
  in the existing-quant detection branch of
  `TurboModel.from_pretrained`.

### Added

- **`TurboModel.is_quantized` public property** plus
  **`TurboModel.report()`** returning a structured dict (`model_id`,
  `params_billion`, `requested_bits`, `effective_loading_bits`,
  `is_quantized`, `quant_method`, `device`, `dtype`, `finetuned`,
  `lora_applied`). Use `report()` to assert programmatically what the
  loader actually produced.
- **Pre-quantized repo detection.** Repository names matching
  `*-bnb-4bit`, `*-bnb-8bit`, `*-AWQ`, `*-GPTQ`, `*-INT4`, `*-INT8`,
  `*-FP8`, `*-EETQ`, `*-HQQ`, `*-AQLM` log a friendly hint that the
  embedded `quantization_config` will be honoured rather than
  re-quantized.
- **GGUF-only repo hint.** When a name contains `-gguf` / `.gguf`,
  `from_pretrained` warns and points the user at `from_gguf`.
- **Expanded `DEFAULT_ARCHITECTURE_FALLBACKS` table** covering Llama 2/3/4,
  Mistral / Mixtral, Qwen 2 / 2-MoE / 3, Phi / Phi-3 / Phi-4, Gemma /
  Gemma 2 / Gemma 3, Falcon, Cohere / Command-R, DeepSeek (V2/V3),
  OLMo / OLMo 2, SmolLM / SmolLM 2 / SmolLM 3, Yi, StarCoder /
  StarCoder 2, InternLM / InternLM 2, Baichuan, ChatGLM and StableLM.
- **Real CI workflow** at `.github/workflows/ci.yml` running ruff,
  pytest on Python 3.10 / 3.11 / 3.12, and `python -m build` +
  `twine check` on every PR.
- **`pyproject.toml`** providing PEP 517 / 518 build metadata, a
  conservative ruff lint profile and pytest defaults.
- **`.pre-commit-config.yaml`** for local enforcement (whitespace,
  end-of-file fixer, large-file guard, ruff with autofix).
- **`docs/guide/consumer-hardware.md`** documenting expected behaviour
  on every tier of consumer hardware (CPU-only, ≤ 8 GB VRAM,
  12 – 24 GB, Apple Silicon, multi-GPU) and how to inspect the loaded
  state.
- **Regression tests** for every fix above:
  * `tests/test_quantization_state.py` — runtime quantization state
    tracking, `from_config_only` honesty, `report()` schema.
  * `tests/test_resolve_model_type.py` — fallback table consultation,
    family-suffix matching, registry-class lookup ergonomics.

### Changed

- `TurboModel.__repr__` now reads from the new `is_quantized` property
  and degrades gracefully when `num_parameters()` is unavailable
  (mocked / lazily-loaded models).
- `TurboModel.from_gguf` now sets `_is_quantized_override = True`
  rather than mutating an attribute the type system thought was a
  property -- this is functionally identical but more honest about the
  contract.
- The "bitsandbytes not installed" warning now explains how to install
  it and explicitly states that loading falls back to full precision.

## [2.0.0] — 2025-12-21

Initial public release of the `turbo()` API and the GGUF / ONNX / MLX
export pipeline. See the GitHub
[releases page](https://github.com/codewithdark-git/QuantLLM/releases/tag/v2.0.0)
for the full notes.
