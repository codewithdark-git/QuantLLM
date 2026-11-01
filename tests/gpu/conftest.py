"""Fixtures for the GPU end-to-end suite.

Every test under ``tests/gpu`` needs a CUDA device and downloads a real model,
so the whole directory is skipped on CPU-only machines. Run it locally on a GPU
box with ``pytest tests/gpu -m gpu`` or remotely with
``modal run scripts/modal_gpu_tests.py``.
"""

import gc
import os

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available; GPU suite skipped", allow_module_level=True)

os.environ.setdefault("QUANTLLM_BANNER", "0")

# Small, ungated, chat-tuned model with a chat template. Override to test others.
GPU_TEST_MODEL = os.environ.get("QUANTLLM_GPU_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/gpu" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.gpu)


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def model_id():
    return GPU_TEST_MODEL


def _load(model_id, **kwargs):
    from quantllm import turbo

    return turbo(model_id, verbose=False, **kwargs)


@pytest.fixture(scope="module")
def model_4bit(model_id):
    m = _load(model_id, bits=4)
    yield m
    del m
    free_cuda()


@pytest.fixture(scope="module")
def model_16bit(model_id):
    m = _load(model_id, bits=16)
    yield m
    del m
    free_cuda()


@pytest.fixture
def load_model(model_id):
    """Factory for tests that need a fresh model they can mutate (finetune/export)."""
    loaded = []

    def _factory(**kwargs):
        m = _load(model_id, **kwargs)
        loaded.append(m)
        return m

    yield _factory
    loaded.clear()
    free_cuda()
