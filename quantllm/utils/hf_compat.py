"""Small shims over transformers API differences across supported versions."""

from functools import lru_cache
from typing import Any, Dict


@lru_cache(maxsize=1)
def _dtype_key() -> str:
    # transformers 4.56 renamed from_pretrained/from_config's ``torch_dtype`` to ``dtype``;
    # 5.x warns on the old name. We still support 4.46+, so pick per installed version.
    try:
        import transformers

        major, minor = (int(p) for p in transformers.__version__.split(".")[:2])
        return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    except Exception:
        return "torch_dtype"


def dtype_kwargs(dtype: Any) -> Dict[str, Any]:
    """Return ``{"dtype": dtype}`` or ``{"torch_dtype": dtype}`` for the installed transformers."""
    return {_dtype_key(): dtype}
