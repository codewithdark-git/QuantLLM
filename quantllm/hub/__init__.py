"""QuantLLM Hub Module - HuggingFace Hub integration."""

from .hub_manager import HubManager
from .checkpoint_manager import CheckpointManager
from .hf_manager import (
    QuantLLMHubManager,
    create_hub_manager,
    is_hf_lifecycle_available,
)

__all__ = [
    # Legacy
    "HubManager",
    "CheckpointManager",
    # New (v2.0 - recommended)
    "QuantLLMHubManager",
    "create_hub_manager",
    "is_hf_lifecycle_available",
]
