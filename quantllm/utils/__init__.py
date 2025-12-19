"""
QuantLLM Utilities

Provides logging, progress tracking, and memory management utilities.
Consistent orange theme across the entire project.
"""

from .progress import (
    # Theme colors
    QUANTLLM_ORANGE,
    QUANTLLM_ORANGE_LIGHT,
    QUANTLLM_ACCENT,
    
    # Logging
    configure_logging,
    get_logger,
    logger,
    console,
    
    # Progress tracking
    QuantLLMProgress,
    StepProgress,
    track_progress,
    stream_subprocess_output,
    
    # Output helpers
    print_header,
    print_subheader,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_step,
    print_table,
    print_model_card,
    print_banner,
    
    # Formatters
    format_size,
    format_time,
)

from .memory_tracker import MemoryTracker

# Standardize enable_logging alias
def enable_logging(level="INFO"):
    configure_logging(level)

__all__ = [
    # Theme
    "QUANTLLM_ORANGE",
    "QUANTLLM_ORANGE_LIGHT", 
    "QUANTLLM_ACCENT",
    
    # Logging
    "configure_logging",
    "enable_logging",
    "get_logger",
    "logger",
    "console",
    
    # Progress tracking
    "QuantLLMProgress",
    "StepProgress",
    "track_progress",
    "stream_subprocess_output",
    
    # Output helpers
    "print_header",
    "print_subheader",
    "print_success",
    "print_warning",
    "print_error",
    "print_info",
    "print_step",
    "print_table",
    "print_model_card",
    "print_banner",
    
    # Formatters
    "format_size",
    "format_time",
    
    # Memory
    "MemoryTracker",
]