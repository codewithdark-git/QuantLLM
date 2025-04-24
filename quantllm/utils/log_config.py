import logging
import warnings
from typing import List, Optional

def configure_logging(
    disable_loggers: Optional[List[str]] = None,
    level: int = logging.WARNING
):
    """
    Configure logging for the entire package.
    
    Args:
        disable_loggers: List of logger names to disable/set to WARNING
        level: Logging level for disabled loggers
    """
    # Default loggers to disable
    default_disable = [
        "transformers",
        "datasets",
        "tokenizers",
        "huggingface_hub",
        "accelerate",
        "torch.distributed",
        "filelock",
        "fsspec",
        "numpy",
        "numexpr"
    ]
    
    disable_loggers = disable_loggers or default_disable
    
    # Set logging level for specified loggers
    for logger_name in disable_loggers:
        logging.getLogger(logger_name).setLevel(level)
    
    # Disable specific warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=".*TimeDeltaIndex.base.*")
    
def enable_logging():
    """Re-enable all logging."""
    logging.getLogger().setLevel(logging.INFO)
    warnings.resetwarnings()