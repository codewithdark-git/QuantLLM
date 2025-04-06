class TrainingLogger:
    def __init__(self):
        """Initialize training logger."""
        pass

    def log_info(self, message: str):
        """Log info message."""
        print(f"INFO: {message}")

    def log_warning(self, message: str):
        """Log warning message."""
        print(f"WARNING: {message}")

    def log_error(self, message: str):
        """Log error message."""
        print(f"ERROR: {message}")

    def log_metrics(self, metrics: dict, step: int = None):
        """Log training metrics."""
        step_str = f" (Step {step})" if step is not None else ""
        print(f"\nMetrics{step_str}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")