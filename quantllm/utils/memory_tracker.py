import torch
import psutil
import time
import logging
from typing import Dict, Optional, Union, Tuple, Any

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MemoryTracker:
    """
    A class to track CPU and GPU memory usage at specified checkpoints.
    Focuses on GPU memory using `torch.cuda` and CPU memory using `psutil`.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None, logger: Optional[logging.Logger] = None):
        """
        Initializes the MemoryTracker.

        Args:
            device (Optional[Union[str, torch.device]]): The primary device to track GPU memory for.
                                                         If None, defaults to `cuda:0` if available, else `cpu`.
            logger (Optional[logging.Logger]): An external logger to use. If None, uses a default logger.
        """
        self.logger = logger if logger else logging.getLogger(__name__)
        
        if device:
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")

        self.process = psutil.Process()
        self.memory_logs: Dict[str, Dict[str, Any]] = {}
        self.tracking_started: bool = False
        self.start_time: Optional[float] = None

        self.logger.info(f"MemoryTracker initialized. Tracking on device: {self.device}.")
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            self.logger.warning(f"Device is set to {self.device} but CUDA is not available. GPU tracking will be disabled.")
            self.device = torch.device('cpu') # Fallback to CPU

    def _get_cpu_memory_gb(self) -> Dict[str, float]:
        """Returns current CPU memory usage by the current process in GB."""
        mem_info = self.process.memory_info()
        return {
            "cpu_rss_gb": mem_info.rss / (1024 ** 3),  # Resident Set Size
            "cpu_vms_gb": mem_info.vms / (1024 ** 3)   # Virtual Memory Size
        }

    def _get_gpu_memory_gb(self) -> Optional[Dict[str, float]]:
        """
        Returns current and peak GPU memory usage on the specified CUDA device in GB.
        Returns None if the device is not CUDA or CUDA is not available.
        """
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(self.device) # Ensure accuracy of memory readings
            return {
                "gpu_allocated_current_gb": torch.cuda.memory_allocated(self.device) / (1024 ** 3),
                "gpu_reserved_current_gb": torch.cuda.memory_reserved(self.device) / (1024 ** 3),
                "gpu_allocated_peak_gb": torch.cuda.max_memory_allocated(self.device) / (1024 ** 3),
                "gpu_reserved_peak_gb": torch.cuda.max_memory_reserved(self.device) / (1024 ** 3),
            }
        return None

    def start_tracking(self):
        """
        Starts memory tracking. Resets previous logs and CUDA peak memory statistics.
        Logs an initial memory checkpoint.
        """
        self.logger.info("Starting memory tracking...")
        self.memory_logs = {} # Clear previous logs
        self.tracking_started = True
        self.start_time = time.perf_counter()

        if self.device.type == 'cuda' and torch.cuda.is_available():
            # Reset peak memory stats for the specified device.
            # This ensures peak values are relative to this tracking session.
            torch.cuda.reset_peak_memory_stats(self.device)
            self.logger.info(f"CUDA peak memory statistics reset for device {self.device}.")
        
        self.log_memory("initial_state")
        self.logger.info("Memory tracking started and initial state logged.")

    def log_memory(self, checkpoint_name: str):
        """
        Logs current CPU and GPU memory usage at a given checkpoint.

        Args:
            checkpoint_name (str): A descriptive name for the checkpoint (e.g., "before_quantization").
        """
        if not self.tracking_started:
            self.logger.warning("Tracking not started. Call start_tracking() first.")
            return

        if checkpoint_name in self.memory_logs:
            self.logger.warning(f"Checkpoint '{checkpoint_name}' already exists. It will be overwritten.")

        current_time_offset = time.perf_counter() - (self.start_time if self.start_time is not None else 0)
        
        log_entry: Dict[str, Any] = {
            "timestamp_s": current_time_offset, # Clarified unit in key name
            "cpu_memory": self._get_cpu_memory_gb()
        }
        
        gpu_mem_info = self._get_gpu_memory_gb()
        if gpu_mem_info:
            log_entry["gpu_memory"] = gpu_mem_info
        else:
            log_entry["gpu_memory"] = "N/A (CUDA not available or device is CPU)"

        self.memory_logs[checkpoint_name] = log_entry
        
        # Formatted log message
        cpu_rss = log_entry['cpu_memory']['cpu_rss_gb']
        cpu_vms = log_entry['cpu_memory']['cpu_vms_gb']
        log_summary = f"Checkpoint '{checkpoint_name}': CPU RSS: {cpu_rss:.3f} GB, CPU VMS: {cpu_vms:.3f} GB"
        
        if gpu_mem_info:
            gpu_alloc_curr = gpu_mem_info['gpu_allocated_current_gb']
            gpu_alloc_peak = gpu_mem_info['gpu_allocated_peak_gb']
            log_summary += f" | GPU Allocated (Curr/Peak): {gpu_alloc_curr:.3f}/{gpu_alloc_peak:.3f} GB"
        
        self.logger.info(log_summary)

    def stop_tracking(self):
        """
        Stops memory tracking and logs a final memory state.
        """
        if not self.tracking_started:
            self.logger.info("Tracking was not started or already stopped.")
            return
        
        self.log_memory("final_state")
        self.tracking_started = False
        elapsed_time = time.perf_counter() - (self.start_time if self.start_time else 0)
        self.logger.info(f"Memory tracking stopped. Total duration: {elapsed_time:.2f} seconds.")

    def get_report(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing all logged memory checkpoints.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are checkpoint names.
        """
        if not self.memory_logs:
            self.logger.info("No memory logs recorded.")
            return {}
        return self.memory_logs

    def print_report(self):
        """
        Prints a formatted summary of all logged memory checkpoints.
        """
        report_data = self.get_report()
        if not report_data:
            self.logger.info("Memory Tracker Report: No data logged in this session.") # Use logger
            return

        print("\n===== Memory Tracker Report =====")
        print(f"Tracked Device: {self.device}") # Added device info to report
        for checkpoint_name, data in report_data.items():
            ts = data.get('timestamp_s', 0.0) # Use updated key name
            cpu_mem = data.get('cpu_memory', {})
            gpu_mem_data = data.get('gpu_memory') # Can be dict or string "N/A..."

            print(f"\n--- Checkpoint: {checkpoint_name} (Time: {ts:.2f}s) ---")
            print(f"  CPU Memory:")
            print(f"  CPU Memory:")
            print(f"    - RSS (Current): {cpu_mem.get('cpu_rss_gb', 0):.3f} GB")
            print(f"    - VMS (Current): {cpu_mem.get('cpu_vms_gb', 0):.3f} GB")

            if isinstance(gpu_mem_data, dict): # GPU data is available
                print(f"  GPU Memory ({self.device}):") # Clarify which device if multiple GPUs
                print(f"    - Allocated (Current): {gpu_mem_data.get('gpu_allocated_current_gb', 0):.3f} GB")
                print(f"    - Reserved (Current):  {gpu_mem_data.get('gpu_reserved_current_gb', 0):.3f} GB")
                # Clarify that peak is for the current tracking session
                print(f"    - Allocated (Peak Session): {gpu_mem_data.get('gpu_allocated_peak_gb', 0):.3f} GB")
                print(f"    - Reserved (Peak Session):  {gpu_mem_data.get('gpu_reserved_peak_gb', 0):.3f} GB")
            else: # GPU data is N/A (e.g. string message)
                print(f"  GPU Memory: {gpu_mem_data}")
        print("=" * 31 + "\n") # Adjusted separator length

    def reset(self):
        """
        Resets the tracker, clearing all logs from the current session and restarting the tracking process.
        This is effectively an alias for start_tracking() in the current implementation,
        as start_tracking() already clears previous logs for the new session.
        """
        self.logger.info("Resetting MemoryTracker.")
        self.start_tracking()

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    tracker = MemoryTracker() # Auto-detects device
    
    tracker.start_tracking() # Logs 'initial_state'
    
    # Simulate some work
    time.sleep(0.5)
    # Example: allocate some CPU list
    # pylint: disable=unused-variable
    temp_list_cpu = [i for i in range(1000000)] # Allocate some CPU memory
    tracker.log_memory("after_cpu_list_allocation")
    del temp_list_cpu
    tracker.log_memory("after_cpu_list_del")

    if tracker.device.type == 'cuda' and torch.cuda.is_available():
        try:
            print(f"--- Simulating GPU work on {tracker.device} ---")
            # pylint: disable=unused-variable
            initial_tensor = torch.zeros(1, device=tracker.device) # Ensure CUDA context
            tracker.log_memory("gpu_context_established")

            tensor_a = torch.randn(1024, 1024, 50, device=tracker.device) # Approx 200MB
            tracker.log_memory("after_gpu_tensor_A_allocation")
            
            tensor_b = torch.randn(1024, 1024, 100, device=tracker.device) # Approx 400MB
            tracker.log_memory("after_gpu_tensor_B_allocation") # Peak should be around A+B
            
            del tensor_a # Free tensor_a
            torch.cuda.empty_cache() # Clear cache to see effect on allocated memory
            tracker.log_memory("after_gpu_tensor_A_freed_and_cache_empty")
            
            del tensor_b # Free tensor_b
            # Peak stats are NOT reset by empty_cache, so peak should remain.
            # Allocated should be low now.
            torch.cuda.empty_cache()
            tracker.log_memory("after_all_gpu_tensors_freed_and_cache_empty")
        except RuntimeError as e: # Catch specific CUDA errors like OOM
            tracker.logger.error(f"Error during GPU example: {e}")
    
    tracker.stop_tracking() # Logs 'final_state'
    
    tracker.print_report()

    # Example 2: Explicitly CPU
    print("\n--- Example 2: Explicit CPU tracking ---")
    cpu_tracker = MemoryTracker(device='cpu')
    cpu_tracker.start_tracking()
    # pylint: disable=unused-variable
    another_cpu_list = [i * 2 for i in range(750000)]
    cpu_tracker.log_memory("cpu_only_checkpoint_1")
    del another_cpu_list
    cpu_tracker.log_memory("cpu_only_checkpoint_2_after_del")
    cpu_tracker.stop_tracking()
    cpu_tracker.print_report()

    # Example 3: Resetting a tracker
    print("\n--- Example 3: Resetting a tracker ---")
    tracker.reset() # Starts a new session, 'initial_state' is logged
    # pylint: disable=unused-variable
    short_list = [1] * 1000
    tracker.log_memory("after_reset_and_small_alloc")
    del short_list
    tracker.stop_tracking()
    tracker.print_report()
