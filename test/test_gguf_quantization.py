import unittest
import torch
import gc
import os
import tempfile
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from quantllm.quant import GGUFQuantizer
from quantllm.quantization_engine import QuantizedLinear
from quantllm.utils.benchmark import QuantizationBenchmark
from quantllm.utils.logger import logger

# Define model names for testing
TEST_MODEL_NAME_SMALL = "facebook/opt-125m" # Small model for quick unit tests
TEST_MODEL_NAME_MEDIUM = "facebook/opt-350m" # Slightly larger model for integration tests

def _get_dummy_calibration_data(vocab_size: int = 32000, seq_len: int = 32, batch_size: int = 4):
    """Helper function to generate dummy calibration data."""
    return torch.randint(0, vocab_size, (batch_size, seq_len))

def _load_model_and_tokenizer(model_name, trust_remote_code=True):
    """Helper to load a Hugging Face model and tokenizer."""
    # Load model on CPU initially to manage memory before explicit test-specific placement.
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code).cpu()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None: # Common practice for models without a PAD token
        tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer # Ensure model is in evaluation mode

class TestGGUFQuantizerConfigs(unittest.TestCase):
    """
    Tests GGUFQuantizer with various configurations using a small model.
    Focuses on:
    - Correct layer replacement (nn.Linear -> QuantizedLinear).
    - Correctness of quantization parameters in modules and model config.
    - Basic inference functionality of the quantized model.
    - GGUF file conversion and optional ctransformers load check.
    """
    @classmethod
    def setUpClass(cls):
        """Load the small model and calibration data once for all config tests."""
        cls.original_model, cls.tokenizer = _load_model_and_tokenizer(TEST_MODEL_NAME_SMALL)
        cls.calibration_data = _get_dummy_calibration_data(vocab_size=cls.original_model.config.vocab_size)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[TestGGUFQuantizerConfigs] Using device: {cls.device} for GGUF quantizer config tests.")

    def _run_quantization_test(self, quantizer_config: Dict, model_instance: PreTrainedModel):
        """
        Helper method to run a quantization test for a given configuration and model instance.

        Args:
            quantizer_config (Dict): Dictionary of arguments for GGUFQuantizer.
            model_instance (PreTrainedModel): A fresh copy of the model to be quantized.
        """
        test_name = unittest.TestCase.id(self).split('.')[-1] # Gets current test method name
        print(f"\nRunning _run_quantization_test for: {test_name} with config: {quantizer_config}")

        # Ensure the passed model_instance is on CPU before quantizer potentially moves it.
        model_instance = model_instance.cpu()

        # GGUFQuantizer's `device` argument is for its internal DeviceManager,
        # which determines where quantization ops and the final model might reside.
        # The `model_name` param (which is `model_instance` here) will be moved by the quantizer.
        gguf_quantizer = GGUFQuantizer(model_name=model_instance, **quantizer_config, device=self.device)
        
        # Quantize the model
        quantized_model = gguf_quantizer.quantize(calibration_data=self.calibration_data.clone()) # Pass a clone of calib data
        self.assertIsNotNone(quantized_model)

        # Check if linear layers are replaced
        replaced_layers = 0
        for _, module in quantized_model.named_modules():
            if isinstance(module, QuantizedLinear):
                self.assertEqual(module.config.format, "gguf")
                self.assertEqual(module.config.bits, quantizer_config["bits"])
                if quantizer_config.get("group_size", -1) > 0 and quantizer_config.get("group_size") < module.in_features :
                     self.assertTrue(module.config.channel_wise)
                     self.assertEqual(module.config.format_config.get("group_size"), quantizer_config["group_size"])
                replaced_layers += 1
        self.assertGreater(replaced_layers, 0, "No layers were replaced with QuantizedLinear")

        # Check model config update
        self.assertIn("quantization_config", quantized_model.config.to_dict())
        q_config = quantized_model.config.quantization_config
        self.assertEqual(q_config["format"], "gguf")
        self.assertEqual(q_config["bits"], quantizer_config["bits"])
        self.assertEqual(q_config["format_config"]["group_size"], quantizer_config.get("group_size", -1)) # GGUFQuantizer stores it here
        self.assertEqual(q_config["format_config"]["use_packed"], quantizer_config.get("use_packed", True))


        # Basic inference check
        dummy_input_ids = self.tokenizer("Hello world", return_tensors="pt").input_ids.to(self.device)
        # Ensure quantized_model is on the correct device for inference
        quantized_model.to(self.device)
        try:
            with torch.no_grad():
                outputs = quantized_model(dummy_input_ids)
            self.assertIsNotNone(outputs.logits)
            self.assertEqual(outputs.logits.shape[0], dummy_input_ids.shape[0]) # Batch size
            self.assertEqual(outputs.logits.shape[1], dummy_input_ids.shape[1]) # Seq len
            self.assertFalse(torch.isnan(outputs.logits).any(), "Logits contain NaNs")
            self.assertFalse(torch.isinf(outputs.logits).any(), "Logits contain Infs")
        except Exception as e:
            self.fail(f"Inference failed for config {quantizer_config}: {e}")
        
        # Test GGUF conversion (file existence and basic structure)
        # Use delete=False and manual cleanup for more robust handling with external processes if any.
        gguf_file_path = "" # Initialize to ensure it's defined in finally
        try:
            with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmpfile:
                gguf_file_path = tmpfile.name
            
            gguf_quantizer.convert_to_gguf(output_path=gguf_file_path)
            self.assertTrue(os.path.exists(gguf_file_path), f"GGUF file not found at {gguf_file_path}")
            self.assertGreater(os.path.getsize(gguf_file_path), 0, "GGUF file is empty.")
            
            # Optional: Try to load with ctransformers if available
            if GGUFQuantizer.CT_AVAILABLE: # Check if ctransformers was successfully imported by GGUFQuantizer
                try:
                    from ctransformers import AutoModelForCausalLM as CTAutoModel
                    # Determine model_type for ctransformers; this is a heuristic.
                    # For OPT models, 'gpt2' is often a compatible type in ctransformers.
                    ct_model_type = 'gpt2' 
                    if "llama" in model_instance.config.model_type.lower(): ct_model_type = "llama"
                    # Add other heuristics as needed for different model architectures.

                    ct_model = CTAutoModel.from_pretrained(gguf_file_path, model_type=ct_model_type)
                    self.assertIsNotNone(ct_model, "Failed to load GGUF model with ctransformers.")
                    del ct_model # Clean up ctransformers model
                except Exception as e:
                    # Non-fatal if ctransformers load fails, as it's a complex dependency and test setup.
                    # Log it as a warning or note for test results.
                    print(f"WARNING: ctransformers loading of GGUF for config {quantizer_config} raised: {e}")
        except Exception as e:
            self.fail(f"convert_to_gguf or subsequent check failed for config {quantizer_config}: {e}")
        finally:
            if os.path.exists(gguf_file_path):
                os.remove(gguf_file_path) # Ensure cleanup of the temp file

        # Cleanup PyTorch model and quantizer
        del quantized_model, gguf_quantizer, model_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def test_gguf_b4_gs32(self):
        """Test GGUF with 4 bits, group_size 32."""
        print("\nTesting GGUF: bits=4, group_size=32")
        model_copy = AutoModelForCausalLM.from_config(self.original_model.config).cpu()
        model_copy.load_state_dict(self.original_model.state_dict())
        model_copy.eval()
        self._run_quantization_test(
            {"bits": 4, "group_size": 32, "use_packed": True, "desc_act": False}, 
            model_copy
        )

    def test_gguf_b8_gs128(self):
        """Test GGUF with 8 bits, group_size 128."""
        print("\nTesting GGUF: bits=8, group_size=128")
        model_copy = AutoModelForCausalLM.from_config(self.original_model.config).cpu()
        model_copy.load_state_dict(self.original_model.state_dict())
        model_copy.eval()
        self._run_quantization_test(
            {"bits": 8, "group_size": 128, "use_packed": True, "desc_act": False}, 
            model_copy
        )

    def test_gguf_b4_per_tensor(self):
        """Test GGUF with 4 bits, per-tensor quantization (group_size = -1)."""
        print("\nTesting GGUF: bits=4, group_size=-1 (per-tensor)")
        model_copy = AutoModelForCausalLM.from_config(self.original_model.config).cpu()
        model_copy.load_state_dict(self.original_model.state_dict())
        model_copy.eval()
        self._run_quantization_test(
            {"bits": 4, "group_size": -1, "use_packed": True, "desc_act": False}, 
            model_copy
        )
    
    def test_gguf_b4_gs32_cpu_offload(self):
        """Test GGUF with 4 bits, group_size 32, and CPU offload enabled."""
        print("\nTesting GGUF: bits=4, group_size=32, cpu_offload=True")
        model_copy = AutoModelForCausalLM.from_config(self.original_model.config).cpu()
        model_copy.load_state_dict(self.original_model.state_dict())
        model_copy.eval()
        self._run_quantization_test(
            {"bits": 4, "group_size": 32, "use_packed": True, "desc_act": False, "cpu_offload": True}, 
            model_copy
        )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping GGUF integration tests that benefit from GPU.")
class TestGGUFIntegrationWithModels(unittest.TestCase):
    """
    Tests GGUFQuantizer integration with different models.
    These tests are generally more resource-intensive.
    """
    @classmethod
    def setUpClass(cls):
        """Load models once for all integration tests in this class."""
        cls.calibration_data = _get_dummy_calibration_data() # Generate dummy calibration data
        cls.device = torch.device("cuda") # These tests are intended for CUDA environments
        print(f"\n[TestGGUFIntegrationWithModels] Using device: {cls.device} for GGUF integration tests.")

        cls.models_to_test_config = {} # Store config, not the full model, to save memory
        
        # Load and store config for the small model
        try:
            small_model, _ = _load_model_and_tokenizer(TEST_MODEL_NAME_SMALL)
            cls.models_to_test_config[TEST_MODEL_NAME_SMALL] = small_model.config
            del small_model # Free model memory
            gc.collect()
        except Exception as e:
            print(f"Could not load {TEST_MODEL_NAME_SMALL} for integration tests, it will be skipped. Error: {e}")

        # Optionally add and attempt to load config for a medium model
        try:
            medium_model, _ = _load_model_and_tokenizer(TEST_MODEL_NAME_MEDIUM)
            cls.models_to_test_config[TEST_MODEL_NAME_MEDIUM] = medium_model.config
            del medium_model
            gc.collect()
        except Exception as e:
            print(f"Could not load {TEST_MODEL_NAME_MEDIUM} for integration tests, it will be skipped. Error: {e}")

    def _run_integration_test(self, model_name: str, model_config: AutoConfig, quant_config: Dict):
        """
        Helper to run an integration test for a given model and quantization configuration.
        Loads the model fresh for each call based on its config.
        """
        print(f"\nRunning GGUF integration test for {model_name} with config: {quant_config}")
        
        # Load the model from config for this specific test run
        model_to_quantize = AutoModelForCausalLM.from_config(model_config).cpu()
        # No state_dict loading needed if from_config is sufficient for a fresh model.
        # If pre-trained weights are essential, load them:
        # model_to_quantize = AutoModelForCausalLM.from_pretrained(model_name).cpu()
        model_to_quantize.eval()


        quantizer = GGUFQuantizer(model_name=model_to_quantize, **quant_config, device=self.device)
        quantized_model = quantizer.quantize(calibration_data=self.calibration_data.clone())
        self.assertIsNotNone(quantized_model)
        
        # Basic inference check
        tokenizer = AutoTokenizer.from_pretrained(model_name) # Re-load tokenizer if needed
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        dummy_input_ids = tokenizer("Test sentence for GGUF model.", return_tensors="pt").input_ids.to(self.device)
        quantized_model.to(self.device) # Ensure model is on device

        try:
            with torch.no_grad():
                outputs = quantized_model(dummy_input_ids)
            self.assertIsNotNone(outputs.logits)
            self.assertFalse(torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any())
        except Exception as e:
            self.fail(f"Inference failed for {model_name} with config {quant_config}: {e}")

        # GGUF conversion and potential load check
        gguf_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmpfile:
                gguf_file_path = tmpfile.name
            
            quantizer.convert_to_gguf(output_path=gguf_file_path)
            self.assertTrue(os.path.exists(gguf_file_path) and os.path.getsize(gguf_file_path) > 0)

            if GGUFQuantizer.CT_AVAILABLE:
                try:
                    from ctransformers import AutoModelForCausalLM as CTAutoModel
                    ct_model_type = 'gpt2' # Default for OPT, adjust if testing other architectures
                    if "llama" in model_name.lower(): ct_model_type = "llama"
                    elif "mistral" in model_name.lower(): ct_model_type = "mistral"
                    
                    ct_model = CTAutoModel.from_pretrained(gguf_file_path, model_type=ct_model_type)
                    self.assertIsNotNone(ct_model)
                    del ct_model
                except Exception as e:
                     print(f"WARNING: ctransformers loading of GGUF for {model_name} (config {quant_config}) raised: {e}")
        except Exception as e:
            self.fail(f"convert_to_gguf or ctransformers check failed for {model_name} with config {quant_config}: {e}")
        finally:
            if os.path.exists(gguf_file_path):
                os.remove(gguf_file_path)

        del quantized_model, quantizer, model_to_quantize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def test_opt_125m_gguf_b4_gs128(self):
        """Integration test for OPT-125M with GGUF 4-bit, group_size 128."""
        if TEST_MODEL_NAME_SMALL in self.models_to_test_config:
            self._run_integration_test(
                TEST_MODEL_NAME_SMALL, 
                self.models_to_test_config[TEST_MODEL_NAME_SMALL], # Pass config, not model instance
                {"bits": 4, "group_size": 128, "use_packed": True}
            )
        else:
            self.skipTest(f"{TEST_MODEL_NAME_SMALL} config not loaded.")
            
    def test_opt_350m_gguf_b4_gs128(self):
        """Integration test for OPT-350M with GGUF 4-bit, group_size 128."""
        if TEST_MODEL_NAME_MEDIUM in self.models_to_test_config:
            self._run_integration_test(
                TEST_MODEL_NAME_MEDIUM,
                self.models_to_test_config[TEST_MODEL_NAME_MEDIUM], # Pass config
                {"bits": 4, "group_size": 128, "use_packed": True}
            )
        else:
            self.skipTest(f"{TEST_MODEL_NAME_MEDIUM} config not loaded.")


class TestGGUFBenchmarkValidation(unittest.TestCase):
    """
    Validates that QuantizationBenchmark runs correctly with GGUFQuantizer
    and produces expected metrics.
    """
    @classmethod
    def setUpClass(cls):
        """Load model and calibration data for benchmark validation."""
        cls.original_model, cls.tokenizer = _load_model_and_tokenizer(TEST_MODEL_NAME_SMALL)
        cls.calibration_data = _get_dummy_calibration_data(vocab_size=cls.original_model.config.vocab_size)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[TestGGUFBenchmarkValidation] Using device: {cls.device} for GGUF benchmark validation.")

    def test_benchmark_utility_with_gguf(self):
        """Test QuantizationBenchmark with a standard GGUF configuration."""
        print("\nTesting QuantizationBenchmark with GGUFQuantizer (4-bit, gs=128)...")
        
        # The QuantizationBenchmark utility expects the original model on CPU.
        # It will handle copying and device placement internally.
        benchmark = QuantizationBenchmark(
            model=self.original_model, 
            calibration_data=self.calibration_data,
            input_shape=(1, 64), # (batch_size, seq_len) for inference
            num_inference_steps=10, # Keep low for a unit test
            # num_warmup_steps is now a parameter to benchmark_quantizer
            device=self.device 
        )
        
        gguf_quant_args = {"bits": 4, "group_size": 128, "use_packed": True}
        
        original_model_size_gb = sum(
            p.numel() * p.element_size() for p in self.original_model.parameters()
        ) / (1024**3)

        # Call benchmark_quantizer, ensuring num_warmup_steps is passed
        results = benchmark.benchmark_quantizer(
            name="GGUF_Test_B4_GS128_BenchValidation", # Unique name for this test run
            quantizer_class=GGUFQuantizer,
            quantizer_args=gguf_quant_args,
            original_model_size_gb=original_model_size_gb,
            num_warmup_steps=2 # Explicitly pass warmup steps for this test
        )
        
        self.assertIsNotNone(results, "Benchmark results dictionary is None.")
        self.assertNotIn("error", results, f"Benchmark run reported an error: {results.get('error')}")
        
        # Check for presence of key metrics
        expected_metrics = [
            "quantization_time_s", "mean_latency_ms", "model_param_size_gb",
            "peak_mem_quantization_gb", "peak_mem_inference_gb", "throughput_inf_per_s",
            "p95_latency_ms", "compression_ratio_params", "memory_efficiency_percent"
        ]
        for metric in expected_metrics:
            self.assertIn(metric, results, f"Metric '{metric}' missing from benchmark results.")
            if isinstance(results[metric], (int, float)): # Numeric check
                 self.assertGreaterEqual(results[metric], 0, f"Metric '{metric}' should be non-negative, got {results[metric]}")
            elif results[metric] == "N/A": # Handle N/A for GPU specific metrics on CPU
                self.assertTrue(self.device.type == 'cpu' or "gpu_utilization" in metric, f"Metric {metric} is N/A on device {self.device}")


        # Basic sanity checks for some numeric metrics
        self.assertTrue(results.get("quantization_time_s", -1.0) >= 0, "Quantization time should be non-negative.")
        self.assertTrue(results.get("mean_latency_ms", -1.0) >= 0, "Mean latency should be non-negative.")
        
        model_param_size_gb = results.get("model_param_size_gb", float('inf'))
        if isinstance(model_param_size_gb, float): # Ensure it's a number before comparison
            # Quantized model size should generally be smaller than original, but allow for some overhead if not much smaller.
            self.assertLess(model_param_size_gb, original_model_size_gb * 1.2, 
                            "Quantized model size is unexpectedly larger than original.")
        
        compression_ratio = results.get("compression_ratio_params", 0.0)
        if isinstance(compression_ratio, float):
             # Expect some compression for 4-bit vs FP16/FP32 (original_model_size_gb implies FP16/32)
            self.assertGreater(compression_ratio, 1.0, "Compression ratio should be > 1.0 for effective quantization.")

        # Clean up benchmark instance
        del benchmark
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class TestGGUFMemoryLeaks(unittest.TestCase):
    """
    Tests for potential memory leaks during GGUF quantization and inference.
    These tests are more meaningful when run on a CUDA device.
    """
    @classmethod
    def setUpClass(cls):
        """Load model configuration and set device for memory tests."""
        # Only load config, actual model loaded per-test to isolate memory
        cls.model_config = AutoConfig.from_pretrained(TEST_MODEL_NAME_SMALL, trust_remote_code=True)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[TestGGUFMemoryLeaks] Using device: {cls.device} for GGUF memory leak tests.")
        if cls.device.type == 'cpu': # Some tests might be skipped or behave differently on CPU
            print("Warning: Memory leak tests are more indicative on GPU.")

    def test_memory_after_quantization_and_cleanup(self):
        """
        Tests memory usage pattern for GGUF quantization, focusing on cleanup.
        Loads a model, quantizes it, performs inference, then cleans up.
        Checks if memory returns to a state close to initial (pre-model load for this test).
        """
        print("\nTesting memory usage pattern for GGUF quantization and cleanup...")
        if self.device.type == 'cpu':
            self.skipTest("This memory leak test is designed for GPU memory validation; skipping for CPU-only.")

        tracker = MemoryTracker(device=self.device)
        tracker.start_tracking() # Logs 'initial_state' (very beginning, before any test-specific model load)
        
        initial_gpu_state = tracker.memory_logs.get("initial_state", {}).get("gpu_memory", {})
        if not isinstance(initial_gpu_state, dict): # Handle case where GPU is not available/tracked
            initial_gpu_allocated_gb = 0.0
            print("Warning: Could not get initial GPU state for memory test.")
        else:
            initial_gpu_allocated_gb = initial_gpu_state.get("gpu_allocated_current_gb", 0.0)


        # --- Stage 1: Load original model for this test ---
        # This model is loaded specifically for this test method's scope.
        model_in_test = AutoModelForCausalLM.from_config(self.model_config).to(self.device)
        model_in_test.eval()
        tracker.log_memory("after_test_model_load")
        
        model_load_gpu_state = tracker.memory_logs["after_test_model_load"]["gpu_memory"]
        model_load_gpu_allocated_gb = model_load_gpu_state.get("gpu_allocated_current_gb", 0.0)
        # Memory should increase after loading the model for this test.
        self.assertGreater(model_load_gpu_allocated_gb, initial_gpu_allocated_gb, 
                           "GPU memory should increase after loading the test-specific model.")

        # --- Stage 2: Quantization ---
        # Use a clone of model_in_test if GGUFQuantizer modifies the model_name instance it's given.
        # GGUFQuantizer takes model_name, which can be a model instance.
        # It's safer to pass a fresh instance or ensure the quantizer doesn't hold persistent refs to this one.
        # For this test, we pass `model_in_test` and will delete it later.
        calibration_data_memtest = _get_dummy_calibration_data(vocab_size=self.model_config.vocab_size).to(self.device)
        quant_args = {"bits": 4, "group_size": 128, "use_packed": True, "desc_act": False}
        
        gguf_quantizer_memtest = GGUFQuantizer(model_name=model_in_test, **quant_args, device=self.device)
        tracker.log_memory("after_quantizer_init")
        
        quantized_model_memtest = gguf_quantizer_memtest.quantize(calibration_data=calibration_data_memtest)
        quantized_model_memtest.to(self.device) # Ensure it's on device
        tracker.log_memory("after_quantization")
        quant_peak_gpu_gb = tracker.memory_logs["after_quantization"]["gpu_memory"].get("gpu_allocated_peak_gb", 0.0)

        # --- Stage 3: Inference ---
        tokenizer_memtest = AutoTokenizer.from_pretrained(TEST_MODEL_NAME_SMALL)
        if tokenizer_memtest.pad_token is None: tokenizer_memtest.pad_token = tokenizer_memtest.eos_token
        dummy_input_ids_memtest = tokenizer_memtest("Memory test", return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            _ = quantized_model_memtest(dummy_input_ids_memtest)
        tracker.log_memory("after_inference")
        inference_peak_gpu_gb = tracker.memory_logs["after_inference"]["gpu_memory"].get("gpu_allocated_peak_gb", 0.0)
        
        # --- Stage 4: Cleanup ---
        # Important: Delete in an order that might break circular references if any.
        # Quantizer might hold references to parts of the original model or the quantized one.
        del dummy_input_ids_memtest, tokenizer_memtest 
        del quantized_model_memtest # This is the output of quantizer.quantize()
        del gguf_quantizer_memtest  # This might hold references to model_in_test
        del model_in_test           # The model that was passed to the quantizer
        del calibration_data_memtest
        
        gc.collect() # Python GC
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear PyTorch's CUDA cache
            torch.cuda.synchronize(self.device) # Wait for CUDA operations to complete
        
        tracker.log_memory("after_cleanup_operations") # Log memory after all deletions and CUDA cache clear
        tracker.stop_tracking() # Logs 'final_state' which is same as 'after_cleanup_operations' here
        tracker.print_report() # Print the full memory log for this test

        cleanup_gpu_state = tracker.memory_logs["after_cleanup_operations"]["gpu_memory"]
        cleanup_gpu_allocated_gb = cleanup_gpu_state.get("gpu_allocated_current_gb", 0.0)
        
        # Assertions:
        # 1. Peak memory during quantization/inference should be higher than after the model was initially loaded for the test.
        self.assertGreaterEqual(quant_peak_gpu_gb, model_load_gpu_allocated_gb, "Quantization peak memory error.")
        self.assertGreaterEqual(inference_peak_gpu_gb, model_load_gpu_allocated_gb, "Inference peak memory error.")
        
        # 2. Memory after cleanup should be significantly less than peak usage.
        #    And ideally close to the `initial_state` (before any model was loaded for *this specific test method*).
        #    Allowing for some Python overhead and CUDA context persistence.
        #    A strict check against `initial_gpu_allocated_gb` (before this test's model load) is best.
        
        residual_overhead_allowance_gb = 0.05 # Allow 50MB overhead over the absolute initial state.
                                              # This accounts for CUDA context, Python objects, etc.
        self.assertLessEqual(
            cleanup_gpu_allocated_gb, 
            initial_gpu_allocated_gb + residual_overhead_allowance_gb,
            f"Memory after cleanup ({cleanup_gpu_allocated_gb:.3f}GB) is significantly higher "
            f"than initial state before this test's model load ({initial_gpu_allocated_gb:.3f}GB) plus allowance. Potential leak."
        )
        # Also check it's much lower than the peak during quantization (heuristic)
        if quant_peak_gpu_gb > 0: # Avoid division by zero or issues if peak was 0 (e.g. CPU test somehow ran)
            self.assertLess(
                cleanup_gpu_allocated_gb,
                quant_peak_gpu_gb * 0.75, # Expect it to be less than 75% of the peak. Stricter might be 0.5.
                "Memory after cleanup is not significantly less than peak quantization memory."
            )

if __name__ == '__main__':
    unittest.main() # This will run all TestCases in this file
