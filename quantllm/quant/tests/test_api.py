import unittest
import torch
import os
import tempfile
from transformers import PreTrainedModel
from quantllm.api.high_level import QuantizerFactory
from quantllm.quant.quantization_engine import QuantizedLinear
from quantllm.quant.gguf import GGUFQuantizer # For direct instantiation in convert_to_gguf test
from .helpers import create_dummy_calibration_data

# Use a small, generally available model for testing
TEST_MODEL_NAME = "facebook/opt-125m" 
# TEST_MODEL_NAME = "EleutherAI/pythia-160m-deduped" # Alternative if OPT is problematic

class TestQuantizerFactory(unittest.TestCase):
    def setUp(self):
        """Set up for test cases."""
        # opt-125m has hidden_size=768, vocab_size=50272
        # pythia-160m has hidden_size=768, vocab_size=50304
        # The calibration data should be token IDs.
        self.calibration_data = create_dummy_calibration_data(
            batch_size=2, 
            vocab_size=30000, # A generic vocab size for dummy data
            seq_len=32,       # Sequence length for calibration samples
            num_samples=8     # Number of calibration samples
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_hidden_size = 768 # For opt-125m or pythia-160m

    def _assert_quantized_model_validity(self, model, tokenizer, expected_method, 
                                         expected_bits, expected_group_size, 
                                         expected_device_type, # 'cpu' or 'cuda'
                                         quant_config_used): # Pass the quant_config for specific checks
        self.assertIsNotNone(model, "Model should not be None")
        self.assertIsNotNone(tokenizer, "Tokenizer should not be None")
        self.assertIsInstance(model, PreTrainedModel, "Model is not a PreTrainedModel instance")
        
        self.assertTrue(hasattr(model, 'config'), "Model does not have a config attribute.")
        self.assertTrue(hasattr(model.config, 'quantization_config'), 
                        "model.config.quantization_config attribute is missing.")
        self.assertIsInstance(model.config.quantization_config, dict, 
                              "model.config.quantization_config is not a dictionary.")
        
        quant_cfg_from_model = model.config.quantization_config
        self.assertEqual(quant_cfg_from_model.get("quant_method"), expected_method, "Quant method in config mismatch.")
        self.assertEqual(quant_cfg_from_model.get("bits"), expected_bits, "Bits in config mismatch.")
        
        # group_size might be -1 for per-tensor, or not present if not applicable to method's storage
        if "group_size" in quant_cfg_from_model and expected_group_size != -1 :
             self.assertEqual(quant_cfg_from_model.get("group_size"), expected_group_size, "Group size in config mismatch.")
        elif expected_group_size == -1 and "group_size" in quant_cfg_from_model:
             self.assertEqual(quant_cfg_from_model.get("group_size"), expected_group_size, "Group size should be -1 for per-tensor.")


        # Device check
        # If GGUF and cpu_offload is True, parameters should be on CPU.
        is_gguf_cpu_offload = (expected_method == 'gguf' and 
                               quant_config_used.get('cpu_offload', False))
        
        final_expected_device_type = 'cpu' if is_gguf_cpu_offload else expected_device_type

        # Check device of parameters (at least one)
        # Some parameters might remain on meta device if not modified.
        # We are interested in the device of actual compute layers.
        found_param_on_device = False
        for name, param in model.named_parameters():
            if param.device.type == final_expected_device_type:
                found_param_on_device = True
                break
        if model.state_dict(): # Only assert if model has parameters
            self.assertTrue(found_param_on_device, f"No parameters found on expected device type: {final_expected_device_type}")


        # Check for QuantizedLinear layers
        has_quantized_linear = any(isinstance(m, QuantizedLinear) for m in model.modules())
        self.assertTrue(has_quantized_linear, "No QuantizedLinear layers found in the model.")

    # --- AWQ Tests ---
    def test_quantize_awq_b4_gs128_cpu(self):
        quant_config = {"bits": 4, "group_size": 128, "zero_point": True, "awq_version": "v2"}
        model, tokenizer = QuantizerFactory.quantize_from_pretrained(
            TEST_MODEL_NAME, "awq", quant_config, self.calibration_data, device="cpu", calibration_steps=5
        )
        self._assert_quantized_model_validity(model, tokenizer, "awq", 4, 128, "cpu", quant_config)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_quantize_awq_b4_gs64_gpu(self):
        quant_config = {"bits": 4, "group_size": 64, "zero_point": False, "awq_version": "v2"}
        model, tokenizer = QuantizerFactory.quantize_from_pretrained(
            TEST_MODEL_NAME, "awq", quant_config, self.calibration_data, device=self.device, calibration_steps=5
        )
        self._assert_quantized_model_validity(model, tokenizer, "awq", 4, 64, self.device, quant_config)

    def test_quantize_awq_b8_gsNeg1_cpu(self): # group_size -1 for per-tensor
        quant_config = {"bits": 8, "group_size": -1, "zero_point": True, "awq_version": "v2"}
        model, tokenizer = QuantizerFactory.quantize_from_pretrained(
            TEST_MODEL_NAME, "awq", quant_config, self.calibration_data, device="cpu", calibration_steps=5
        )
        self._assert_quantized_model_validity(model, tokenizer, "awq", 8, -1, "cpu", quant_config)

    # --- GPTQ Tests ---
    def test_quantize_gptq_b4_gs128_symTrue_cpu(self):
        quant_config = {"bits": 4, "group_size": 128, "sym": True, "actorder": False}
        model, tokenizer = QuantizerFactory.quantize_from_pretrained(
            TEST_MODEL_NAME, "gptq", quant_config, self.calibration_data, device="cpu"
        )
        self._assert_quantized_model_validity(model, tokenizer, "gptq", 4, 128, "cpu", quant_config)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_quantize_gptq_b4_gs64_symFalse_actOrder_gpu(self):
        quant_config = {"bits": 4, "group_size": 64, "sym": False, "actorder": True}
        model, tokenizer = QuantizerFactory.quantize_from_pretrained(
            TEST_MODEL_NAME, "gptq", quant_config, self.calibration_data, device=self.device
        )
        self._assert_quantized_model_validity(model, tokenizer, "gptq", 4, 64, self.device, quant_config)

    def test_quantize_gptq_b8_gs32_cpu(self):
        quant_config = {"bits": 8, "group_size": 32, "sym": True}
        model, tokenizer = QuantizerFactory.quantize_from_pretrained(
            TEST_MODEL_NAME, "gptq", quant_config, self.calibration_data, device="cpu"
        )
        self._assert_quantized_model_validity(model, tokenizer, "gptq", 8, 32, "cpu", quant_config)
        
    # --- GGUF Tests ---
    def test_quantize_gguf_b4_gs32_packed_cpu(self):
        quant_config = {"bits": 4, "group_size": 32, "use_packed": True, "cpu_offload": False}
        model, tokenizer = QuantizerFactory.quantize_from_pretrained(
            TEST_MODEL_NAME, "gguf", quant_config, self.calibration_data, device="cpu"
        )
        self._assert_quantized_model_validity(model, tokenizer, "gguf", 4, 32, "cpu", quant_config)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_quantize_gguf_b4_gs64_packed_gpu(self):
        quant_config = {"bits": 4, "group_size": 64, "use_packed": True, "cpu_offload": False}
        model, tokenizer = QuantizerFactory.quantize_from_pretrained(
            TEST_MODEL_NAME, "gguf", quant_config, self.calibration_data, device=self.device
        )
        self._assert_quantized_model_validity(model, tokenizer, "gguf", 4, 64, self.device, quant_config)

    def test_quantize_gguf_b8_gs32_noPacked_cpuOffload_cpu(self): # Quantizer on CPU, layers on CPU
        quant_config = {"bits": 8, "group_size": 32, "use_packed": False, "cpu_offload": True}
        model, tokenizer = QuantizerFactory.quantize_from_pretrained(
            TEST_MODEL_NAME, "gguf", quant_config, self.calibration_data, device="cpu"
        )
        # Even if device="cpu", cpu_offload=True means layers are on CPU.
        self._assert_quantized_model_validity(model, tokenizer, "gguf", 8, 32, "cpu", quant_config)
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_quantize_gguf_b4_gs32_packed_cpuOffload_gpu(self): # Quantizer on GPU, layers on CPU
        quant_config = {"bits": 4, "group_size": 32, "use_packed": True, "cpu_offload": True}
        model, tokenizer = QuantizerFactory.quantize_from_pretrained(
            TEST_MODEL_NAME, "gguf", quant_config, self.calibration_data, device=self.device 
        )
        # Device for assertion should be 'cpu' because of cpu_offload=True
        self._assert_quantized_model_validity(model, tokenizer, "gguf", 4, 32, "cpu", quant_config)


    # --- GGUF Conversion Test (using direct GGUFQuantizer for now) ---
    def _run_gguf_conversion_test(self, quantizer_device, gguf_cpu_offload):
        quant_config = {"bits": 4, "group_size": 32, "cpu_offload": gguf_cpu_offload, "use_packed": True}
        
        # Instantiate GGUFQuantizer directly for this test
        # BaseQuantizer now handles model loading from name/path
        gguf_quantizer = GGUFQuantizer(
            model_or_model_name_or_path=TEST_MODEL_NAME,
            bits=quant_config["bits"],
            group_size=quant_config["group_size"],
            cpu_offload=quant_config["cpu_offload"],
            use_packed=quant_config["use_packed"],
            device=quantizer_device 
        )
        gguf_quantizer.quantize(calibration_data=self.calibration_data.clone())
        
        temp_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as tmp_file:
                temp_file_path = tmp_file.name
            
            gguf_quantizer.convert_to_gguf(output_path=temp_file_path)
            self.assertTrue(os.path.exists(temp_file_path), f"GGUF file was not created at {temp_file_path}")
            self.assertTrue(os.path.getsize(temp_file_path) > 0, "GGUF file is empty.")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_gguf_convert_to_gguf_cpu_layersCpu(self): # Quantizer on CPU, layers on CPU
        self._run_gguf_conversion_test(quantizer_device="cpu", gguf_cpu_offload=True)

    def test_gguf_convert_to_gguf_cpu_layersDevice(self): # Quantizer on CPU, layers on CPU (no offload means quantizer device)
        self._run_gguf_conversion_test(quantizer_device="cpu", gguf_cpu_offload=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gguf_convert_to_gguf_gpu_layersCpu(self): # Quantizer on GPU, layers on CPU
        self._run_gguf_conversion_test(quantizer_device=self.device, gguf_cpu_offload=True)
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gguf_convert_to_gguf_gpu_layersDevice(self): # Quantizer on GPU, layers on GPU
        self._run_gguf_conversion_test(quantizer_device=self.device, gguf_cpu_offload=False)

# New test class for move_to_device
from quantllm.quant.quantization_engine import move_to_device
import torch.nn as nn

class TestMoveToDevice(unittest.TestCase):
    def test_move_tensor_and_module(self):
        """Test move_to_device with both torch.Tensor and torch.nn.Module."""
        target_device_str = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(target_device_str)

        # 1. Create a simple torch.Tensor
        my_tensor = torch.randn(2, 3, device="cpu") # Start on CPU

        # 2. Create a simple torch.nn.Module
        my_module = nn.Linear(10, 10).to("cpu") # Start on CPU

        # 4. Call move_to_device for the tensor and the module
        moved_tensor = move_to_device(my_tensor, target_device)
        moved_module = move_to_device(my_module, target_device)

        # 5. Assert that the tensor is on the target device
        self.assertEqual(moved_tensor.device, target_device, "Tensor not moved to target device.")

        # 6. Assert that the module is on the target device
        self.assertIsInstance(moved_module, nn.Module, "move_to_device did not return a Module.")
        
        # Check device of a parameter
        if list(moved_module.parameters()): # Check if module has parameters
            self.assertEqual(
                next(moved_module.parameters()).device, 
                target_device, 
                "Module's parameters not moved to target device."
            )
        else: # Handle modules with no parameters (e.g. nn.ReLU()) if needed for future tests
            # For a simple Linear layer, this else block shouldn't be hit.
            # If testing with modules without parameters, one might check an attribute
            # or skip device check if not applicable. For nn.Linear, parameters exist.
            pass

        # Test with force_copy=True for tensors
        another_tensor = torch.randn(2,3, device=target_device)
        copied_tensor = move_to_device(another_tensor, target_device, force_copy=True)
        self.assertEqual(copied_tensor.device, target_device)
        if target_device_str == "cpu": # On CPU, to() without copy=True might return same object if already on device
             pass # Data pointer check is more complex and not strictly necessary for device check
        else: # On CUDA, .to(device) typically creates a new tensor unless it's already there.
             # force_copy=True should ensure it's a different object.
             if another_tensor.is_cuda and copied_tensor.is_cuda: # Both on CUDA
                self.assertNotEqual(another_tensor.data_ptr(), copied_tensor.data_ptr(), "force_copy=True did not create a new tensor copy on CUDA.")
        
        # Test moving a module already on the target device
        module_on_target = nn.Linear(5,5).to(target_device)
        moved_module_again = move_to_device(module_on_target, target_device)
        self.assertEqual(next(moved_module_again.parameters()).device, target_device)


if __name__ == '__main__':
    unittest.main()
