import unittest
import torch
import os
import tempfile
from transformers import PreTrainedModel
from quantllm.quant.gguf import GGUFQuantizer
from quantllm.quant.quantization_engine import QuantizedLinear
from .helpers import create_dummy_model, create_dummy_calibration_data

class TestGGUFQuantizer(unittest.TestCase):
    def setUp(self):
        """Set up for test cases."""
        self.input_dim = 16
        self.hidden_dim = 32
        self.output_dim = 10
        self.calibration_data = create_dummy_calibration_data(
            batch_size=4, 
            dim=self.input_dim, 
            num_samples=20
        )

    def _run_quantization_test(self, device_quantizer, bits, group_size, use_packed, cpu_offload_quant):
        """Helper function to run GGUF quantization test."""
        # Create a fresh model, quantizer will move it according to its device/cpu_offload settings
        model_to_quantize = create_dummy_model(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        # Calibration data is moved to device_quantizer by quantizer.prepare_calibration_data()

        quantizer = GGUFQuantizer(
            model=model_to_quantize,
            bits=bits,
            group_size=group_size,
            use_packed=use_packed,
            device=device_quantizer, # This is the device for the quantizer's main operations
            cpu_offload=cpu_offload_quant # This determines where quantized layers are placed
        )
        
        quantized_model = quantizer.quantize(calibration_data=self.calibration_data.clone())
        
        self.assertIsNotNone(quantized_model, "Quantized model should not be None.")
        self.assertIsInstance(quantized_model, PreTrainedModel, "Quantized model should be a PreTrainedModel.")

        # Determine expected device of quantized layers based on cpu_offload_quant
        expected_layer_device_type = 'cpu' if cpu_offload_quant else device_quantizer

        replaced_layers = 0
        for module_name, module in quantized_model.named_modules():
            if isinstance(module, QuantizedLinear):
                replaced_layers += 1
                self.assertEqual(module.config.bits, bits, f"QL {module_name} bits incorrect.")
                self.assertEqual(module.config.format, "gguf", f"QL {module_name} format incorrect.")
                self.assertEqual(module.config.format_config.get("use_packed"), use_packed, f"QL {module_name} use_packed incorrect.")
                self.assertEqual(module.config.format_config.get("group_size"), group_size, f"QL {module_name} group_size incorrect.")
                
                # Check device of the QuantizedLinear module's parameters
                # For GGUF with cpu_offload=True, weights are on CPU.
                # Otherwise, they are on the device_quantizer.
                for param_name, param in module.named_parameters():
                     self.assertEqual(param.device.type, expected_layer_device_type, 
                                     f"Parameter {param_name} of QL {module_name} on wrong device. Expected {expected_layer_device_type}, got {param.device.type}")

        self.assertTrue(replaced_layers > 0, "No nn.Linear modules replaced.")
        self.assertEqual(replaced_layers, 2, "Incorrect number of nn.Linear modules replaced.")

    # --- Parameter Variation Tests ---
    def test_gguf_b4_gs32_packedTrue_cpu(self):
        self._run_quantization_test(device_quantizer='cpu', bits=4, group_size=32, use_packed=True, cpu_offload_quant=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gguf_b4_gs32_packedTrue_gpu(self):
        self._run_quantization_test(device_quantizer='cuda', bits=4, group_size=32, use_packed=True, cpu_offload_quant=False)

    def test_gguf_b8_gs64_packedFalse_cpu(self):
        self._run_quantization_test(device_quantizer='cpu', bits=8, group_size=64, use_packed=False, cpu_offload_quant=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gguf_b8_gs64_packedFalse_gpu(self):
        self._run_quantization_test(device_quantizer='cuda', bits=8, group_size=64, use_packed=False, cpu_offload_quant=False)
    
    # Test with cpu_offload=True (quantizer on CPU, layers also on CPU)
    def test_gguf_b4_gs32_packedTrue_cpuOffloadTrue_cpuQuantizer(self):
        self._run_quantization_test(device_quantizer='cpu', bits=4, group_size=32, use_packed=True, cpu_offload_quant=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gguf_b4_gs32_packedTrue_cpuOffloadTrue_gpuQuantizer(self):
         # Quantizer runs on GPU, but quantized layers are offloaded to CPU
        self._run_quantization_test(device_quantizer='cuda', bits=4, group_size=32, use_packed=True, cpu_offload_quant=True)


    # --- Output Consistency Tests ---
    def _run_output_consistency_test(self, device_quantizer, cpu_offload_quant):
        # Model for original output, placed on the device where inference will happen
        # If cpu_offload_quant is true, quantized model runs on CPU. So original should too for comparison.
        eval_device = 'cpu' if cpu_offload_quant else device_quantizer
        
        original_model = create_dummy_model(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim
        ).to(eval_device).eval()

        sample_input = self.calibration_data[0:1].clone().to(eval_device)
        with torch.no_grad():
            original_output = original_model(input_ids=sample_input)

        model_to_quantize = create_dummy_model(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim
        ) # Initialized on CPU

        quantizer = GGUFQuantizer(
            model=model_to_quantize, bits=8, group_size=32, use_packed=True, 
            device=device_quantizer, cpu_offload=cpu_offload_quant
        )
        quantized_model = quantizer.quantize(calibration_data=self.calibration_data.clone()).eval()
        # Quantized model is already on `eval_device` due to cpu_offload logic in GGUFQuantizer

        with torch.no_grad():
            quantized_output = quantized_model(input_ids=sample_input)
        
        self.assertTrue(
            torch.allclose(original_output, quantized_output, atol=0.5),
            f"Outputs not close. DeviceQ: {device_quantizer}, CPUOffloadQ: {cpu_offload_quant}. "
            f"Orig: {original_output}, Quant: {quantized_output}"
        )

    def test_output_consistency_cpu_noOffload(self):
        self._run_output_consistency_test(device_quantizer='cpu', cpu_offload_quant=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_output_consistency_gpu_noOffload(self):
        self._run_output_consistency_test(device_quantizer='cuda', cpu_offload_quant=False)

    def test_output_consistency_cpu_withOffload(self): # Quantizer on CPU, layers on CPU
        self._run_output_consistency_test(device_quantizer='cpu', cpu_offload_quant=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_output_consistency_gpu_withOffload(self): # Quantizer on GPU, layers on CPU
        self._run_output_consistency_test(device_quantizer='cuda', cpu_offload_quant=True)

    # --- GGUF Conversion Test ---
    def _run_convert_to_gguf_test(self, device_quantizer, cpu_offload_quant):
        model_to_quantize = create_dummy_model(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim
        )
        quantizer = GGUFQuantizer(
            model=model_to_quantize, bits=4, group_size=32, use_packed=True,
            device=device_quantizer, cpu_offload=cpu_offload_quant
        )
        quantizer.quantize(calibration_data=self.calibration_data.clone()) # Quantize first

        temp_file = None
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as tmp_file:
                temp_file_path = tmp_file.name
            
            quantizer.convert_to_gguf(output_path=temp_file_path)
            
            self.assertTrue(os.path.exists(temp_file_path), "GGUF file was not created.")
            self.assertTrue(os.path.getsize(temp_file_path) > 0, "GGUF file is empty.")
            
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_convert_to_gguf_cpu_noOffload(self):
        self._run_convert_to_gguf_test(device_quantizer='cpu', cpu_offload_quant=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_convert_to_gguf_gpu_noOffload(self):
        self._run_convert_to_gguf_test(device_quantizer='cuda', cpu_offload_quant=False)
    
    def test_convert_to_gguf_cpu_withOffload(self):
        self._run_convert_to_gguf_test(device_quantizer='cpu', cpu_offload_quant=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_convert_to_gguf_gpu_withOffload(self):
        self._run_convert_to_gguf_test(device_quantizer='cuda', cpu_offload_quant=True)


if __name__ == '__main__':
    unittest.main()
