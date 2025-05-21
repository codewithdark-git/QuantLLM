import unittest
import torch
from transformers import PreTrainedModel
from quantllm.quant.awq import AWQQuantizer
from quantllm.quant.quantization_engine import QuantizedLinear # For checking replacement
from .helpers import create_dummy_model, create_dummy_calibration_data

class TestAWQQuantizer(unittest.TestCase):
    def setUp(self):
        """Set up for test cases."""
        self.input_dim = 16
        self.hidden_dim = 32
        self.output_dim = 10
        self.dummy_model = create_dummy_model(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            output_dim=self.output_dim
        )
        # AWQ might need more samples or specific data characteristics for robust testing,
        # but this is a start.
        self.calibration_data = create_dummy_calibration_data(
            batch_size=4, 
            dim=self.input_dim, 
            num_samples=20 
        )

    def _run_quantization_test(self, device, bits, group_size, zero_point):
        """Helper function to run AWQ quantization test."""
        # Create a fresh model for each test to avoid state issues
        model_to_quantize = create_dummy_model(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            output_dim=self.output_dim
        )
        model_to_quantize = model_to_quantize.to(device)
        cal_data = self.calibration_data.to(device)

        quantizer = AWQQuantizer(
            model=model_to_quantize, 
            bits=bits,
            group_size=group_size,
            zero_point=zero_point,
            device=device
        )
        
        quantized_model = quantizer.quantize(
            calibration_data=cal_data,
            calibration_steps=5 # Small number of steps for testing
        )
        
        self.assertIsNotNone(quantized_model, "Quantized model should not be None.")
        self.assertIsInstance(quantized_model, PreTrainedModel, "Quantized model should be a PreTrainedModel.")
        if device != 'cpu':
             self.assertEqual(quantized_model.device.type, device, f"Model should be on {device} device after quantization.")

        replaced_layers = 0
        for module_name, module in quantized_model.named_modules():
            if isinstance(module, QuantizedLinear):
                replaced_layers += 1
                self.assertEqual(module.config.bits, bits, f"QuantizedLinear layer {module_name} has incorrect bits.")
                # AWQ always uses symmetric for weights, zero_point for weights is part of its scale calculation logic
                # The zero_point parameter in AWQQuantizer controls activation zero point, which is not directly in QuantizedLinear.config
                # Granularity check based on group_size
                if group_size > 0 :
                    self.assertTrue(module.config.channel_wise, f"QuantizedLinear layer {module_name} should be channel_wise for group_size > 0.")
                    self.assertEqual(module.config.granularity, "per-channel", f"QuantizedLinear layer {module_name} has incorrect granularity.")
                else: # group_size == -1 (per-tensor)
                    self.assertFalse(module.config.channel_wise, f"QuantizedLinear layer {module_name} should not be channel_wise for group_size == -1.")
                    self.assertEqual(module.config.granularity, "per-tensor", f"QuantizedLinear layer {module_name} has incorrect granularity.")

        self.assertTrue(replaced_layers > 0, "At least one nn.Linear module should have been replaced by QuantizedLinear.")
        # Assuming DummyModel has 2 linear layers
        self.assertEqual(replaced_layers, 2, "Both linear layers should have been replaced.")

    def test_awq_b8_gs32_zpTrue_cpu(self):
        """Test AWQ 8-bit, group_size=32, zero_point=True on CPU."""
        self._run_quantization_test(device='cpu', bits=8, group_size=32, zero_point=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_awq_b4_gs32_zpTrue_gpu(self):
        """Test AWQ 4-bit, group_size=32, zero_point=True on GPU."""
        self._run_quantization_test(device='cuda', bits=4, group_size=32, zero_point=True)

    def test_awq_b4_gs64_zpTrue_cpu(self):
        """Test AWQ 4-bit, group_size=64, zero_point=True on CPU."""
        self._run_quantization_test(device='cpu', bits=4, group_size=64, zero_point=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_awq_b4_gs64_zpTrue_gpu(self):
        """Test AWQ 4-bit, group_size=64, zero_point=True on GPU."""
        self._run_quantization_test(device='cuda', bits=4, group_size=64, zero_point=True)

    def test_awq_b8_gsNeg1_zpFalse_cpu(self): # gsNeg1 means per-tensor
        """Test AWQ 8-bit, group_size=-1 (per-tensor), zero_point=False on CPU."""
        self._run_quantization_test(device='cpu', bits=8, group_size=-1, zero_point=False)
        
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_awq_b8_gsNeg1_zpFalse_gpu(self): # gsNeg1 means per-tensor
        """Test AWQ 8-bit, group_size=-1 (per-tensor), zero_point=False on GPU."""
        self._run_quantization_test(device='cuda', bits=8, group_size=-1, zero_point=False)

    def test_awq_b4_gs32_zpFalse_cpu(self):
        """Test AWQ 4-bit, group_size=32, zero_point=False on CPU."""
        self._run_quantization_test(device='cpu', bits=4, group_size=32, zero_point=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_awq_b4_gs32_zpFalse_gpu(self):
        """Test AWQ 4-bit, group_size=32, zero_point=False on GPU."""
        self._run_quantization_test(device='cuda', bits=4, group_size=32, zero_point=False)
        
    def _run_output_consistency_test(self, device):
        # Create a fresh model for each test
        original_model = create_dummy_model(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            output_dim=self.output_dim
        ).to(device).eval()
        
        # Prepare sample input
        # Use a single sample from calibration data for consistency, or a new random tensor
        # The dummy model's forward expects input_ids to be features if last dim matches input_dim
        sample_input = self.calibration_data[0:1].clone().to(device) # Shape (1, input_dim)

        # Get original output
        with torch.no_grad():
            original_output = original_model(input_ids=sample_input)

        # Quantize the model (using a copy for quantization)
        model_to_quantize = create_dummy_model(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            output_dim=self.output_dim
        ).to(device) # Quantizer will move it if device arg is different, but good practice

        quantizer = AWQQuantizer(
            model=model_to_quantize,
            bits=8, # Using 8-bit for potentially closer results
            group_size=32,
            zero_point=True,
            device=device 
        )
        cal_data = self.calibration_data.to(device)
        quantized_model = quantizer.quantize(
            calibration_data=cal_data,
            calibration_steps=5 
        ).eval() # Set to eval mode

        # Get quantized output
        with torch.no_grad():
            quantized_output = quantized_model(input_ids=sample_input)
        
        self.assertTrue(
            torch.allclose(original_output, quantized_output, atol=0.5), # Increased atol
            f"Output from original and quantized model on {device} are not close enough. "
            f"Original: {original_output}, Quantized: {quantized_output}"
        )

    def test_output_consistency_cpu(self):
        """Test output consistency on CPU for AWQ."""
        self._run_output_consistency_test(device='cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_output_consistency_gpu(self):
        """Test output consistency on GPU for AWQ."""
        self._run_output_consistency_test(device='cuda')

if __name__ == '__main__':
    unittest.main()
