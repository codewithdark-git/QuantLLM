import unittest
import torch
from transformers import PreTrainedModel
from quantllm.quant.gptq import GPTQQuantizer
from quantllm.quant.quantization_engine import QuantizedLinear
from .helpers import create_dummy_model, create_dummy_calibration_data

class TestGPTQQuantizer(unittest.TestCase):
    def setUp(self):
        """Set up for test cases."""
        self.input_dim = 16
        self.hidden_dim = 32
        self.output_dim = 10
        # self.dummy_model is created per test in _run_quantization_test to ensure fresh state
        self.calibration_data = create_dummy_calibration_data(
            batch_size=4, 
            dim=self.input_dim, 
            num_samples=20 # GPTQ might need sufficient calibration data
        )

    def _run_quantization_test(self, device, bits, group_size, sym, actorder=False):
        """Helper function to run GPTQ quantization test."""
        model_to_quantize = create_dummy_model(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        ).to(device)
        cal_data = self.calibration_data.to(device)

        quantizer = GPTQQuantizer(
            model=model_to_quantize,
            bits=bits,
            group_size=group_size,
            sym=sym,
            actorder=actorder, # Typically False for basic tests unless specifically testing actorder
            device=device
        )
        
        quantized_model = quantizer.quantize(calibration_data=cal_data)
        
        self.assertIsNotNone(quantized_model, "Quantized model should not be None.")
        self.assertIsInstance(quantized_model, PreTrainedModel, "Quantized model should be a PreTrainedModel.")
        if device != 'cpu':
            self.assertEqual(quantized_model.device.type, device, f"Model should be on {device} device.")

        replaced_layers = 0
        for module_name, module in quantized_model.named_modules():
            if isinstance(module, QuantizedLinear):
                replaced_layers += 1
                self.assertEqual(module.config.bits, bits, f"QuantizedLinear layer {module_name} has incorrect bits.")
                expected_scheme = "symmetric" if sym else "asymmetric"
                self.assertEqual(module.config.scheme, expected_scheme, f"QuantizedLinear layer {module_name} has incorrect scheme.")
                # GPTQ in this implementation sets granularity to per-channel in QuantizedLinear's config
                self.assertEqual(module.config.granularity, "per-channel", f"QuantizedLinear layer {module_name} has incorrect granularity.")
        
        self.assertTrue(replaced_layers > 0, "At least one nn.Linear module should be replaced.")
        # DummyModel has 2 linear layers
        self.assertEqual(replaced_layers, 2, "Both linear layers should have been replaced.")

    # --- Parameter Variation Tests ---
    # bits=4, group_size=128, sym=True
    def test_gptq_b4_gs128_symTrue_cpu(self):
        self._run_quantization_test(device='cpu', bits=4, group_size=128, sym=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gptq_b4_gs128_symTrue_gpu(self):
        self._run_quantization_test(device='cuda', bits=4, group_size=128, sym=True)

    # bits=8, group_size=64, sym=False
    def test_gptq_b8_gs64_symFalse_cpu(self):
        self._run_quantization_test(device='cpu', bits=8, group_size=64, sym=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gptq_b8_gs64_symFalse_gpu(self):
        self._run_quantization_test(device='cuda', bits=8, group_size=64, sym=False)
        
    # bits=4, group_size=32 (smaller group size), sym=True
    def test_gptq_b4_gs32_symTrue_cpu(self):
        self._run_quantization_test(device='cpu', bits=4, group_size=32, sym=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gptq_b4_gs32_symTrue_gpu(self):
        self._run_quantization_test(device='cuda', bits=4, group_size=32, sym=True)

    # --- Output Consistency Tests ---
    def _run_output_consistency_test(self, device):
        original_model = create_dummy_model(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        ).to(device).eval()

        # Using a single sample for consistency check
        sample_input = self.calibration_data[0:1].clone().to(device)

        with torch.no_grad():
            original_output = original_model(input_ids=sample_input)

        model_to_quantize = create_dummy_model(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        ).to(device)
        
        cal_data = self.calibration_data.to(device)

        quantizer = GPTQQuantizer(
            model=model_to_quantize,
            bits=8, # Using 8-bit for better chance of closeness
            group_size=128,
            sym=True, # Symmetric often has less error for simple models
            device=device
        )
        
        quantized_model = quantizer.quantize(calibration_data=cal_data).eval()

        with torch.no_grad():
            quantized_output = quantized_model(input_ids=sample_input)
        
        self.assertTrue(
            torch.allclose(original_output, quantized_output, atol=0.5), # GPTQ can have larger diffs
            f"Output from original and quantized GPTQ model on {device} not close. "
            f"Original: {original_output}, Quantized: {quantized_output}"
        )

    def test_output_consistency_cpu(self):
        """Test output consistency on CPU for GPTQ."""
        self._run_output_consistency_test(device='cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_output_consistency_gpu(self):
        """Test output consistency on GPU for GPTQ."""
        self._run_output_consistency_test(device='cuda')


if __name__ == '__main__':
    unittest.main()
