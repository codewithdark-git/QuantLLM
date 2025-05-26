import unittest
import torch
import torch.nn as nn
from typing import List

# Assuming AWQQuantizer and QuantizedLinear are accessible from this path
# Adjust the import path based on your project structure if necessary
from quantllm.quant.awq import AWQQuantizer
from quantllm.quant.quantization_engine import QuantizedLinear, QuantizationConfig

# 1. Dummy Model Definition
class DummyModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # If input is 3D (batch, seq, features), flatten sequence for linear layers
        original_shape = x.shape
        if x.ndim == 3:
            x = x.view(-1, original_shape[-1])
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # Reshape back if original input was 3D
        if len(original_shape) == 3:
            x = x.view(original_shape[0], original_shape[1], -1)
        return x

class TestAWQQuantizer(unittest.TestCase):
    def setUp(self):
        self.in_features = 16
        self.hidden_features = 32 # Must be divisible by group_size if group_size is not -1 or 1
        self.out_features = 8
        self.seq_len = 10
        self.batch_size = 4

        # Instantiate the dummy model
        self.model = DummyModel(self.in_features, self.out_features, self.hidden_features)
        self.model.eval() # Important for quantization

        # Create dummy calibration data
        # Shape: (batch_size, seq_len, in_features) - typical for NLP tasks
        self.dummy_calibration_data = torch.randn(self.batch_size, self.seq_len, self.in_features)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dummy_calibration_data = self.dummy_calibration_data.to(self.device)

    def test_awq_scale_computation_and_application(self):
        # Instantiate AWQQuantizer
        # group_size chosen to be compatible with hidden_features for fc2's input
        # and in_features for fc1's input if group_size were applied to fc1's weights
        # For this test, group_size = -1 would also work if not testing grouping specifically for fc1.
        # Let's use a group_size that divides in_features for fc1 and hidden_features for fc2
        group_size = 16 # Divides self.in_features (16) and self.hidden_features (32)
        
        quantizer = AWQQuantizer(
            model_name=self.model, # Pass the model instance
            bits=4,
            group_size=group_size, 
            zero_point=True,
            device=self.device
        )

        # --- Collect Activation Stats ---
        # Process a single batch for simplicity in checking act_scales structure
        # _collect_activation_stats expects a single batch
        quantizer._collect_activation_stats(self.dummy_calibration_data[0].unsqueeze(0)) # Pass one sample from batch

        # Assert: Check quantizer.act_scales
        self.assertIn('fc1', quantizer.act_scales)
        self.assertIn('fc2', quantizer.act_scales)

        # For fc1
        fc1_scales_list = quantizer.act_scales['fc1']
        self.assertIsInstance(fc1_scales_list, list)
        self.assertTrue(len(fc1_scales_list) > 0, "fc1_scales_list should not be empty")
        for scale_tensor in fc1_scales_list:
            self.assertIsInstance(scale_tensor, torch.Tensor)
            self.assertEqual(scale_tensor.ndim, 1)
            self.assertEqual(scale_tensor.shape[0], self.in_features) # in_features of fc1

        # For fc2
        fc2_scales_list = quantizer.act_scales['fc2']
        self.assertIsInstance(fc2_scales_list, list)
        self.assertTrue(len(fc2_scales_list) > 0, "fc2_scales_list should not be empty")
        for scale_tensor in fc2_scales_list:
            self.assertIsInstance(scale_tensor, torch.Tensor)
            self.assertEqual(scale_tensor.ndim, 1)
            self.assertEqual(scale_tensor.shape[0], self.hidden_features) # in_features of fc2

        # --- Quantize Layer (Focusing on Scale Application for fc1) ---
        layer_to_quantize_fc1 = self.model.fc1
        act_scale_list_fc1 = quantizer.act_scales.get('fc1')
        self.assertIsNotNone(act_scale_list_fc1)
        self.assertIsInstance(act_scale_list_fc1, list)
        
        # Average the collected scales
        act_scale_tensor_fc1 = torch.stack(act_scale_list_fc1).mean(dim=0)
        
        self.assertEqual(act_scale_tensor_fc1.ndim, 1)
        self.assertEqual(act_scale_tensor_fc1.shape[0], self.in_features)

        try:
            quantized_layer_fc1 = quantizer._quantize_layer(layer_to_quantize_fc1, act_scale_tensor_fc1)
        except RuntimeError as e:
            self.fail(f"_quantize_layer raised RuntimeError unexpectedly: {e}")
        
        self.assertIsInstance(quantized_layer_fc1, QuantizedLinear)

        # --- Full Quantization and Forward Pass (Integration Check) ---
        # Re-instantiate quantizer for a clean full run or clear previous act_scales
        quantizer_full = AWQQuantizer(
            model_name=self.model, # Pass a new copy or re-initialize
            bits=4,
            group_size=group_size,
            zero_point=True,
            device=self.device
        )
        
        try:
            # Use the full calibration dataset and specify steps (can be number of batches)
            quantized_model = quantizer_full.quantize(
                calibration_data=self.dummy_calibration_data, 
                calibration_steps=self.batch_size 
            )
        except Exception as e: # Catch any exception during full quantization
            self.fail(f"quantizer.quantize raised an exception unexpectedly: {e}")

        self.assertIsNotNone(quantized_model)
        # Check if layers are replaced
        self.assertIsInstance(quantized_model.fc1, QuantizedLinear)
        self.assertIsInstance(quantized_model.fc2, QuantizedLinear)

        # Perform a forward pass
        sample_input = self.dummy_calibration_data[0].unsqueeze(0) # Take one sample
        try:
            output = quantized_model(sample_input.to(self.device))
        except RuntimeError as e:
            self.fail(f"Forward pass on quantized_model raised RuntimeError unexpectedly: {e}")

        # Assert output shape
        # Output shape should be (batch_size_sample, seq_len_sample, out_features)
        # For sample_input: (1, self.seq_len, self.out_features)
        self.assertEqual(output.shape, (1, self.seq_len, self.out_features))

if __name__ == '__main__':
    unittest.main()
