"""Tests for quantization methods in QuantLLM."""

import pytest
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from quantllm.quant import (
    GPTQQuantizer,
    AWQQuantizer,
    GGUFQuantizer,
    QuantizationConfig
)
from quantllm import Model, ModelConfig

@pytest.fixture
def model():
    """Fixture for loading a small model for testing."""
    model_config = ModelConfig(model_name="facebook/opt-125m")  # Using smaller model for tests
    return Model(model_config).get_model().cpu()

@pytest.fixture
def calibration_data():
    """Fixture for preparing small calibration dataset."""
    def _prepare_data(model_name="facebook/opt-125m", num_samples=4):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        dataset = load_dataset("c4", "en", split="train", streaming=True)
        data = []
        
        max_length = 64  # Short sequences for testing
        
        for item in dataset.take(num_samples):
            encoded = tokenizer(
                item["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            data.append(encoded["input_ids"])
            
        return torch.cat(data, dim=0).cpu()
    
    return _prepare_data()

def test_gptq_quantizer(model, calibration_data):
    """Test GPTQ quantization."""
    quantizer = GPTQQuantizer(
        model=model,
        bits=4,
        group_size=128,
        actorder=False,  # Disable for faster testing
        use_triton=False
    )
    
    try:
        # Verify quantization runs without error
        quantized_model = quantizer.quantize(calibration_data=calibration_data)
        
        # Basic validation checks
        assert quantized_model is not None
        assert next(quantized_model.parameters()).dtype == torch.int8  # Quantized weights
        
        # Verify model can still do inference
        test_input = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            output = quantized_model(test_input)
        assert output.logits.shape[1] == model.config.vocab_size
        
    except Exception as e:
        pytest.fail(f"GPTQ quantization failed: {str(e)}")
    
def test_awq_quantizer(model, calibration_data):
    """Test AWQ quantization."""
    quantizer = AWQQuantizer(
        model=model,
        bits=4,
        group_size=128,
        zero_point=True
    )
    
    try:
        # Verify quantization runs without error
        quantized_model = quantizer.quantize(
            calibration_data=calibration_data,
            calibration_steps=2  # Minimal steps for testing
        )
        
        # Basic validation
        assert quantized_model is not None
        for name, module in quantized_model.named_modules():
            if hasattr(module, 'weight_quantized'):
                assert module.weight_quantized.dtype == torch.int8
                assert hasattr(module, 'weight_scale')
        
        # Inference check
        test_input = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            output = quantized_model(test_input)
        assert output.logits.shape[1] == model.config.vocab_size
        
    except Exception as e:
        pytest.fail(f"AWQ quantization failed: {str(e)}")

def test_gguf_quantizer(model, calibration_data):
    """Test GGUF quantization."""
    quantizer = GGUFQuantizer(
        model=model,
        bits=4,
        group_size=32,
        use_packed=True
    )
    
    try:
        # Test quantization
        quantized_model = quantizer.quantize(calibration_data=calibration_data)
        
        # Validation
        assert quantized_model is not None
        
        # Test inference
        test_input = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            output = quantized_model(test_input)
        assert output.logits.shape[1] == model.config.vocab_size
        
        # Test GGUF export
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.gguf') as tmp:
            quantizer.convert_to_gguf(tmp.name)
            # Verify file exists and has content
            assert tmp.tell() > 0
            
    except Exception as e:
        pytest.fail(f"GGUF quantization failed: {str(e)}")

def test_invalid_parameters():
    """Test error handling for invalid quantization parameters."""
    model_config = ModelConfig(model_name="facebook/opt-125m")
    model = Model(model_config).get_model().cpu()
    
    # Test invalid bits
    with pytest.raises(ValueError):
        GPTQQuantizer(model=model, bits=16)  # Bits must be <= 8
    
    with pytest.raises(ValueError):
        AWQQuantizer(model=model, bits=0)  # Bits must be positive
        
    # Test missing calibration data
    quantizer = GPTQQuantizer(model=model, bits=4)
    with pytest.raises(ValueError):
        quantizer.quantize(calibration_data=None)  # Calibration data required
        
def test_quantization_consistency(model, calibration_data):
    """Test consistency of quantization across multiple runs."""
    config = {
        'bits': 4,
        'group_size': 128
    }
    
    def get_model_outputs(quantizer_class):
        outputs = []
        for _ in range(2):  # Run twice to check consistency
            quantizer = quantizer_class(model=model.clone(), **config)
            quantized = quantizer.quantize(calibration_data=calibration_data)
            
            test_input = torch.randint(0, 1000, (1, 32))
            with torch.no_grad():
                output = quantized(test_input)
            outputs.append(output.logits)
        return outputs
    
    # Test GPTQ consistency
    gptq_outputs = get_model_outputs(GPTQQuantizer)
    assert torch.allclose(gptq_outputs[0], gptq_outputs[1], rtol=1e-3)
    
    # Test AWQ consistency
    awq_outputs = get_model_outputs(lambda **kwargs: AWQQuantizer(**kwargs, zero_point=True))
    assert torch.allclose(awq_outputs[0], awq_outputs[1], rtol=1e-3)

def test_memory_usage(model, calibration_data):
    """Test memory efficiency during quantization."""
    def measure_peak_memory(fn):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
            
        torch.cuda.reset_peak_memory_stats()
        fn()
        return torch.cuda.max_memory_allocated()
    
    # Test GPTQ memory
    gptq_mem = measure_peak_memory(
        lambda: GPTQQuantizer(model=model.cuda(), bits=4).quantize(
            calibration_data=calibration_data.cuda()
        )
    )
    
    # Test AWQ memory
    awq_mem = measure_peak_memory(
        lambda: AWQQuantizer(model=model.cuda(), bits=4).quantize(
            calibration_data=calibration_data.cuda(),
            calibration_steps=2
        )
    )
    
    # AWQ should use less peak memory than GPTQ due to batched processing
    assert awq_mem <= gptq_mem * 1.2  # Allow 20% margin
