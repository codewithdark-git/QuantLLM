"""Tests for HQQ (Half-Quadratic Quantization)."""

import pytest
import torch
import torch.nn as nn

from quantllm.quant.hqq import HQQLinear, HQQQuantizer, HQQConfig, HQQ_QUANT_TYPES


class TestHQQLinear:
    def test_quantize_dequantize_4bit(self):
        layer = HQQLinear(64, 128, bits=4, group_size=32)
        weight = torch.randn(128, 64)
        layer.quantize(weight)
        deq = layer.dequantize()
        assert deq.shape == (128, 64)
        assert deq.dtype == torch.float16
        assert not torch.isnan(deq).any()

    def test_quantize_dequantize_8bit(self):
        layer = HQQLinear(64, 128, bits=8, group_size=32)
        weight = torch.randn(128, 64)
        layer.quantize(weight)
        deq = layer.dequantize()
        assert deq.shape == (128, 64)
        # 8-bit should be very close
        mse = ((deq.float() - weight.float()) ** 2).mean()
        assert mse < 0.01

    def test_quantize_dequantize_2bit(self):
        layer = HQQLinear(64, 128, bits=2, group_size=32)
        weight = torch.randn(128, 64)
        layer.quantize(weight)
        deq = layer.dequantize()
        assert deq.shape == (128, 64)

    def test_forward_pass(self):
        layer = HQQLinear(64, 128, bits=4, group_size=32, bias=True)
        weight = torch.randn(128, 64)
        layer.quantize(weight)
        layer.bias = nn.Parameter(torch.randn(128, dtype=torch.float16))
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 128)
        assert not torch.isnan(out).any()

    def test_no_bias(self):
        layer = HQQLinear(64, 128, bits=4, group_size=32, bias=False)
        assert layer.bias is None


class TestHQQQuantizer:
    def test_quantize_model(self):
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        quantizer = HQQQuantizer(HQQConfig(bits=4, group_size=32))
        qmodel = quantizer.quantize_model(model)
        assert isinstance(qmodel[0], HQQLinear)
        assert isinstance(qmodel[2], HQQLinear)

    def test_estimate_size(self):
        size = HQQQuantizer.estimate_size(1_000_000_000, 4)
        assert size > 0
        assert size < 10  # 1B params at 4-bit should be ~0.5 GB


class TestHQQConfig:
    def test_default_config(self):
        cfg = HQQConfig()
        assert cfg.bits == 4
        assert cfg.group_size == 128

    def test_custom_config(self):
        cfg = HQQConfig(bits=8, group_size=64)
        assert cfg.bits == 8
        assert cfg.group_size == 64
