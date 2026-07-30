"""Tests for CPU kernels."""

import pytest
import torch
import torch.nn as nn

from quantllm.kernels.cpu import CPUInt4Linear, CPUInt8Linear, optimize_for_cpu, is_cpu_kernel_available


class TestCPUInt4Linear:
    def test_quantize_dequantize(self):
        layer = CPUInt4Linear(64, 128, bits=4, group_size=32)
        weight = torch.randn(128, 64)
        layer.quantize(weight)
        deq = layer._dequantize_weights()
        assert deq.shape == (128, 64)

    def test_forward(self):
        layer = CPUInt4Linear(64, 128, bits=4, group_size=32, bias=True)
        weight = torch.randn(128, 64)
        layer.quantize(weight)
        layer.bias = nn.Parameter(torch.randn(128, dtype=torch.float32))
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 128)


class TestCPUInt8Linear:
    def test_quantize_dequantize(self):
        layer = CPUInt8Linear(64, 128)
        weight = torch.randn(128, 64)
        layer.quantize(weight)
        w_deq = layer.qweight.float() * layer.scale
        assert w_deq.shape == (128, 64)

    def test_forward(self):
        layer = CPUInt8Linear(64, 128, bias=True)
        weight = torch.randn(128, 64)
        layer.quantize(weight)
        layer.bias = nn.Parameter(torch.randn(128, dtype=torch.float32))
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 128)


class TestOptimizeCPU:
    def test_optimize_model_int4(self):
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        qmodel = optimize_for_cpu(model, bits=4, group_size=32)
        assert isinstance(qmodel[0], CPUInt4Linear)
        assert isinstance(qmodel[2], CPUInt4Linear)

    def test_optimize_model_int8(self):
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        qmodel = optimize_for_cpu(model, bits=8)
        assert isinstance(qmodel[0], CPUInt8Linear)
        assert isinstance(qmodel[2], CPUInt8Linear)
