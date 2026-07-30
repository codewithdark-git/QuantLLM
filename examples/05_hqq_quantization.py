"""
QuantLLM v2.2 -- HQQ Native Quantization

Demonstrates HQQ quantization on a single linear layer and
on a full small model, using the actual package APIs.
"""

import torch
from quantllm import HQQLinear, HQQQuantizer, HQQConfig

# Configure HQQ
config = HQQConfig(nbits=4, group_size=64)
print(f"HQQ config: nbits={config.bits}, group_size={config.group_size}")

# ------------------------------------------------------------------
# Quantise a single linear layer
# ------------------------------------------------------------------
print("\nQuantising a single linear layer...")
weight = torch.randn(256, 512)
bias = torch.randn(256)

hqq_layer = HQQLinear.quantize(weight, config=config, bias=bias)
print(f"  Input shape:  {list(weight.shape)}")
print(f"  Output shape: {list(hqq_layer.W.shape)}")
print(f"  Forward pass: {list(hqq_layer(torch.randn(8, 512)).shape)}")
print(f"  Dequantised:  {list(hqq_layer.dequantize().shape)}")

# Size comparison
orig = weight.numel() * 4
quant = HQQQuantizer.estimate_size(weight, config)
print(f"  Size: {orig/1024:.0f} KB -> {quant/1024:.0f} KB ({quant/orig:.3f}x)")

# ------------------------------------------------------------------
# Quantise a full model
# ------------------------------------------------------------------
print("\nQuantising a full model...")

class DemoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = DemoModel()
print(f"  Original params: {sum(p.numel() for p in model.parameters())}")

quantizer = HQQQuantizer(config=config)
qmodel = quantizer.quantize_model(model)
x = torch.randn(4, 128)
print(f"  Quantised forward: {list(qmodel(x).shape)}")

print("\nHQQ example complete.")
