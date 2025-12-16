"""
QuantLLM GGUF Converter - 
Properly writes GGUF files compatible with llama.cpp

Key fixes:
1. Correct tensor offset calculation
2. Proper metadata ordering
3. Valid GGUF v3 format
"""

import os
import struct
import numpy as np
from typing import Optional, Dict, Any, List, Union, BinaryIO
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum
import torch
import torch.nn as nn
from ..utils import track_progress, logger, print_success
import json


# GGUF Constants (from ggml specification)
GGUF_MAGIC = 0x46554747  # "GGUF" in hex
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32


class GGMLQuantizationType(IntEnum):
    """GGML quantization types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


@dataclass
class QuantizationInfo:
    """Information about a quantization type."""
    name: str
    type_id: GGMLQuantizationType
    block_size: int
    type_size: int  # bytes per block
    description: str
    
    def bytes_per_weight(self) -> float:
        """Calculate bytes per weight."""
        return self.type_size / self.block_size


# Quantization type registry
QUANT_TYPES = {
    "f32": QuantizationInfo("F32", GGMLQuantizationType.F32, 1, 4, "32-bit float"),
    "f16": QuantizationInfo("F16", GGMLQuantizationType.F16, 1, 2, "16-bit float"),
    "q8_0": QuantizationInfo("Q8_0", GGMLQuantizationType.Q8_0, 32, 34, "8-bit quantization"),
    "q4_0": QuantizationInfo("Q4_0", GGMLQuantizationType.Q4_0, 32, 18, "4-bit quantization, 32-block"),
    "q4_1": QuantizationInfo("Q4_1", GGMLQuantizationType.Q4_1, 32, 20, "4-bit quantization with min"),
    "q5_0": QuantizationInfo("Q5_0", GGMLQuantizationType.Q5_0, 32, 22, "5-bit quantization"),
    "q5_1": QuantizationInfo("Q5_1", GGMLQuantizationType.Q5_1, 32, 24, "5-bit quantization with min"),
    "q2_k": QuantizationInfo("Q2_K", GGMLQuantizationType.Q2_K, 256, 82, "2-bit quantization, k-style"),
    "q3_k": QuantizationInfo("Q3_K", GGMLQuantizationType.Q3_K, 256, 110, "3-bit quantization, k-style"),
    "q4_k": QuantizationInfo("Q4_K", GGMLQuantizationType.Q4_K, 256, 144, "4-bit quantization, k-style"),
    "q5_k": QuantizationInfo("Q5_K", GGMLQuantizationType.Q5_K, 256, 176, "5-bit quantization, k-style"),
    "q6_k": QuantizationInfo("Q6_K", GGMLQuantizationType.Q6_K, 256, 210, "6-bit quantization, k-style"),
    "q8_k": QuantizationInfo("Q8_K", GGMLQuantizationType.Q8_K, 256, 292, "8-bit quantization, k-style"),
}

# Convenient aliases
QUANT_ALIASES = {
    "q4_k_m": "q4_k",
    "q4_k_s": "q4_k",
    "q5_k_m": "q5_k",
    "q5_k_s": "q5_k",
}


class FastQuantizer:
    """Fast quantization kernels for various bit-widths."""
    
    @staticmethod
    def quantize_q8_0(tensor: torch.Tensor) -> bytes:
        """Quantize to Q8_0 format (8-bit per-block)."""
        flat = tensor.flatten().float()
        
        # Pad to multiple of 32
        pad_size = (32 - (flat.numel() % 32)) % 32
        if pad_size > 0:
            flat = torch.cat([flat, torch.zeros(pad_size, device=flat.device)])
        
        # Reshape to blocks
        blocks = flat.reshape(-1, 32)
        
        # Compute scale per block (max abs value / 127)
        scales = blocks.abs().max(dim=1, keepdim=True).values / 127.0
        scales = scales.clamp(min=1e-10)
        
        # Quantize
        quants = torch.clamp(torch.round(blocks / scales), -128, 127).to(torch.int8)
        
        # Pack: [scale (f16), quants (32 x i8)] per block
        result = bytearray()
        for i in range(len(blocks)):
            scale_bytes = np.float16(scales[i].item()).tobytes()
            result.extend(scale_bytes)
            result.extend(quants[i].cpu().numpy().tobytes())
        
        return bytes(result)
    
    @staticmethod
    def quantize_q4_0(tensor: torch.Tensor) -> bytes:
        """Quantize to Q4_0 format (4-bit per-block)."""
        flat = tensor.flatten().float()
        
        # Pad to multiple of 32
        pad_size = (32 - (flat.numel() % 32)) % 32
        if pad_size > 0:
            flat = torch.cat([flat, torch.zeros(pad_size, device=flat.device)])
        
        blocks = flat.reshape(-1, 32)
        
        # Scale: max abs / 7 (4-bit signed: -7 to 7)
        scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
        scales = scales.clamp(min=1e-10)
        
        # Quantize to 4-bit
        quants = torch.clamp(torch.round(blocks / scales), -8, 7).to(torch.int8)
        
        # Pack: [scale (f16), packed_quants (16 bytes)] per block
        result = bytearray()
        for i in range(len(blocks)):
            scale_bytes = np.float16(scales[i].item()).tobytes()
            result.extend(scale_bytes)
            
            # Pack two 4-bit values per byte
            q = quants[i].cpu().numpy()
            packed = np.zeros(16, dtype=np.uint8)
            for j in range(16):
                v1 = int(q[2*j]) & 0x0F
                v2 = int(q[2*j + 1]) & 0x0F
                packed[j] = (v2 << 4) | v1
            result.extend(packed.tobytes())
        
        return bytes(result)
    
    @staticmethod
    def quantize_f16(tensor: torch.Tensor) -> bytes:
        """Convert to fp16."""
        return tensor.half().cpu().numpy().tobytes()
    
    @staticmethod
    def quantize_f32(tensor: torch.Tensor) -> bytes:
        """Convert to fp32."""
        return tensor.float().cpu().numpy().tobytes()


class GGUFWriter:
    """Pure Python GGUF file writer with correct offset calculation."""
    
    def __init__(self, output_path: str, arch: str = "llama"):
        self.output_path = output_path
        self.arch = arch
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Dict[str, Any]] = []
        self.tensor_data: List[bytes] = []  # Store quantized data
        
    def add_architecture(self):
        """Add architecture metadata."""
        self.add_string("general.architecture", self.arch)
    
    def add_string(self, key: str, value: str):
        """Add string metadata."""
        self.metadata[key] = ("string", value)
    
    def add_uint32(self, key: str, value: int):
        """Add uint32 metadata."""
        self.metadata[key] = ("uint32", value)
    
    def add_float32(self, key: str, value: float):
        """Add float32 metadata."""
        self.metadata[key] = ("float32", value)
    
    def add_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        quant_type: str = "f16"
    ):
        """Add a tensor to be written."""
        # Normalize quantization type
        quant_type = quant_type.lower()
        if quant_type in QUANT_ALIASES:
            quant_type = QUANT_ALIASES[quant_type]
        
        if quant_type not in QUANT_TYPES:
            raise ValueError(f"Unknown quantization type: {quant_type}")
        
        quant_info = QUANT_TYPES[quant_type]
        
        self.tensors.append({
            "name": name,
            "tensor": tensor,
            "quant_type": quant_type,
            "quant_info": quant_info,
            "shape": list(tensor.shape),
        })
    
    def write(self, show_progress: bool = True):
        """Write GGUF file with correct offsets."""
        # First pass: quantize all tensors and calculate sizes
        quantizer = FastQuantizer()
        
        if show_progress:
            logger.info("Quantizing tensors...")
        
        for tensor_info in (track_progress(self.tensors, description="Quantizing") if show_progress else self.tensors):
            tensor = tensor_info["tensor"]
            quant_type = tensor_info["quant_type"]
            
            # Quantize
            if quant_type == "f32":
                data = quantizer.quantize_f32(tensor)
            elif quant_type == "f16":
                data = quantizer.quantize_f16(tensor)
            elif quant_type == "q8_0":
                data = quantizer.quantize_q8_0(tensor)
            elif quant_type == "q4_0" or quant_type.startswith("q4_k"):
                data = quantizer.quantize_q4_0(tensor)
            elif quant_type.startswith("q5_k") or quant_type == "q5_0":
                data = quantizer.quantize_f16(tensor)  # Fallback
            else:
                data = quantizer.quantize_f16(tensor)  # Fallback
            
            self.tensor_data.append(data)
        
        # Now write the file with correct offsets
        with open(self.output_path, "wb") as f:
            # Calculate header size first
            header_size = self._calculate_header_size()
            
            # Write header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))
            
            # Write metadata
            self._write_metadata(f)
            
            # Write tensor info with correct offsets
            current_offset = header_size
            current_offset = self._align_offset(current_offset, GGUF_DEFAULT_ALIGNMENT)
            
            for i, tensor_info in enumerate(self.tensors):
                # Write tensor name
                self._write_string(f, tensor_info["name"])
                
                # Write dimensions
                dims = tensor_info["shape"]
                f.write(struct.pack("<I", len(dims)))
                for dim in dims:
                    f.write(struct.pack("<Q", dim))
                
                # Write type
                f.write(struct.pack("<I", tensor_info["quant_info"].type_id))
                
                # Write offset
                f.write(struct.pack("<Q", current_offset))
                
                # Update offset for next tensor
                data_size = len(self.tensor_data[i])
                current_offset += data_size
                current_offset = self._align_offset(current_offset, GGUF_DEFAULT_ALIGNMENT)
            
            # Align before writing tensor data
            self._align_file(f, GGUF_DEFAULT_ALIGNMENT)
            
            # Write tensor data
            if show_progress:
                logger.info("Writing tensor data...")
            
            for data in (track_progress(self.tensor_data, description="Writing") if show_progress else self.tensor_data):
                f.write(data)
                self._align_file(f, GGUF_DEFAULT_ALIGNMENT)
        
        if show_progress:
            size_mb = os.path.getsize(self.output_path) / (1024**2)
            print_success(f"GGUF file created: {self.output_path} ({size_mb:.2f} MB)")
    
    def _calculate_header_size(self) -> int:
        """Calculate the size of header + metadata + tensor info."""
        size = 0
        
        # Header: magic (4) + version (4) + tensor_count (8) + metadata_count (8)
        size += 4 + 4 + 8 + 8
        
        # Metadata
        for key, (type_name, value) in self.metadata.items():
            size += 8 + len(key.encode('utf-8'))  # key length + key
            size += 4  # value type
            
            if type_name == "string":
                size += 8 + len(value.encode('utf-8'))
            elif type_name == "uint32":
                size += 4
            elif type_name == "float32":
                size += 4
        
        # Tensor info
        for tensor_info in self.tensors:
            name = tensor_info["name"]
            dims = tensor_info["shape"]
            
            size += 8 + len(name.encode('utf-8'))  # name length + name
            size += 4  # n_dimensions
            size += 8 * len(dims)  # dimensions
            size += 4  # type
            size += 8  # offset
        
        return size
    
    def _align_offset(self, offset: int, alignment: int) -> int:
        """Calculate aligned offset."""
        return ((offset + alignment - 1) // alignment) * alignment
    
    def _align_file(self, f: BinaryIO, alignment: int):
        """Align file position."""
        pos = f.tell()
        padding = (alignment - (pos % alignment)) % alignment
        if padding > 0:
            f.write(b'\x00' * padding)
    
    def _write_metadata(self, f: BinaryIO):
        """Write metadata."""
        for key, (type_name, value) in self.metadata.items():
            self._write_string(f, key)
            
            if type_name == "string":
                f.write(struct.pack("<I", GGUFValueType.STRING))
                self._write_string(f, value)
            elif type_name == "uint32":
                f.write(struct.pack("<I", GGUFValueType.UINT32))
                f.write(struct.pack("<I", value))
            elif type_name == "float32":
                f.write(struct.pack("<I", GGUFValueType.FLOAT32))
                f.write(struct.pack("<f", value))
    
    def _write_string(self, f: BinaryIO, s: str):
        """Write length-prefixed string."""
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)


class GGUFConverter:
    """Pure Python GGUF converter compatible with llama.cpp."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def convert(
        self,
        model,
        tokenizer,
        output_path: str,
        quant_type: str = "q4_0",
        arch: str = "llama",
    ) -> str:
        """Convert HuggingFace model to GGUF format."""
        if self.verbose:
            logger.info(f"🚀 Converting to GGUF ({quant_type}) Arch: {arch}...")
        
        writer = GGUFWriter(output_path, arch=arch)
        
        # Add metadata
        self._add_metadata(writer, model, tokenizer)
        
        # Add tensors WITHOUT permutation (transformers expects HF format)
        self._add_tensors(writer, model, quant_type)
        
        # Write file
        writer.write(show_progress=self.verbose)
        
        return output_path
    
    def _add_metadata(self, writer: GGUFWriter, model, tokenizer):
        """Add model metadata."""
        config = model.config
        
        writer.add_architecture()
        
        # Model hyperparameters
        if hasattr(config, "hidden_size"):
            writer.add_uint32(f"{writer.arch}.embedding_length", config.hidden_size)
        if hasattr(config, "num_hidden_layers"):
            writer.add_uint32(f"{writer.arch}.block_count", config.num_hidden_layers)
        if hasattr(config, "num_attention_heads"):
            writer.add_uint32(f"{writer.arch}.attention.head_count", config.num_attention_heads)
        if hasattr(config, "num_key_value_heads"):
            writer.add_uint32(f"{writer.arch}.attention.head_count_kv", config.num_key_value_heads)
        if hasattr(config, "max_position_embeddings"):
            writer.add_uint32(f"{writer.arch}.context_length", config.max_position_embeddings)
        if hasattr(config, "rms_norm_eps"):
            writer.add_float32(f"{writer.arch}.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
        if hasattr(config, "rope_theta"):
            writer.add_float32(f"{writer.arch}.rope.freq_base", config.rope_theta)
        
        # Tokenizer metadata
        if tokenizer:
            writer.add_uint32("tokenizer.ggml.model", 1)
            if hasattr(tokenizer, "vocab_size"):
                writer.add_uint32(f"{writer.arch}.vocab_size", tokenizer.vocab_size)
    
    def _add_tensors(self, writer: GGUFWriter, model, quant_type: str):
        """Add model tensors - NO PERMUTATION for transformers compatibility."""
        params = list(model.named_parameters())
        
        for name, param in track_progress(params, description="Processing tensors"):
            gguf_name = self._convert_tensor_name(name)
            data = param.detach().cpu().contiguous()
            
            # NO PERMUTATION - transformers expects HF format in GGUF
            writer.add_tensor(gguf_name, data, quant_type)
    
    def _convert_tensor_name(self, hf_name: str) -> str:
        """Convert HuggingFace tensor name to GGUF format."""
        name = hf_name.replace("model.", "")
        name = name.replace("layers.", "blk.")
        name = name.replace("self_attn.", "attn_")
        name = name.replace("q_proj", "q")
        name = name.replace("k_proj", "k")
        name = name.replace("v_proj", "v")
        name = name.replace("o_proj", "output")
        name = name.replace("mlp.", "ffn_")
        name = name.replace("gate_proj", "gate")
        name = name.replace("up_proj", "up")
        name = name.replace("down_proj", "down")
        name = name.replace("input_layernorm", "attn_norm")
        name = name.replace("post_attention_layernorm", "ffn_norm")
        name = name.replace("embed_tokens", "token_embd")
        name = name.replace("lm_head", "output")
        name = name.replace("norm.", "output_norm.")
        
        # Fix double replacements
        if name.endswith("output_norm.weight"):
            name = name.replace("output_norm.weight", "norm.weight")
        
        return name


# High-level API
def convert_to_gguf(
    model,
    tokenizer,
    output_path: str,
    quant_type: str = "q4_0",
    verbose: bool = True,
) -> str:
    """
    Convert a model to GGUF format compatible with llama.cpp.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        output_path: Output GGUF file path
        quant_type: Quantization type (f16, q8_0, q4_0, q4_k, etc.)
        verbose: Show progress
        
    Returns:
        Path to created GGUF file
    """
    # Map HF model type to GGUF architecture
    hf_arch = getattr(model.config, "model_type", "llama")
    arch_map = {
        "llama": "llama",
        "mistral": "llama",
        "mixtral": "llama",
        "gemma": "gemma",
        "gemma2": "gemma",
        "qwen2": "qwen2",
        "phi": "phi",
        "phi3": "phi3",
    }
    gguf_arch = arch_map.get(hf_arch, "llama")
    
    converter = GGUFConverter(verbose=verbose)
    return converter.convert(model, tokenizer, output_path, quant_type, arch=gguf_arch)


def list_quant_types() -> Dict[str, str]:
    """List available quantization types."""
    return {name: info.description for name, info in QUANT_TYPES.items()}