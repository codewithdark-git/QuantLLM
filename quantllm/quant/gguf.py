"""GGUF (GGML Universal Format) quantization implementation."""

import gc
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import PreTrainedModel
from .quantization_engine import move_to_device, BaseQuantizer, QuantizationConfig, QuantizedLinear
from ..utils.logger import logger
from ..utils.memory_tracker import memory_tracker

try:
    import ctransformers
    CT_AVAILABLE = True
except ImportError:
    CT_AVAILABLE = False

SUPPORTED_GGUF_BITS = [2, 3, 4, 5, 6, 8]
SUPPORTED_GGUF_TYPES = {
    2: ["Q2_K"],
    3: ["Q3_K_S", "Q3_K_M", "Q3_K_L"],
    4: ["Q4_K_S", "Q4_K_M"],
    5: ["Q5_K_S", "Q5_K_M"],
    6: ["Q6_K"],
    8: ["Q8_0"]
}

class GGUFQuantizer(BaseQuantizer):
    """
    GGUF-specific quantizer implementation optimized for memory efficiency and performance.
    Focuses exclusively on GGUF format, removing AWQ/GPTQ specific logic.
    """
    
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel],
        bits: int = 4,
        group_size: int = 32,
        desc_act: bool = False,
        desc_ten: bool = False,
        use_packed: bool = True,
        legacy_format: bool = False,
        batch_size: int = 4,
        device: Optional[Union[str, torch.device]] = None,
        cpu_offload: bool = False,
        quant_type: Optional[str] = None
    ):
        if not CT_AVAILABLE:
            raise ImportError("CTransformers is required for GGUF quantization. Install with: pip install ctransformers")

        if bits not in SUPPORTED_GGUF_BITS:
            raise ValueError(f"Unsupported bits for GGUF: {bits}. Supported values are {SUPPORTED_GGUF_BITS}")
        
        if quant_type and quant_type not in SUPPORTED_GGUF_TYPES.get(bits, []):
            raise ValueError(f"Unsupported GGUF quantization type {quant_type} for {bits} bits. "
                           f"Supported types: {SUPPORTED_GGUF_TYPES.get(bits, [])}")
        
        super().__init__(model_name=model_name, bits=bits, device=device)

        self.group_size = group_size 
        self.desc_act = desc_act
        self.desc_ten = desc_ten
        self.use_packed = use_packed
        self.legacy_format = legacy_format
        self.batch_size = batch_size
        self.cpu_offload = cpu_offload
        self.quant_type = quant_type or self._get_default_quant_type(bits)

        if self.device_manager.primary_device is None:
            self.device_manager.determine_primary_device()
            logger.log_info(f"Primary device for GGUF operations: {self.device_manager.primary_device}")

    def _get_default_quant_type(self, bits: int) -> str:
        """Select optimal GGUF quantization type based on bit width."""
        if bits in SUPPORTED_GGUF_TYPES:
            # Prefer middle options for better balance of size/quality
            types = SUPPORTED_GGUF_TYPES[bits]
            return types[len(types)//2] if len(types) > 1 else types[0]
        raise ValueError(f"No supported GGUF types for {bits} bits")

    def quantize(
        self,
        calibration_data: Optional[torch.Tensor] = None
    ) -> PreTrainedModel:
        """
        Quantize model using GGUF format with optimized memory handling.
        Focuses exclusively on GGUF quantization, removing AWQ/GPTQ specific logic.

        Args:
            calibration_data: Optional tensor for calibration (not typically needed for GGUF).

        Returns:
            PreTrainedModel: Quantized model with GGUF-specific parameters.

        Raises:
            ImportError: If ctransformers is not available.
            RuntimeError: If quantization fails.
        """
        if not CT_AVAILABLE:
            raise ImportError("CTransformers is required for GGUF quantization")

        try:
            logger.log_info("Starting GGUF quantization process...")
            memory_tracker.log_memory("gguf_quantization_start")
            
            # Prepare model for quantization
            if not hasattr(self.model, '_prepared_for_quantization'): 
                self.model = self._prepare_model(self.model) 
            
            # Move model to appropriate device
            device = torch.device('cpu') if self.cpu_offload else self.device_manager.primary_device
            if self.model.device != device:
                logger.log_info(f"Moving model to {device} for quantization")
                self.model.to(device)
                self._clear_memory()
                memory_tracker.log_memory("model_moved_to_device")
            
            self.model.eval()
            
            # Process layers
            logger.log_info("Starting layer-by-layer GGUF quantization...")
            modules_to_quantize = [
                (name, module) for name, module in self.model.named_modules() 
                if isinstance(module, nn.Linear)
            ]
            
            total_layers = len(modules_to_quantize)
            for idx, (name, module) in enumerate(modules_to_quantize, 1):
                try:
                    logger.log_info(f"Quantizing layer {idx}/{total_layers}: {name}")
                    memory_tracker.log_memory(f"before_layer_{idx}")
                    
                    quantized_layer = self._quantize_layer(module)
                    
                    # Update model with quantized layer
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = self.model.get_submodule(parent_name)
                        setattr(parent, child_name, quantized_layer)
                    else:
                        setattr(self.model, name, quantized_layer)
                    
                    self._clear_memory()
                    memory_tracker.log_memory(f"after_layer_{idx}")
                    
                except Exception as e:
                    logger.log_error(f"Failed to quantize layer {name}: {str(e)}")
                    raise RuntimeError(f"GGUF quantization failed at layer {name}: {str(e)}") from e

            # Update model config with GGUF parameters
            logger.log_info("Updating model configuration with GGUF parameters...")
            gguf_config = {
                "format": "gguf",
                "bits": self.bits,
                "group_size": self.group_size,
                "quant_type": self.quant_type,
                "format_config": {
                "desc_act": self.desc_act,
                "desc_ten": self.desc_ten,
                    "use_packed": self.use_packed,
                    "legacy_format": self.legacy_format
                }
            }
            self._update_model_config_with_quant_params("gguf", gguf_config)

            memory_tracker.log_memory("gguf_quantization_complete")
            logger.log_info("GGUF quantization completed successfully")
            return self.model

        except Exception as e: 
            logger.log_error(f"GGUF quantization failed: {str(e)}")
            raise RuntimeError(f"GGUF quantization failed: {str(e)}") from e
        finally:
            self._clear_memory()
        
    def _quantize_layer(
        self,
        layer: nn.Linear,
        stats: Optional[Dict[str, torch.Tensor]] = None
    ) -> QuantizedLinear:
        """
        Quantize a single linear layer to GGUF format.
        Implements GGUF-specific quantization with improved memory handling.
        """
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear layer, got {type(layer)}")

        device = torch.device('cpu') if self.cpu_offload else self.device_manager.primary_device
        memory_tracker.log_memory("layer_quantization_start")
        
        try:
            # Move layer to processing device
            layer = move_to_device(layer, device)
            memory_tracker.log_memory("layer_moved_to_device")
            
            # Get weight tensor
            weight = layer.weight.data
            
            # Calculate scales and zero points
            if self.group_size > 0:
                # Group-wise quantization
                num_groups = weight.shape[-1] // self.group_size
                weight_groups = weight.view(-1, num_groups, self.group_size)
                scales = weight_groups.amax(dim=-1, keepdim=True)
                weight_scaled = weight_groups / scales.clamp(min=1e-5)
                zeros = torch.zeros_like(scales)  # GGUF uses symmetric quantization
            else:
                # Per-tensor quantization
                scale = weight.abs().max()
                weight_scaled = weight / scale.clamp(min=1e-5)
                scales = scale.expand(1)
                zeros = torch.zeros_like(scales)

            memory_tracker.log_memory("scales_computed")

            # Quantize weights
            qweight = torch.clamp(
                torch.round(weight_scaled * (2**(self.bits-1) - 1)),
                -2**(self.bits-1),
                2**(self.bits-1) - 1
            ).to(torch.int8)

            memory_tracker.log_memory("weights_quantized")

            # Create quantized layer
            qlayer = QuantizedLinear(
                in_features=layer.in_features,
                out_features=layer.out_features,
                bias=layer.bias is not None,
                config=QuantizationConfig(
                    bits=self.bits,
                    scheme="symmetric",
                    granularity="per-group" if self.group_size > 0 else "per-tensor",
                    format="gguf",
                    format_config={
                        "type": self.quant_type,
                        "group_size": self.group_size,
                        "is_packed": self.use_packed
                    }
                )
            )

            # Store quantized weights and parameters
            qlayer.weight_quantized = qweight
            qlayer.weight_scale = 1.0 / scales
            qlayer.weight_zero_point = zeros
            if layer.bias is not None:
                qlayer.bias = layer.bias.data.clone()

            memory_tracker.log_memory("layer_quantization_complete")
            return qlayer

        except Exception as e:
            logger.log_error(f"Failed to quantize layer: {str(e)}")
            raise RuntimeError(f"GGUF layer quantization failed: {str(e)}") from e
        finally:
            # Clean up original layer if it was moved
            if layer.device != device:
                del layer
            torch.cuda.empty_cache()
            gc.collect()

    def convert_to_gguf(self, output_path: str):
        """
        Convert quantized model to GGUF format using ctransformers.
        """
        if not CT_AVAILABLE:
            raise ImportError("CTransformers is required for GGUF conversion")
        
        try:
            logger.log_info(f"Converting model to GGUF format: {output_path}")
            logger.log_info(f"Using quantization type: {self.quant_type}")
            memory_tracker.log_memory("gguf_conversion_start")
            
            # Ensure model is on CPU for conversion
            if not self.cpu_offload:
                self.model.to('cpu')
                memory_tracker.log_memory("model_moved_to_cpu")
            
            # Prepare GGUF conversion config
            config = {
                "quantization": {
                    "bits": self.bits,
                    "type": self.quant_type,
                    "group_size": self.group_size if self.group_size > 0 else None,
                },
                "metadata": {
                    "description": "Model quantized using QuantLLM GGUF quantizer",
                    "format_version": "legacy" if self.legacy_format else "latest",
                    "has_act_desc": self.desc_act,
                    "has_tensor_desc": self.desc_ten
                }
            }
            
            # Convert using ctransformers
            ctransformers.convert(
                self.model,
                output_path,
                config=config,
                legacy=self.legacy_format
            )
            
            memory_tracker.log_memory("gguf_conversion_complete")
            logger.log_info("GGUF conversion completed successfully")
            
        except Exception as e:
            logger.log_error(f"GGUF conversion failed: {str(e)}")
            raise RuntimeError(f"Failed to convert model to GGUF: {str(e)}") from e
        finally:
            self._clear_memory()
    
    def _clear_memory(self):
        """Enhanced memory cleanup for GGUF operations."""
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        memory_tracker.clear_memory()

