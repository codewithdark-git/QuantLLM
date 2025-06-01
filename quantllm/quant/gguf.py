"""GGUF (GGML Universal Format) quantization implementation."""

import gc
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import bitsandbytes as bnb
from .quantization_engine import move_to_device, BaseQuantizer, QuantizationConfig
from ..utils.logger import logger
import time
from tqdm.auto import tqdm

try:
    import ctransformers
    from ctransformers import AutoModelForCausalLM as CTAutoModel
    CT_AVAILABLE = True
except ImportError:
    CT_AVAILABLE = False

# Updated GGUF quantization types with detailed configurations
SUPPORTED_GGUF_TYPES = {
    2: {
        "Q2_K": {
            "description": "Uses Q4_K for attention.vw and feed_forward.w2, Q2_K for others",
            "tensor_configs": {
                "attention.wv": "Q4_K",
                "feed_forward.w2": "Q4_K",
                "default": "Q2_K"
            }
        }
    },
    3: {
        "Q3_K_L": {
            "description": "Uses Q5_K for attention.wv, attention.wo, and feed_forward.w2, Q3_K for others",
            "tensor_configs": {
                "attention.wv": "Q5_K",
                "attention.wo": "Q5_K",
                "feed_forward.w2": "Q5_K",
                "default": "Q3_K"
            }
        },
        "Q3_K_M": {
            "description": "Uses Q4_K for attention.wv, attention.wo, and feed_forward.w2, Q3_K for others",
            "tensor_configs": {
                "attention.wv": "Q4_K",
                "attention.wo": "Q4_K",
                "feed_forward.w2": "Q4_K",
                "default": "Q3_K"
            }
        },
        "Q3_K_S": {
            "description": "Uses Q3_K for all tensors",
            "tensor_configs": {
                "default": "Q3_K"
            }
        }
    },
    4: {
        "Q4_0": {
            "description": "Original quant method, 4-bit",
            "tensor_configs": {
                "default": "Q4_0"
            }
        },
        "Q4_1": {
            "description": "Higher accuracy than Q4_0, quicker inference than Q5",
            "tensor_configs": {
                "default": "Q4_1"
            }
        },
        "Q4_K_M": {
            "description": "Uses Q6_K for half of attention.wv and feed_forward.w2, Q4_K for others",
            "tensor_configs": {
                "attention.wv": ["Q6_K", "Q4_K"],  # Split tensors
                "feed_forward.w2": ["Q6_K", "Q4_K"],  # Split tensors
                "default": "Q4_K"
            }
        },
        "Q4_K_S": {
            "description": "Uses Q4_K for all tensors",
            "tensor_configs": {
                "default": "Q4_K"
            }
        }
    },
    5: {
        "Q5_0": {
            "description": "Higher accuracy, higher resource usage and slower inference",
            "tensor_configs": {
                "default": "Q5_0"
            }
        },
        "Q5_1": {
            "description": "Even higher accuracy, resource usage and slower inference",
            "tensor_configs": {
                "default": "Q5_1"
            }
        },
        "Q5_K_M": {
            "description": "Uses Q6_K for half of attention.wv and feed_forward.w2, Q5_K for others",
            "tensor_configs": {
                "attention.wv": ["Q6_K", "Q5_K"],  # Split tensors
                "feed_forward.w2": ["Q6_K", "Q5_K"],  # Split tensors
                "default": "Q5_K"
            }
        },
        "Q5_K_S": {
            "description": "Uses Q5_K for all tensors",
            "tensor_configs": {
                "default": "Q5_K"
            }
        }
    },
    6: {
        "Q6_K": {
            "description": "Uses Q8_K for all tensors",
            "tensor_configs": {
                "default": "Q8_K"
            }
        }
    },
    8: {
        "Q8_0": {
            "description": "Almost indistinguishable from float16. High resource use and slow",
            "tensor_configs": {
                "default": "Q8_0"
            }
        }
    }
}

# List of supported bits
SUPPORTED_GGUF_BITS = list(SUPPORTED_GGUF_TYPES.keys())

class GGUFQuantizer(BaseQuantizer):
    """
    GGUF-specific quantizer implementation using BitsAndBytes for efficient loading
    and direct GGUF conversion.
    """
    
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel],
        bits: int = 4,
        group_size: int = 32,
        use_packed: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        quant_type: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: torch.dtype = torch.float16,
        bnb_4bit_use_double_quant: bool = True,
        use_gradient_checkpointing: bool = True,
        device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]] = "auto",
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        offload_folder: Optional[str] = None,
        offload_state_dict: bool = False,
        torch_dtype: Optional[torch.dtype] = torch.float16
    ):
        if not CT_AVAILABLE:
            raise ImportError("CTransformers is required for GGUF conversion. Install with: pip install ctransformers")

        if bits not in SUPPORTED_GGUF_BITS:
            raise ValueError(f"Unsupported bits for GGUF: {bits}. Supported values are {SUPPORTED_GGUF_BITS}")
        
        if quant_type and quant_type not in SUPPORTED_GGUF_TYPES.get(bits, {}):
            raise ValueError(f"Unsupported GGUF quantization type {quant_type} for {bits} bits. "
                           f"Supported types: {list(SUPPORTED_GGUF_TYPES.get(bits, {}).keys())}")

        self.bits = bits
        self.group_size = group_size
        self.use_packed = use_packed
        self.quant_type = quant_type or self._get_default_quant_type(bits)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get tensor-specific quantization configuration
        self.tensor_configs = SUPPORTED_GGUF_TYPES[bits][self.quant_type]["tensor_configs"]
        
        # BitsAndBytes config
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        
        # Accelerate config
        self.device_map = device_map
        self.max_memory = max_memory
        self.offload_folder = offload_folder
        self.offload_state_dict = offload_state_dict
        self.torch_dtype = torch_dtype
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Initialize model and tokenizer
        self._initialize_model_and_tokenizer(model_name)

    def _get_default_quant_type(self, bits: int) -> str:
        """Select optimal GGUF quantization type based on bit width."""
        if bits in SUPPORTED_GGUF_TYPES:
            types = list(SUPPORTED_GGUF_TYPES[bits].keys())
            # Prefer balanced options (e.g., Q4_K_M over Q4_K_S or Q4_0)
            preferences = {
                2: "Q2_K",
                3: "Q3_K_M",
                4: "Q4_K_M",
                5: "Q5_K_M",
                6: "Q6_K",
                8: "Q8_0"
            }
            return preferences.get(bits, types[0])
        raise ValueError(f"No supported GGUF types for {bits} bits")

    def get_tensor_quant_type(self, tensor_name: str) -> Union[str, List[str]]:
        """Get the quantization type for a specific tensor."""
        # Check for exact match
        if tensor_name in self.tensor_configs:
            return self.tensor_configs[tensor_name]
        
        # Check for partial matches (e.g., "attention.wv" in "model.attention.wv.weight")
        for key in self.tensor_configs:
            if key != "default" and key in tensor_name:
                return self.tensor_configs[key]
        
        # Return default if no specific config found
        return self.tensor_configs["default"]

    def _get_quant_description(self) -> str:
        """Get the description of the current quantization configuration."""
        return SUPPORTED_GGUF_TYPES[self.bits][self.quant_type]["description"]

    def _initialize_model_and_tokenizer(self, model_name: Union[str, PreTrainedModel]):
        """Initialize model and tokenizer using BitsAndBytes."""
        if isinstance(model_name, str):
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(model_name)
            self.model.tie_weights()
            self.model = load_checkpoint_and_dispatch(
                self.model,
                model_name,
                device_map=self.device_map,
                no_split_module_classes=["BloomBlock"],
                dtype=self.torch_dtype,
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant
            )
        elif isinstance(model_name, PreTrainedModel):
            self.model = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
        else:
            raise TypeError("model_name must be a string or a PreTrainedModel")

        # Enable gradient checkpointing if requested
        if self.use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.log_info("Gradient checkpointing enabled for memory efficiency")

    def _log_model_stats(self, model: PreTrainedModel, stage: str = ""):
        """Log meaningful model statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = (param_size + buffer_size) / (1024 * 1024)  # MB
        
        prefix = f"{stage} " if stage else ""
        logger.log_info(f"\n{prefix}Model Statistics:")
        logger.log_info(f"Total Parameters: {total_params:,}")
        logger.log_info(f"Model Size: {total_size:.2f} MB")
        if torch.cuda.is_available():
            logger.log_info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
            logger.log_info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB")

    def quantize(
        self,
        calibration_data: Optional[torch.Tensor] = None
    ) -> PreTrainedModel:
        """Quantize model using GGUF format with optimized memory handling."""
        if not CT_AVAILABLE:
            raise ImportError("CTransformers is required for GGUF quantization")

        try:
            logger.log_info("\n" + "="*60)
            logger.log_info("Starting GGUF Quantization Process")
            logger.log_info("="*60)
            
            # Log initial model stats
            self._log_model_stats(self.model, "Original")
            
            # Prepare model for quantization
            if not hasattr(self.model, '_prepared_for_quantization'): 
                self.model = self._prepare_model_instance(self.model, make_copy=True)
                self.model._prepared_for_quantization = True
            
            # Determine device strategy
            device = torch.device('cpu') if self.cpu_offload else self.device_manager.primary_device
            logger.log_info(f"\nUsing device {device} for quantization")
            
            # Move model to appropriate device if not already there
            if next(self.model.parameters()).device != device:
                logger.log_info(f"Moving model to {device}")
                self.model.to(device)
                self._clear_memory()
            
            self.model.eval()
            
            # Process layers in chunks
            logger.log_info("\nStarting layer-by-layer quantization...")
            modules_to_quantize = [
                (name, module) for name, module in self.model.named_modules() 
                if isinstance(module, nn.Linear)
            ]
            
            total_layers = len(modules_to_quantize)
            chunks = [modules_to_quantize[i:i + self.chunk_size] 
                     for i in range(0, total_layers, self.chunk_size)]
            
            start_time = time.perf_counter()
            
            # Create progress bars
            chunk_pbar = tqdm(chunks, desc="Processing chunks", position=0)
            layer_pbar = tqdm(total=total_layers, desc="Quantizing layers", position=1, leave=True)
            
            for chunk_idx, chunk in enumerate(chunk_pbar):
                chunk_pbar.set_description(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
                
                for idx, (name, module) in enumerate(chunk, 1):
                    try:
                        current_layer = idx + chunk_idx * self.chunk_size
                        layer_shape = list(module.weight.shape)
                        layer_pbar.set_description(
                            f"Layer {current_layer}/{total_layers}: {name} {layer_shape}"
                        )
                        
                        # Move layer to target device if needed
                        if module.weight.device != device:
                            module = module.to(device)
                        
                        quantized_layer = self._quantize_layer(module)
                        
                        # Update model with quantized layer
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        
                        if parent_name:
                            parent = self.model.get_submodule(parent_name)
                            setattr(parent, child_name, quantized_layer)
                        else:
                            setattr(self.model, name, quantized_layer)
                        
                        # Update progress
                        layer_pbar.update(1)
                        elapsed_time = time.perf_counter() - start_time
                        eta = elapsed_time / (current_layer / total_layers) - elapsed_time if current_layer > 0 else 0
                        layer_pbar.set_postfix({"ETA": f"{eta:.1f}s"})
                        
                        self._clear_memory()
                        
                    except Exception as e:
                        logger.log_error(f"Failed to quantize layer {name}: {str(e)}")
                        raise RuntimeError(f"GGUF quantization failed at layer {name}: {str(e)}") from e
                
                # Clear memory after each chunk
                if not self.cpu_offload:
                    torch.cuda.empty_cache()
                gc.collect()

            # Close progress bars
            layer_pbar.close()
            chunk_pbar.close()

            # Log final statistics
            total_time = time.perf_counter() - start_time
            logger.log_info("\n" + "="*60)
            logger.log_info("Quantization Complete")
            logger.log_info(f"Total time: {total_time:.2f} seconds")
            logger.log_info(f"Average time per layer: {total_time/total_layers:.2f} seconds")
            
            # Log quantized model stats
            self._log_model_stats(self.model, "Quantized")
            
            # Calculate compression ratio
            original_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            quantized_size = sum(p.numel() * (self.bits / 8) for p in self.model.parameters()) / (1024 * 1024)
            compression_ratio = original_size / quantized_size
            logger.log_info(f"Compression Ratio: {compression_ratio:.2f}x")
            logger.log_info("="*60 + "\n")

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
        Quantize a single linear layer to GGUF format with device-aware processing.
        """
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear layer, got {type(layer)}")

        device = torch.device('cpu') if self.cpu_offload else self.device_manager.primary_device
        
        try:
            # Get layer device and ensure it's on the correct device
            layer_device = next(layer.parameters()).device
            if layer_device != device:
                layer = layer.to(device)
            
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

            # Quantize weights
            qweight = torch.clamp(
                torch.round(weight_scaled * (2**(self.bits-1) - 1)),
                -2**(self.bits-1),
                2**(self.bits-1) - 1
            ).to(torch.int8)

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
            ).to(device)

            # Store quantized weights and parameters
            qlayer.weight_quantized = qweight
            qlayer.weight_scale = 1.0 / scales
            qlayer.weight_zero_point = zeros
            if layer.bias is not None:
                qlayer.bias = layer.bias.data.clone()

            return qlayer

        except Exception as e:
            logger.log_error(f"Failed to quantize layer: {str(e)}")
            raise RuntimeError(f"GGUF layer quantization failed: {str(e)}") from e
        finally:
            # Clean up original layer if it was moved
            if layer_device != device:
                del layer
            if not self.cpu_offload:
                torch.cuda.empty_cache()
            gc.collect()

    def convert_to_gguf(self, output_path: str):
        """
        Convert quantized model to GGUF format using llama.cpp conversion tools.
        """
        if not CT_AVAILABLE:
            raise ImportError("CTransformers is required for GGUF conversion")
        
        try:
            logger.log_info(f"\nConverting model to GGUF format: {output_path}")
            logger.log_info(f"Using quantization type: {self.quant_type}")
            
            # Ensure model is on CPU for conversion
            if not self.cpu_offload:
                self.model.to('cpu')
            
            # Save model in HF format first
            temp_dir = f"{output_path}_temp_hf"
            logger.log_info(f"Saving temporary HF checkpoint to: {temp_dir}")
            self.model.save_pretrained(temp_dir, safe_serialization=True)
            
            # Prepare conversion command
            import subprocess
            import sys
            import os
            
            # Try to find convert.py from llama.cpp
            convert_script = None
            potential_paths = [
                "llama.cpp/convert.py",
                os.path.join(os.path.dirname(sys.executable), "llama.cpp/convert.py"),
                os.path.expanduser("~/.local/lib/python*/site-packages/llama_cpp_python/convert.py"),
                "/usr/local/lib/python*/site-packages/llama_cpp_python/convert.py"
            ]
            
            for path in potential_paths:
                if "*" in path:
                    import glob
                    matches = glob.glob(path)
                    if matches:
                        convert_script = matches[0]
                        break
                elif os.path.exists(path):
                    convert_script = path
                    break
            
            if not convert_script:
                raise RuntimeError(
                    "Could not find llama.cpp convert.py script. Please install llama-cpp-python: "
                    "pip install llama-cpp-python"
                )
            
            # Build conversion command
            cmd = [
                sys.executable,
                convert_script,
                temp_dir,
                "--outfile", output_path,
                "--outtype", f"q{self.bits}" if self.bits < 16 else "f16" if self.bits == 16 else "f32",
            ]
            
            # Add model type specific args
            model_type = self.model.config.model_type if hasattr(self.model, 'config') else "llama"
            if model_type in ["llama", "mistral"]:
                cmd.extend(["--model-type", model_type])
            
            # Execute conversion
            logger.log_info("Running GGUF conversion...")
            logger.log_info(f"Command: {' '.join(cmd)}")
            
            with tqdm(total=100, desc="Converting to GGUF", unit="%") as pbar:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Monitor conversion progress
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        if "Converting" in output:
                            try:
                                progress = int(output.split("%")[0].split()[-1])
                                pbar.n = progress
                                pbar.refresh()
                            except:
                                pass
                
                # Get return code and output
                return_code = process.wait()
                if return_code != 0:
                    error_output = process.stderr.read()
                    raise RuntimeError(f"GGUF conversion failed with error:\n{error_output}")
            
            # Cleanup temporary files
            import shutil
            logger.log_info("Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            logger.log_info(f"Successfully saved model in GGUF format to: {output_path}")
            
        except Exception as e:
            logger.log_error(f"GGUF conversion failed: {str(e)}")
            raise RuntimeError(f"Failed to convert model to GGUF: {str(e)}") from e
        finally:
            self._clear_memory()
    
    def _clear_memory(self):
        """Enhanced memory cleanup for GGUF operations."""
        gc.collect()
        if not self.cpu_offload and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    