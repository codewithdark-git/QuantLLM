"""GGUF (GGML Universal Format) quantization implementation with enhanced 2-bit support."""

import gc
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import bitsandbytes as bnb
from ..utils.logger import logger
import time
from tqdm.auto import tqdm
import os
import sys
import shutil
import subprocess
import glob
from pathlib import Path

try:
    import ctransformers
    from ctransformers import AutoModelForCausalLM as CTAutoModel
    CT_AVAILABLE = True
except ImportError:
    CT_AVAILABLE = False

# Updated GGUF quantization types with modern 2-bit support
SUPPORTED_GGUF_TYPES = {
    2: {
        "Q2_K": {
            "description": "Uses Q4_K for attention.vw and feed_forward.w2, Q2_K for others",
            "tensor_configs": {
                "attention.wv": "Q4_K",
                "feed_forward.w2": "Q4_K",
                "default": "Q2_K"
            }
        },
        "IQ2_XXS": {
            "description": "Importance-based 2-bit quantization (extra small)",
            "tensor_configs": {
                "default": "IQ2_XXS"
            }
        },
        "IQ2_XS": {
            "description": "Importance-based 2-bit quantization (small)",
            "tensor_configs": {
                "default": "IQ2_XS"
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
                "attention.wv": ["Q6_K", "Q4_K"],
                "feed_forward.w2": ["Q6_K", "Q4_K"],
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
                "attention.wv": ["Q6_K", "Q5_K"],
                "feed_forward.w2": ["Q6_K", "Q5_K"],
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

class GGUFQuantizer:
    """GGUF-specific quantizer implementation with enhanced quantization support."""
    
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel],
        bits: int = 4,
        group_size: int = 32,
        use_packed: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        quant_type: Optional[str] = None,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        use_gradient_checkpointing: bool = True,
        device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]] = "auto",
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        offload_folder: Optional[str] = None,
        offload_state_dict: bool = False,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        cpu_offload: bool = False
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
        self.cpu_offload = cpu_offload
        
        # Get tensor-specific quantization configuration
        self.tensor_configs = SUPPORTED_GGUF_TYPES[bits][self.quant_type]["tensor_configs"]
        
        # Accelerate config
        self.device_map = device_map
        self.max_memory = max_memory
        self.offload_folder = offload_folder
        self.offload_state_dict = offload_state_dict
        self.torch_dtype = torch_dtype
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.quantization_config = quantization_config

        # Initialize model and tokenizer
        self._initialize_model_and_tokenizer(model_name)

    def _get_default_quant_type(self, bits: int) -> str:
        """Select optimal GGUF quantization type based on bit width."""
        preferences = {
            2: "IQ2_XS",  # Prefer modern 2-bit quantization
            3: "Q3_K_M",
            4: "Q4_K_M",
            5: "Q5_K_M",
            6: "Q6_K",
            8: "Q8_0"
        }
        if bits in SUPPORTED_GGUF_TYPES:
            types = list(SUPPORTED_GGUF_TYPES[bits].keys())
            return preferences.get(bits, types[0])
        raise ValueError(f"No supported GGUF types for {bits} bits")

    def get_tensor_quant_type(self, tensor_name: str) -> Union[str, List[str]]:
        """Get the quantization type for a specific tensor."""
        if tensor_name in self.tensor_configs:
            return self.tensor_configs[tensor_name]
        
        for key in self.tensor_configs:
            if key != "default" and key in tensor_name:
                return self.tensor_configs[key]
        
        return self.tensor_configs["default"]

    def _get_quant_description(self) -> str:
        """Get the description of the current quantization configuration."""
        return SUPPORTED_GGUF_TYPES[self.bits][self.quant_type]["description"]

    def _initialize_model_and_tokenizer(self, model_name: Union[str, PreTrainedModel]):
        """Initialize model and tokenizer using BitsAndBytes for 4/8-bit or FP16 for others."""
        try:
            if isinstance(model_name, str):
                logger.log_info(f"Loading tokenizer from: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                
                logger.log_info(f"Loading model with {'BitsAndBytes' if self.bits in [4, 8] else 'FP16'} configuration")
                
                load_kwargs = {
                    "device_map": self.device_map,
                    "max_memory": self.max_memory,
                    "offload_folder": self.offload_folder,
                    "offload_state_dict": self.offload_state_dict,
                    "torch_dtype": self.torch_dtype,
                    "trust_remote_code": True,
                }
                
                if self.quantization_config and self.bits in [4, 8]:
                    load_kwargs["quantization_config"] = self.quantization_config
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **load_kwargs
                )
            elif isinstance(model_name, PreTrainedModel):
                self.model = model_name
                if hasattr(model_name.config, '_name_or_path') and model_name.config._name_or_path:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name.config._name_or_path,
                            trust_remote_code=True
                        )
                    except Exception as e:
                        logger.log_warning(f"Could not load tokenizer: {e}")
            else:
                raise TypeError("model_name must be a string or PreTrainedModel instance")

            if self.use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                logger.log_info("Gradient checkpointing enabled for memory efficiency")

            self._log_model_stats(self.model, "Initial")

        except Exception as e:
            logger.log_error(f"Failed to initialize model: {str(e)}")
            raise

    def _log_model_stats(self, model: PreTrainedModel, stage: str = ""):
        """Log meaningful model statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = (param_size + buffer_size) / (1024 * 1024)
        
        prefix = f"{stage} " if stage else ""
        logger.log_info(f"\n{prefix}Model Statistics:")
        logger.log_info(f"Total Parameters: {total_params:,}")
        logger.log_info(f"Model Size: {total_size:.2f} MB")
        if torch.cuda.is_available():
            logger.log_info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
            logger.log_info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB")

    def convert_to_gguf(self, output_path: str):
        """Convert model to GGUF format with separate quantization step."""
        if not CT_AVAILABLE:
            raise ImportError("CTransformers is required for GGUF conversion")
        
        temp_dir = None
        temp_gguf = None
        try:
            logger.log_info("\n" + "="*80)
            logger.log_info("🚀 Starting GGUF Conversion Process".center(80))
            logger.log_info("="*80 + "\n")
            
            # Model Information
            logger.log_info("📊 Model Information:")
            logger.log_info("-"*40)
            model_type = self.model.config.model_type if hasattr(self.model, 'config') else None
            supported_types = ["llama", "mistral", "falcon", "mpt", "gpt_neox", "pythia", "stablelm"]
            
            if model_type in supported_types:
                logger.log_info(f"• Architecture: {model_type.upper()}")
            else:
                logger.log_info(f"• Architecture: Unknown (using default LLAMA)")
                model_type = "llama"
            
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.log_info(f"• Total Parameters: {total_params:,}")
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3)
            logger.log_info(f"• Model Size: {model_size:.2f} GB")
            logger.log_info("")
            
            # Conversion Settings
            logger.log_info("⚙️ Conversion Settings:")
            logger.log_info("-"*40)
            logger.log_info(f"• Output Path: {output_path}")
            logger.log_info(f"• Quantization Type: {self.quant_type}")
            logger.log_info(f"• Target Bits: {self.bits}")
            logger.log_info(f"• Group Size: {self.group_size}")
            logger.log_info("")
            
            # Save temporary checkpoint
            temp_dir = f"{output_path}_temp_hf"
            logger.log_info("💾 Saving Temporary Checkpoint:")
            logger.log_info("-"*40)
            logger.log_info(f"• Checkpoint Path: {temp_dir}")
            self.model.save_pretrained(temp_dir, safe_serialization=True)
            logger.log_info("• Checkpoint saved successfully")
            logger.log_info("")
            
            # Find llama.cpp tools
            logger.log_info("🔍 Locating GGUF Conversion Tools:")
            logger.log_info("-"*40)
            
            try:
                import llama_cpp
                llama_cpp_path = os.path.dirname(llama_cpp.__file__)
                convert_script = os.path.join(llama_cpp_path, "convert.py")
                quantize_bin = os.path.join(llama_cpp_path, "quantize")
                if not os.path.exists(convert_script):
                    raise FileNotFoundError("convert.py not found")
                if not os.path.exists(quantize_bin):
                    raise FileNotFoundError("quantize binary not found")
                logger.log_info(f"• Found convert.py: {convert_script}")
                logger.log_info(f"• Found quantize: {quantize_bin}")
            except (ImportError, FileNotFoundError) as e:
                logger.log_error(f"• Failed to locate llama.cpp tools: {e}")
                try:
                    logger.log_info("• Attempting to install llama-cpp-python...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "llama-cpp-python"])
                    import llama_cpp
                    llama_cpp_path = os.path.dirname(llama_cpp.__file__)
                    convert_script = os.path.join(llama_cpp_path, "convert.py")
                    quantize_bin = os.path.join(llama_cpp_path, "quantize")
                    logger.log_info("• Successfully installed and located tools")
                except Exception as inst_err:
                    raise RuntimeError(
                        f"Could not find or install llama-cpp-python: {inst_err}\n"
                        "Install manually: pip install llama-cpp-python --upgrade"
                    ) from e
            
            # Convert to FP16 GGUF
            logger.log_info("🛠️ Converting to FP16 GGUF:")
            logger.log_info("-"*40)
            temp_gguf = f"{output_path}_temp_f16.gguf"
            cmd_convert = [
                sys.executable,
                convert_script,
                temp_dir,
                "--outfile", temp_gguf,
                "--outtype", "f16",
                "--model-type", model_type
            ]
            
            logger.log_info(f"• Command: {' '.join(cmd_convert)}")
            with tqdm(total=100, desc="Converting to FP16", unit="%") as pbar:
                process = subprocess.Popen(
                    cmd_convert,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output and "Converting" in output:
                        try:
                            progress = int(output.split("%")[0].split()[-1])
                            pbar.n = progress
                            pbar.refresh()
                        except:
                            pass
                        logger.log_info(f"• {output.strip()}")
            
            return_code = process.wait()
            if return_code != 0:
                error_output = process.stderr.read()
                raise RuntimeError(f"FP16 GGUF conversion failed:\n{error_output}")
            
            # Quantize to target type
            logger.log_info("\n🔄 Quantizing GGUF:")
            logger.log_info("-"*40)
            cmd_quantize = [
                quantize_bin,
                temp_gguf,
                output_path,
                self.quant_type.lower()  # llama.cpp expects lowercase
            ]
            
            logger.log_info(f"• Command: {' '.join(cmd_quantize)}")
            with tqdm(total=100, desc="Quantizing GGUF", unit="%") as pbar:
                process = subprocess.Popen(
                    cmd_quantize,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output and "%" in output:
                        try:
                            progress = int(output.split("%")[0].split()[-1])
                            pbar.n = progress
                            pbar.refresh()
                        except:
                            pass
                        logger.log_info(f"• {output.strip()}")
            
            return_code = process.wait()
            if return_code != 0:
                error_output = process.stderr.read()
                raise RuntimeError(f"GGUF quantization failed:\n{error_output}")
            
            # Verify results
            if os.path.exists(output_path):
                logger.log_info("\n✅ Conversion Results:")
                logger.log_info("-"*40)
                
                file_size = os.path.getsize(output_path) / (1024**3)
                logger.log_info(f"• GGUF File Size: {file_size:.2f} GB")
                
                compression_ratio = model_size / file_size
                logger.log_info(f"• Compression Ratio: {compression_ratio:.2f}x")
                logger.log_info(f"• Output Path: {output_path}")
                
                logger.log_info("\n" + "="*80)
                logger.log_info("✨ GGUF Conversion Completed Successfully! ✨".center(80))
                logger.log_info("="*80 + "\n")
            else:
                raise RuntimeError(f"GGUF file was not created at {output_path}")
            
        except Exception as e:
            logger.log_error("\n❌ Conversion Failed:")
            logger.log_error("-"*40)
            logger.log_error(f"• Error: {str(e)}")
            raise RuntimeError(f"Failed to convert model to GGUF: {str(e)}") from e
        
        finally:
            if temp_dir and os.path.exists(temp_dir):
                logger.log_info("\n🧹 Cleaning Up:")
                logger.log_info("-"*40)
                logger.log_info("• Removing temporary files...")
                shutil.rmtree(temp_dir, ignore_errors=True)
            if temp_gguf and os.path.exists(temp_gguf):
                os.remove(temp_gguf)
            self._clear_memory()

    def _clear_memory(self):
        """Enhanced memory cleanup for GGUF operations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
