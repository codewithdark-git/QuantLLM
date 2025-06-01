"""GGUF conversion utilities using llama.cpp."""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import torch
from transformers import PreTrainedModel, AutoTokenizer
from ..utils.logger import logger

class LlamaCppConverter:
    """Handles conversion of models to GGUF format using llama.cpp."""
    
    SUPPORTED_TYPES = [
        "llama", "mistral", "falcon", "mpt", 
        "gpt_neox", "pythia", "stablelm", "phi"
    ]
    
    OUTTYPE_MAP = {
        2: "tq2_0",
        4: "q8_0",  # Using q8_0 for better compatibility
        8: "q8_0",
        16: "f16",
        32: "f32"
    }
    
    def __init__(self):
        """Initialize converter with installed llama-cpp-python package."""
        try:
            import llama_cpp
            self.llama_cpp_path = os.path.dirname(llama_cpp.__file__)
            self.convert_script = None
            for script in ["convert_hf_to_gguf.py", "convert.py"]:
                potential_script = os.path.join(self.llama_cpp_path, script)
                if os.path.exists(potential_script):
                    self.convert_script = potential_script
                    break
            
            if not self.convert_script:
                raise ImportError("Conversion script not found in llama-cpp-python package")
                
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required. Install with:\n"
                "pip install llama-cpp-python --upgrade"
            )
    
    def _detect_model_type(self, model: PreTrainedModel) -> str:
        """Detect model architecture type."""
        if not hasattr(model, 'config'):
            return "llama"
            
        model_type = model.config.model_type.lower()
        type_mapping = {
            "llama2": "llama",
            "mistral-7b": "mistral",
            "falcon-7b": "falcon",
            "falcon-40b": "falcon",
            "mpt-7b": "mpt",
            "pythia-7b": "pythia",
            "stablelm-base": "stablelm"
        }
        
        model_type = type_mapping.get(model_type, model_type)
        return model_type if model_type in self.SUPPORTED_TYPES else "llama"
    
    def convert_to_gguf(
        self,
        model: PreTrainedModel,
        output_dir: str,
        model_type: Optional[str] = None,
        bits: int = 4,
        group_size: int = 128,
        save_tokenizer: bool = True
    ) -> str:
        """Convert model to GGUF format."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            model_type = model_type or self._detect_model_type(model)
            gguf_path = os.path.join(output_dir, "model.gguf")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save minimal config
                config = {
                    "model_type": model_type,
                    "architectures": model.config.architectures if hasattr(model.config, 'architectures') else None,
                    "torch_dtype": str(model.dtype) if hasattr(model, 'dtype') else "float32"
                }
                with open(os.path.join(temp_dir, "config.json"), "w") as f:
                    json.dump(config, f)
                
                # Save model in safetensors format
                model.save_pretrained(temp_dir, safe_serialization=True)
                
                # Save tokenizer if requested
                if save_tokenizer and hasattr(model.config, '_name_or_path'):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            model.config._name_or_path,
                            trust_remote_code=True
                        )
                        tokenizer.save_pretrained(temp_dir)
                    except Exception:
                        pass
                
                # Convert to GGUF
                outtype = self.OUTTYPE_MAP.get(bits, "q8_0")
                cmd = [
                    sys.executable,
                    self.convert_script,
                    temp_dir,
                    "--outfile", gguf_path,
                    "--outtype", outtype,
                    "--model-type", model_type
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                _, error = process.communicate()
                
                if process.returncode != 0:
                    raise RuntimeError(f"GGUF conversion failed: {error}")
                
                if not os.path.exists(gguf_path):
                    raise RuntimeError("GGUF file was not created")
                
                return gguf_path
                
        except Exception as e:
            raise RuntimeError(f"Failed to convert model to GGUF: {str(e)}")
        finally:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None 