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
    
    LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"
    CONVERT_SCRIPTS = [
        "convert_hf_to_gguf.py",
        "convert_hf_to_gguf_update.py",
        "convert_llama_to_gguf.py",
        "convert_lora_to_gguf.py"
    ]
    
    def __init__(self):
        self.llama_cpp_path = None
        self.convert_script = None
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup llama.cpp paths and environment."""
        # First check if we already have llama.cpp cloned
        potential_paths = [
            Path.cwd() / "llama.cpp",
            Path.home() / "llama.cpp",
            Path(os.getenv("LLAMA_CPP_DIR", "")) if os.getenv("LLAMA_CPP_DIR") else None
        ]
        
        for path in potential_paths:
            if path and path.exists():
                # Check for any of the conversion scripts
                for script in self.CONVERT_SCRIPTS:
                    script_path = path / script
                    if script_path.exists():
                        self.llama_cpp_path = path
                        self.convert_script = str(script_path)
                        logger.log_info(f"✓ Found existing llama.cpp installation at: {path}")
                        logger.log_info(f"✓ Using conversion script: {script}")
                        return
                    
        # If not found, we'll clone it
        self._clone_llama_cpp()
    
    def _clone_llama_cpp(self):
        """Clone llama.cpp from GitHub."""
        try:
            logger.log_info("\n🔄 Setting up llama.cpp:")
            logger.log_info("-" * 40)
            
            # Create a directory for llama.cpp
            self.llama_cpp_path = Path.cwd() / "llama.cpp"
            if self.llama_cpp_path.exists():
                logger.log_info("• Cleaning existing llama.cpp directory...")
                shutil.rmtree(self.llama_cpp_path)
            
            # Clone the repository
            logger.log_info("• Cloning llama.cpp repository...")
            subprocess.run(
                ["git", "clone", self.LLAMA_CPP_REPO],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Verify the clone
            if not self.llama_cpp_path.exists():
                raise RuntimeError("Failed to clone llama.cpp repository")
            
            # Find appropriate conversion script
            for script in self.CONVERT_SCRIPTS:
                script_path = self.llama_cpp_path / script
                if script_path.exists():
                    self.convert_script = str(script_path)
                    logger.log_info(f"• Found conversion script: {script}")
                    break
            
            if not self.convert_script:
                # List available files for debugging
                available_files = list(self.llama_cpp_path.glob("convert*.py"))
                logger.log_info("• Available conversion scripts:")
                for file in available_files:
                    logger.log_info(f"  - {file.name}")
                raise RuntimeError(
                    "Could not find appropriate conversion script in llama.cpp. "
                    f"Available scripts: {[f.name for f in available_files]}"
                )
            
            logger.log_info("• Successfully set up llama.cpp")
            logger.log_info(f"• Convert script location: {self.convert_script}")
            
        except Exception as e:
            logger.log_error(f"Failed to clone/setup llama.cpp: {e}")
            raise RuntimeError(
                "Could not set up llama.cpp. Please clone manually:\n"
                "git clone https://github.com/ggerganov/llama.cpp.git"
            )
    
    def _detect_model_type(self, model: PreTrainedModel) -> str:
        """Detect model architecture type."""
        if not hasattr(model, 'config'):
            return "llama"  # Default to llama
            
        model_type = model.config.model_type.lower()
        
        # Map similar architectures to supported types
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
        if model_type not in self.SUPPORTED_TYPES:
            logger.log_warning(f"Model type {model_type} not directly supported, using llama")
            return "llama"
            
        return model_type
    
    def _save_model_config(self, model: PreTrainedModel, save_dir: str):
        """Save minimal model configuration."""
        config = model.config.to_dict()
        
        # Keep only essential config items
        essential_keys = [
            "model_type", "vocab_size", "hidden_size", "num_attention_heads",
            "num_hidden_layers", "intermediate_size", "max_position_embeddings"
        ]
        
        minimal_config = {k: config[k] for k in essential_keys if k in config}
        
        # Add quantization info if available
        if hasattr(model.config, 'quantization_config'):
            quant_config = model.config.quantization_config
            if isinstance(quant_config, dict):
                minimal_config['quantization_config'] = quant_config
            else:
                # Convert BitsAndBytesConfig to dict
                minimal_config['quantization_config'] = {
                    'bits': quant_config.bits if hasattr(quant_config, 'bits') else None,
                    'group_size': quant_config.group_size if hasattr(quant_config, 'group_size') else None,
                    'quant_method': quant_config.quant_method if hasattr(quant_config, 'quant_method') else None,
                    'load_in_4bit': quant_config.load_in_4bit if hasattr(quant_config, 'load_in_4bit') else False,
                    'load_in_8bit': quant_config.load_in_8bit if hasattr(quant_config, 'load_in_8bit') else False,
                    'llm_int8_threshold': quant_config.llm_int8_threshold if hasattr(quant_config, 'llm_int8_threshold') else 6.0,
                    'llm_int8_has_fp16_weight': quant_config.llm_int8_has_fp16_weight if hasattr(quant_config, 'llm_int8_has_fp16_weight') else False,
                    'bnb_4bit_compute_dtype': str(quant_config.bnb_4bit_compute_dtype) if hasattr(quant_config, 'bnb_4bit_compute_dtype') else None,
                    'bnb_4bit_quant_type': quant_config.bnb_4bit_quant_type if hasattr(quant_config, 'bnb_4bit_quant_type') else None,
                    'bnb_4bit_use_double_quant': quant_config.bnb_4bit_use_double_quant if hasattr(quant_config, 'bnb_4bit_use_double_quant') else False
                }
                # Remove None values
                minimal_config['quantization_config'] = {
                    k: v for k, v in minimal_config['quantization_config'].items() 
                    if v is not None
                }
            
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f, indent=2)
    
    def convert_to_gguf(
        self,
        model: PreTrainedModel,
        output_dir: str,
        model_type: Optional[str] = None,
        bits: int = 4,
        group_size: int = 128,
        save_tokenizer: bool = True
    ) -> str:
        """
        Convert model to GGUF format.
        
        Args:
            model: The model to convert
            output_dir: Output directory
            model_type: Model architecture type (auto-detected if None)
            bits: Target quantization bits
            group_size: Quantization group size
            save_tokenizer: Whether to save tokenizer
            
        Returns:
            Path to the converted GGUF file
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Ensure llama.cpp is available
            if not self.convert_script:
                raise RuntimeError(
                    "llama.cpp conversion script not found. Please ensure llama.cpp is properly set up."
                )
            
            # Detect model type
            if not model_type:
                model_type = self._detect_model_type(model)
            
            # Create temporary directory for minimal checkpoint
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.log_info("\n💾 Preparing model for conversion:")
                logger.log_info("-" * 40)
                
                # Save minimal checkpoint
                logger.log_info("• Saving minimal checkpoint...")
                self._save_model_config(model, temp_dir)
                
                # Save only model weights without optimizer states
                logger.log_info("• Saving model weights...")
                torch.save(
                    model.state_dict(),
                    os.path.join(temp_dir, "pytorch_model.bin")
                )
                
                # Save tokenizer if requested
                if save_tokenizer:
                    logger.log_info("• Saving tokenizer...")
                    if hasattr(model.config, '_name_or_path'):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(
                                model.config._name_or_path,
                                trust_remote_code=True
                            )
                            tokenizer.save_pretrained(output_dir)
                        except Exception as e:
                            logger.log_warning(f"Failed to save tokenizer: {e}")
                
                # Prepare GGUF conversion command
                gguf_path = os.path.join(output_dir, "model.gguf")
                cmd = [
                    sys.executable,
                    self.convert_script,
                    temp_dir,
                    "--outfile", gguf_path,
                    "--outtype", f"q{bits}",
                    "--model-type", model_type
                ]
                
                logger.log_info("\n🛠️ Converting to GGUF:")
                logger.log_info("-" * 40)
                logger.log_info(f"• Model type: {model_type}")
                logger.log_info(f"• Target bits: {bits}")
                logger.log_info(f"• Output path: {gguf_path}")
                
                # Run conversion
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Monitor progress
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        logger.log_info(f"• {output.strip()}")
                
                # Check for errors
                return_code = process.wait()
                if return_code != 0:
                    error = process.stderr.read()
                    raise RuntimeError(f"GGUF conversion failed: {error}")
                
                # Verify output
                if not os.path.exists(gguf_path):
                    raise RuntimeError("GGUF file was not created")
                
                # Log results
                file_size = os.path.getsize(gguf_path) / (1024**3)
                logger.log_info("\n✅ Conversion completed:")
                logger.log_info("-" * 40)
                logger.log_info(f"• GGUF file size: {file_size:.2f} GB")
                logger.log_info(f"• Saved to: {gguf_path}")
                
                return gguf_path
                
        except Exception as e:
            logger.log_error("\n❌ Conversion failed:")
            logger.log_error("-" * 40)
            logger.log_error(f"• Error: {str(e)}")
            raise RuntimeError(f"Failed to convert model to GGUF: {str(e)}") 