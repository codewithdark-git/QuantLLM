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
    
    def __init__(self):
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup llama.cpp paths and environment."""
        self.llama_cpp_path = None
        self.convert_script = None
        
        # Try to find llama.cpp installation
        try:
            import llama_cpp
            self.llama_cpp_path = Path(llama_cpp.__file__).parent
            potential_paths = [
                self.llama_cpp_path / "convert.py",
                self.llama_cpp_path / "llama_cpp" / "convert.py",
                Path(sys.prefix) / "llama_cpp_python" / "convert.py",
            ]
            
            for path in potential_paths:
                if path.exists():
                    self.convert_script = str(path)
                    break
                    
        except ImportError:
            pass
    
    def _install_llama_cpp(self) -> bool:
        """Install llama-cpp-python package."""
        try:
            logger.log_info("📦 Installing llama-cpp-python...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", "llama-cpp-python"
            ])
            self._setup_paths()
            return self.convert_script is not None
        except Exception as e:
            logger.log_error(f"Failed to install llama-cpp-python: {e}")
            return False
    
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
            minimal_config['quantization_config'] = model.config.quantization_config
            
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
            
            # Setup paths
            if not self.convert_script and not self._install_llama_cpp():
                raise RuntimeError(
                    "Could not find or install llama-cpp-python. "
                    "Please install manually: pip install llama-cpp-python"
                )
            
            # Detect model type
            if not model_type:
                model_type = self._detect_model_type(model)
            
            # Create temporary directory for minimal checkpoint
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.log_info("💾 Preparing model for conversion:")
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