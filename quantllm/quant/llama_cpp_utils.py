"""Enhanced GGUF conversion utilities using llama.cpp with improved performance and logging."""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Callable
import torch
from transformers import PreTrainedModel, AutoTokenizer
from ..utils.logger import logger
import multiprocessing as mp

class ProgressTracker:
    """Track and display conversion progress with clean output."""
    
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        self.status = "Initializing..."
        
    def update(self, step: int, status: str = ""):
        self.current_step = step
        if status:
            self.status = status
        self.step_times.append(time.time())
        
    def get_progress_info(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        progress_pct = min((self.current_step / self.total_steps) * 100, 100)
        
        # Estimate remaining time
        if len(self.step_times) > 1:
            avg_step_time = elapsed / len(self.step_times)
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * avg_step_time
        else:
            eta = 0
            
        return {
            'progress': progress_pct,
            'elapsed': elapsed,
            'eta': eta,
            'status': self.status,
            'step': self.current_step,
            'total': self.total_steps
        }

class LlamaCppConverter:
    """Enhanced GGUF converter with improved performance and user experience."""
    
    SUPPORTED_TYPES = [
        "llama", "mistral", "falcon", "mpt", "gpt_neox", "pythia", 
        "stablelm", "phi", "gemma", "qwen", "baichuan", "yi"
    ]
    
    # Enhanced quantization type mapping with better defaults
    QUANT_TYPE_MAP = {
        2: "q2_k",      # 2-bit with K-quant for better quality
        3: "q3_k_m",    # 3-bit medium quality
        4: "q4_k_m",    # 4-bit medium quality (balanced)
        5: "q5_k_m",    # 5-bit medium quality
        6: "q6_k",      # 6-bit high quality
        8: "q8_0",      # 8-bit highest quality
        16: "f16",      # 16-bit float
        32: "f32"       # 32-bit float
    }
    
    # Performance optimization flags
    OPTIMIZATION_FLAGS = {
        "fast": ["--threads", str(min(mp.cpu_count(), 8)), "--batch-size", "512"],
        "balanced": ["--threads", str(min(mp.cpu_count(), 6)), "--batch-size", "256"],
        "quality": ["--threads", str(min(mp.cpu_count(), 4)), "--batch-size", "128"]
    }
    
    def __init__(self, verbose: bool = True):
        """Initialize converter with performance optimizations."""
        self.verbose = verbose
        self.progress_tracker = None
        
        try:
            # Try multiple ways to find llama-cpp-python
            self._find_llama_cpp_installation()
            
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for GGUF conversion.\n"
                "Install with: pip install llama-cpp-python --upgrade\n"
                f"Error: {e}"
            )
    
    def _find_llama_cpp_installation(self):
        """Find llama-cpp-python installation and conversion scripts."""
        try:
            # First try: Direct import
            import llama_cpp
            self.llama_cpp_path = os.path.dirname(llama_cpp.__file__)
            
            # Look for conversion script in package
            script_path = os.path.join(self.llama_cpp_path, "convert.py")
            if os.path.exists(script_path):
                self.convert_script = script_path
                return
            
            # Look in package scripts directory
            scripts_dir = os.path.join(os.path.dirname(self.llama_cpp_path), "scripts")
            if os.path.exists(scripts_dir):
                for script in ["convert.py", "convert_hf_to_gguf.py"]:
                    script_path = os.path.join(scripts_dir, script)
                    if os.path.exists(script_path):
                        self.convert_script = script_path
                        return
            
            # Try pip installation path
            try:
                import site
                for site_dir in site.getsitepackages():
                    for script in ["convert.py", "convert_hf_to_gguf.py"]:
                        script_path = os.path.join(site_dir, "llama_cpp", script)
                        if os.path.exists(script_path):
                            self.convert_script = script_path
                            return
                        script_path = os.path.join(site_dir, "llama_cpp", "scripts", script)
                        if os.path.exists(script_path):
                            self.convert_script = script_path
                            return
            except Exception:
                pass
            
            # Try PATH
            for script in ["convert.py", "convert_hf_to_gguf.py"]:
                script_path = shutil.which(script)
                if script_path:
                    self.convert_script = script_path
                    return
                
            raise ImportError("GGUF conversion script not found")
            
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF conversion.\n"
                "Install with: pip install llama-cpp-python --upgrade"
            )
    
    def _detect_model_type(self, model: PreTrainedModel) -> str:
        """Enhanced model type detection with better coverage."""
        if not hasattr(model, 'config'):
            return "llama"
        
        model_type = getattr(model.config, 'model_type', 'unknown').lower()
        
        # Enhanced type mapping
        type_mapping = {
            "llama": "llama",
            "llama2": "llama", 
            "mistral": "mistral",
            "mixtral": "mistral",
            "falcon": "falcon",
            "mpt": "mpt",
            "gpt_neox": "gpt_neox",
            "pythia": "pythia", 
            "stablelm": "stablelm",
            "phi": "phi",
            "phi-msft": "phi",
            "gemma": "gemma",
            "qwen": "qwen",
            "qwen2": "qwen",
            "baichuan": "baichuan",
            "yi": "yi",
            "internlm": "llama",
            "chatglm": "llama"
        }
        
        detected_type = type_mapping.get(model_type, "llama")
        
        if self.verbose:
            logger.log_info(f"🔍 Detected model type: {detected_type} (from {model_type})")
        
        return detected_type
    
    def _optimize_for_conversion(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply optimizations before conversion."""
        if self.verbose:
            logger.log_info("⚡ Applying conversion optimizations...")
        
        # Move to CPU and optimize memory
        if hasattr(model, 'to'):
            model = model.to('cpu')
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Enable eval mode for stability
        model.eval()
        
        return model
    
    def _log_progress(self, process: subprocess.Popen, progress_callback: Optional[Callable] = None):
        """Monitor conversion progress with clean logging."""
        if not self.verbose:
            return
            
        def read_output():
            current_step = 0
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    line = output.strip()
                    if not line:
                        continue
                    
                    # Parse progress indicators
                    if "%" in line or "processing" in line.lower():
                        current_step += 1
                        if self.progress_tracker:
                            self.progress_tracker.update(current_step, line)
                        
                        if progress_callback:
                            progress_callback(current_step, line)
                    
                    # Log important messages
                    if any(keyword in line.lower() for keyword in ['error', 'warning', 'failed']):
                        logger.log_warning(f"⚠️  {line}")
                    elif any(keyword in line.lower() for keyword in ['completed', 'success', 'done']):
                        logger.log_info(f"✅ {line}")
        
        thread = threading.Thread(target=read_output)
        thread.daemon = True
        thread.start()
        return thread
    
    def convert_to_gguf(
        self,
        model: PreTrainedModel,
        output_dir: str,
        model_type: Optional[str] = None,
        bits: int = 4,
        group_size: int = 128,
        optimization_level: str = "balanced",
        save_tokenizer: bool = True,
        progress_callback: Optional[Callable] = None,
        custom_name: Optional[str] = None
    ) -> str:
        """
        Convert model to GGUF format with enhanced performance and logging.
        
        Args:
            model: PreTrainedModel instance
            output_dir: Output directory path
            model_type: Model architecture type (auto-detected if None)
            bits: Quantization bits (2, 3, 4, 5, 6, 8, 16, 32)
            group_size: Quantization group size
            optimization_level: "fast", "balanced", or "quality"
            save_tokenizer: Whether to save tokenizer
            progress_callback: Optional progress callback function
            custom_name: Custom output filename (default: model.gguf)
            
        Returns:
            Path to generated GGUF file
        """
        start_time = time.time()
        
        try:
            # Setup output directory
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename
            if custom_name:
                filename = custom_name if custom_name.endswith('.gguf') else f"{custom_name}.gguf"
            else:
                model_name = model.config.name_or_path.split('/')[-1] if hasattr(model.config, 'name_or_path') else "model"
                filename = f"{model_name}-{bits}bit-{optimization_level}.gguf"
            
            gguf_path = os.path.join(output_dir, filename)
            
            if self.verbose:
                logger.log_info("\n" + "🚀 " + "="*78)
                logger.log_info(" GGUF CONVERSION STARTED ".center(80, "="))
                logger.log_info("="*80)
                logger.log_info(f"📋 Model Type: {model_type or self._detect_model_type(model)}")
                logger.log_info(f"🎯 Target Bits: {bits}")
                logger.log_info(f"📦 Group Size: {group_size}")
                logger.log_info(f"⚡ Optimization: {optimization_level}")
                logger.log_info(f"💾 Output: {gguf_path}")
                logger.log_info("="*80 + "\n")
            
            # Use a temporary directory for intermediate files
            with tempfile.TemporaryDirectory(prefix="gguf_convert_", dir=output_dir) as temp_dir:
                if self.verbose:
                    logger.log_info("📁 Setting up workspace...")
                
                # Initialize progress tracking
                self.progress_tracker = ProgressTracker(total_steps=10)
                
                # Step 1: Optimize model
                self.progress_tracker.update(1, "Optimizing model...")
                model = self._optimize_for_conversion(model)
                
                # Step 2: Save model in optimal format with sharding
                self.progress_tracker.update(2, "Saving model...")
                model.save_pretrained(
                    temp_dir,
                    safe_serialization=True,
                    max_shard_size="2GB"
                )
                
                # Step 3: Save tokenizer if requested
                if save_tokenizer:
                    self.progress_tracker.update(3, "Saving tokenizer...")
                    self._save_tokenizer(model, temp_dir)
                
                # Step 4: Prepare and run conversion
                self.progress_tracker.update(4, "Converting to GGUF...")
                cmd = self._build_conversion_command(
                    temp_dir, gguf_path,
                    model_type or self._detect_model_type(model),
                    bits, optimization_level
                )
                
                success = self._run_conversion(cmd, progress_callback)
                
                if not success or not os.path.exists(gguf_path):
                    raise RuntimeError("GGUF conversion failed")
                
                # Log completion
                file_size = os.path.getsize(gguf_path) / (1024**3)
                elapsed_time = time.time() - start_time
                
                if self.verbose:
                    logger.log_info("\n" + "🎉 " + "="*78)
                    logger.log_info(" CONVERSION COMPLETED ".center(80, "="))
                    logger.log_info("="*80)
                    logger.log_info(f"📁 Output: {gguf_path}")
                    logger.log_info(f"📏 Size: {file_size:.2f} GB")
                    logger.log_info(f"⏱️  Time: {elapsed_time:.1f} seconds")
                    logger.log_info("="*80 + "\n")
                
                return gguf_path
                
        except Exception as e:
            if self.verbose:
                logger.log_error(f"\n❌ Conversion failed: {str(e)}")
            raise RuntimeError(f"GGUF conversion failed: {str(e)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _prepare_model_config(self, model: PreTrainedModel, model_type: str) -> Dict[str, Any]:
        """Prepare optimized model configuration."""
        config = model.config.to_dict() if hasattr(model.config, 'to_dict') else {}
        
        # Essential fields for GGUF conversion
        essential_config = {
            "model_type": model_type,
            "architectures": getattr(model.config, 'architectures', [model_type]),
            "torch_dtype": str(getattr(model, 'dtype', torch.float32)).replace('torch.', ''),
            "transformers_version": "4.36.0",  # Compatibility version
        }
        
        # Preserve important model-specific fields
        important_fields = [
            'hidden_size', 'num_hidden_layers', 'num_attention_heads',
            'vocab_size', 'max_position_embeddings', 'intermediate_size',
            'rms_norm_eps', 'rope_theta', 'sliding_window'
        ]
        
        for field in important_fields:
            if hasattr(model.config, field):
                essential_config[field] = getattr(model.config, field)
        
        return essential_config
    
    def _save_tokenizer(self, model: PreTrainedModel, temp_dir: str):
        """Save tokenizer with error handling."""
        try:
            if hasattr(model.config, '_name_or_path') and model.config._name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(
                    model.config._name_or_path,
                    trust_remote_code=True,
                    use_fast=False  # Use slow tokenizer for better compatibility
                )
                tokenizer.save_pretrained(temp_dir)
                if self.verbose:
                    logger.log_info("💬 Tokenizer saved successfully")
            else:
                if self.verbose:
                    logger.log_warning("⚠️  No tokenizer path found, skipping tokenizer save")
        except Exception as e:
            if self.verbose:
                logger.log_warning(f"⚠️  Could not save tokenizer: {e}")
    
    def _build_conversion_command(
        self, 
        temp_dir: str, 
        gguf_path: str, 
        model_type: str, 
        bits: int,
        optimization_level: str
    ) -> List[str]:
        """Build optimized conversion command."""
        quant_type = self.QUANT_TYPE_MAP.get(bits, "q4_k_m")
        
        cmd = [
            sys.executable,
            self.convert_script,
            temp_dir,
            "--outfile", gguf_path,
            "--outtype", quant_type,
        ]
        
        # Add optimization flags
        if optimization_level in self.OPTIMIZATION_FLAGS:
            cmd.extend(self.OPTIMIZATION_FLAGS[optimization_level])
        
        # Add model-specific flags
        if model_type != "llama":
            cmd.extend(["--model-type", model_type])
        
        # Add memory optimization for large models
        cmd.extend([
            "--no-lazy",  # Disable lazy loading for speed
            "--big-endian"  # Better compatibility
        ])
        
        return cmd
    
    def _run_conversion(self, cmd: List[str], progress_callback: Optional[Callable] = None) -> bool:
        """Run conversion process with enhanced monitoring."""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor progress
            monitor_thread = self._log_progress(process, progress_callback)
            
            # Wait for completion
            return_code = process.wait()
            
            if monitor_thread:
                monitor_thread.join(timeout=5.0)
            
            return return_code == 0
            
        except Exception as e:
            logger.log_error(f"Conversion process error: {e}")
            return False

    def get_supported_quantization_types(self, bits: Optional[int] = None) -> Dict[int, List[str]]:
        """Get supported quantization types."""
        if bits:
            return {bits: [self.QUANT_TYPE_MAP.get(bits, "q4_k_m")]}
        return {k: [v] for k, v in self.QUANT_TYPE_MAP.items()}