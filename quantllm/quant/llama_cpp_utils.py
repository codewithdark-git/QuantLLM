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
    
    # Enhanced quantization type mapping with 2-bit support
    QUANT_TYPE_MAP = {
        2: ["q2_k", "iq2_xxs", "iq2_xs"],
        3: ["q3_k_m", "q3_k_s", "q3_k_l"],
        4: ["q4_k_m", "q4_k_s", "q4_0", "q4_1"],
        5: ["q5_k_m", "q5_k_s", "q5_0", "q5_1"],
        6: ["q6_k"],
        8: ["q8_0"],
        16: ["f16"],
        32: ["f32"]
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
        self.convert_script = None
        self.quantize_bin = None
        
        try:
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
            import llama_cpp
            self.llama_cpp_path = os.path.dirname(llama_cpp.__file__)
            
            # Look for conversion script
            script_path = os.path.join(self.llama_cpp_path, "convert.py")
            if os.path.exists(script_path):
                self.convert_script = script_path
            else:
                scripts_dir = os.path.join(os.path.dirname(self.llama_cpp_path), "scripts")
                for script in ["convert.py", "convert_hf_to_gguf.py"]:
                    script_path = os.path.join(scripts_dir, script)
                    if os.path.exists(script_path):
                        self.convert_script = script_path
                        break
            
            # Look for quantize binary
            quantize_path = os.path.join(self.llama_cpp_path, "quantize")
            if os.path.exists(quantize_path):
                self.quantize_bin = quantize_path
            
            if not self.convert_script or not self.quantize_bin:
                raise FileNotFoundError("Required llama.cpp tools (convert.py or quantize) not found")
            
            if self.verbose:
                logger.log_info(f"Found convert.py: {self.convert_script}")
                logger.log_info(f"Found quantize: {self.quantize_bin}")
            
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
        
        if hasattr(model, 'to'):
            model = model.to('cpu')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
                    
                    if "%" in line or "processing" in line.lower():
                        current_step += 1
                        if self.progress_tracker:
                            self.progress_tracker.update(current_step, line)
                        
                        if progress_callback:
                            progress_callback(current_step, line)
                    
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
        custom_name: Optional[str] = None,
        quant_type: Optional[str] = None
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
            quant_type: Specific quantization type (e.g., Q2_K, IQ2_XXS)
            
        Returns:
            Path to generated GGUF file
        """
        start_time = time.time()
        
        try:
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
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
            
            with tempfile.TemporaryDirectory(prefix="gguf_convert_", dir=output_dir) as temp_dir:
                if self.verbose:
                    logger.log_info("📁 Setting up workspace...")
                
                self.progress_tracker = ProgressTracker(total_steps=10)
                
                self.progress_tracker.update(1, "Optimizing model...")
                model = self._optimize_for_conversion(model)
                
                self.progress_tracker.update(2, "Saving model...")
                model.save_pretrained(
                    temp_dir,
                    safe_serialization=True,
                    max_shard_size="2GB"
                )
                
                if save_tokenizer:
                    self.progress_tracker.update(3, "Saving tokenizer...")
                    self._save_tokenizer(model, temp_dir)
                
                self.progress_tracker.update(4, "Converting to FP16 GGUF...")
                temp_gguf = os.path.join(temp_dir, "temp_f16.gguf")
                cmd_convert = self._build_conversion_command(
                    temp_dir, temp_gguf,
                    model_type or self._detect_model_type(model),
                    bits, optimization_level
                )
                
                success = self._run_conversion(cmd_convert, progress_callback)
                if not success or not os.path.exists(temp_gguf):
                    raise RuntimeError("FP16 GGUF conversion failed")
                
                self.progress_tracker.update(7, f"Quantizing to {quant_type or self.QUANT_TYPE_MAP.get(bits)[0]}...")
                cmd_quantize = [
                    self.quantize_bin,
                    temp_gguf,
                    gguf_path,
                    (quant_type or self.QUANT_TYPE_MAP.get(bits)[0]).lower()
                ]
                
                success = self._run_conversion(cmd_quantize, progress_callback)
                if not success or not os.path.exists(gguf_path):
                    raise RuntimeError(f"GGUF quantization to {quant_type} failed")
                
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
        
        essential_config = {
            "model_type": model_type,
            "architectures": getattr(model.config, 'architectures', [model_type]),
            "torch_dtype": str(getattr(model, 'dtype', torch.float32)).replace('torch.', ''),
            "transformers_version": "4.36.0",
        }
        
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
                    use_fast=False
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
        """Build optimized conversion command for FP16."""
        cmd = [
            sys.executable,
            self.convert_script,
            temp_dir,
            "--outfile", gguf_path,
            "--outtype", "f16",
        ]
        
        if optimization_level in self.OPTIMIZATION_FLAGS:
            cmd.extend(self.OPTIMIZATION_FLAGS[optimization_level])
        
        if model_type != "llama":
            cmd.extend(["--model-type", model_type])
        
        cmd.extend([
            "--no-lazy",
            "--big-endian"
        ])
        
        return cmd
    
    def _run_conversion(self, cmd: List[str], progress_callback: Optional[Callable] = None) -> bool:
        """Run conversion or quantization process with enhanced monitoring."""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            monitor_thread = self._log_progress(process, progress_callback)
            
            return_code = process.wait()
            
            if monitor_thread:
                monitor_thread.join(timeout=5.0)
            
            return return_code == 0
            
        except Exception as e:
            logger.log_error(f"Process error: {e}")
            return False

    def get_supported_quantization_types(self, bits: Optional[int] = None) -> Dict[int, List[str]]:
        """Get supported quantization types."""
        if bits:
            return {bits: self.QUANT_TYPE_MAP.get(bits, ["q4_k_m"])}
        return self.QUANT_TYPE_MAP.copy()
