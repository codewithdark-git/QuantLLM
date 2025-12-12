"""
QuantLLM GGUF Converter v2.0 - Inspired by Unsloth's approach.

Features:
- Auto-installs llama.cpp (source build or pip wheel)
- Supports 45+ model architectures
- Dynamic quantization with multiple types
- Better error messages and logging
"""

import os
import sys
import subprocess
import shutil
import tempfile
import gc
import importlib.util
from typing import Optional, List, Union, Tuple, Dict, Any
from pathlib import Path

import torch

# Check platform
IS_WINDOWS = sys.platform == "win32"
IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = os.path.exists("/kaggle")

# Allowed quantization types with descriptions
ALLOWED_QUANTS = {
    "f32": "Full precision. Very large files, slowest inference.",
    "f16": "Half precision. Large files, slow inference.",
    "bf16": "Brain float16. Large files, slow inference (requires hardware support).",
    "q8_0": "8-bit quantization. Fast conversion, good quality.",
    "q4_0": "4-bit quantization. Fast inference, smaller files.",
    "q4_1": "4-bit with better accuracy than q4_0.",
    "q4_k_m": "Recommended. Uses Q6_K for half attention, else Q4_K.",
    "q4_k_s": "Uses Q4_K for all tensors.",
    "q5_0": "5-bit quantization. Higher accuracy than q4.",
    "q5_1": "5-bit with even higher accuracy.",
    "q5_k_m": "Recommended. Uses Q6_K for half attention, else Q5_K.",
    "q5_k_s": "Uses Q5_K for all tensors.",
    "q6_k": "6-bit quantization. High quality, larger files.",
    "q2_k": "2-bit quantization. Smallest files, lowest quality.",
    "q3_k_l": "3-bit large. Uses Q5_K for some tensors.",
    "q3_k_m": "3-bit medium. Uses Q4_K for some tensors.",
    "q3_k_s": "3-bit small. Uses Q3_K for all tensors.",
    "q3_k_xs": "3-bit extra small.",
    "iq2_xxs": "2.06 bpw imatrix quantization.",
    "iq2_xs": "2.31 bpw imatrix quantization.",
    "iq3_xxs": "3.06 bpw imatrix quantization.",
}

# Aliases
QUANT_ALIASES = {
    "not_quantized": "f16",
    "fast_quantized": "q8_0",
    "quantized": "q4_k_m",
    "q4_k": "q4_k_m",
    "q5_k": "q5_k_m",
}


class GGUFExporter:
    """
    High-level GGUF exporter that auto-installs llama.cpp.
    
    Example:
        >>> exporter = GGUFExporter()
        >>> exporter.export(model, tokenizer, "output.gguf", "q4_k_m")
    """
    
    def __init__(
        self,
        llama_cpp_path: Optional[str] = None,
        auto_install: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the GGUF exporter.
        
        Args:
            llama_cpp_path: Path to llama.cpp directory (auto-detected if None)
            auto_install: Automatically install llama.cpp if not found
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.llama_cpp_path = llama_cpp_path
        self.auto_install = auto_install
        self._quantize_bin = None
        self._convert_script = None
        self._use_python_lib = False
        
        # Try to find existing installation
        self._find_llama_cpp()
    
    def _log(self, msg: str, level: str = "info"):
        """Print log message."""
        if not self.verbose:
            return
        icons = {"info": "📦", "success": "✅", "warning": "⚠️", "error": "❌"}
        print(f"{icons.get(level, '  ')} {msg}")
    
    def _find_llama_cpp(self):
        """Find llama.cpp installation (binary or python module)."""
        # 1. Check for python package `llama_cpp`
        if importlib.util.find_spec("llama_cpp"):
            self._use_python_lib = True
            
        # 2. Check for binaries
        search_paths = [
            self.llama_cpp_path,
            "llama.cpp",
            "./llama.cpp",
            "../llama.cpp",
            os.path.expanduser("~/llama.cpp"),
        ]
        
        if IS_WINDOWS:
            quantize_names = ["llama-quantize.exe", "quantize.exe"]
        else:
            quantize_names = ["llama-quantize", "quantize"]
        
        # Check PATH
        for name in quantize_names:
            result = shutil.which(name)
            if result:
                self._quantize_bin = result
                self._log(f"Found {name} in PATH")
                return True
        
        # Check folders
        for base_path in search_paths:
            if not base_path or not os.path.exists(base_path):
                continue
            
            # Check multiple locations
            for name in quantize_names:
                for sub in ["", "build/bin", "build/bin/Release"]:
                    path = os.path.join(base_path, sub, name)
                    if os.path.exists(path):
                        self._quantize_bin = path
                        self.llama_cpp_path = base_path
                        self._log(f"Found llama.cpp at {base_path}")
                        return True
            
            # Check for convert script
            convert_script = os.path.join(base_path, "convert_hf_to_gguf.py")
            if os.path.exists(convert_script):
                self._convert_script = convert_script
        
        return self._quantize_bin is not None
    
    def install_llama_cpp(self, force: bool = False) -> bool:
        """
        Install llama.cpp by cloning and building, OR via pip.
        
        Args:
            force: Force reinstall even if exists
            
        Returns:
            True if installation succeeded
        """
        if (self._quantize_bin or self._use_python_lib) and not force:
            self._log("llama.cpp already installed", "success")
            return True
        
        self._log("Installing llama.cpp...")
        
        # Strategy 1: Try building from source (better performance usually)
        build_success = False
        try:
            build_success = self._install_from_source(force)
        except Exception as e:
            self._log(f"Source build failed: {e}", "warning")
        
        if build_success:
             self._find_llama_cpp()
             if self._quantize_bin:
                 return True
        
        # Strategy 2: PIP Install (Pre-built wheels)
        self._log("Falling back to pip install llama-cpp-python...", "info")
        try:
            # Uninstall first to avoid conflicts if needed, or just install
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "--no-cache-dir", "llama-cpp-python"],
                capture_output=True,
                check=True
            )
            # Verify
            if importlib.util.find_spec("llama_cpp"):
                self._use_python_lib = True
                self._log("Installed llama-cpp-python successfully!", "success")
                return True
        except Exception as e:
            self._log(f"Pip install failed: {e}", "error")
            
        return False

    def _install_from_source(self, force: bool) -> bool:
        """Clone and build llama.cpp from source."""
        install_dir = "llama.cpp"
        
        # Clone repository
        if not os.path.exists(install_dir) or force:
            if os.path.exists(install_dir) and force:
                shutil.rmtree(install_dir)
            
            self._log("Cloning llama.cpp repository...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")
        
        # Install Python dependencies
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "gguf", "protobuf"],
            capture_output=True,
        )
        
        # Build llama.cpp
        self._log("Building llama.cpp...")
        
        build_commands = self._get_build_commands(install_dir)
        
        success = False
        for cmd in build_commands:
            result = subprocess.run(
                cmd,
                shell=isinstance(cmd, str),
                capture_output=True,
                text=True,
                cwd=install_dir if not isinstance(cmd, str) else None,
            )
            if result.returncode == 0:
                success = True
                break # Stop if one method works
        
        return success
    
    def _get_build_commands(self, install_dir: str) -> List:
        """Get platform-specific build commands."""
        import psutil
        n_jobs = max(psutil.cpu_count() or 1, 1)
        
        if IS_WINDOWS:
            return [
                f"cmake -B build -DBUILD_SHARED_LIBS=OFF",
                f"cmake --build build --config Release -j {n_jobs}",
            ]
        else:
            return [
                ["make", "clean"],
                ["make", "all", f"-j{n_jobs}"],
            ]
    
    def export(
        self,
        model,
        tokenizer,
        output_path: str,
        quant_type: str = "q4_k_m",
        model_name: Optional[str] = None,
    ) -> str:
        """
        Export a HuggingFace model to GGUF format.
        
        Args:
            model: HuggingFace model (or path to saved model)
            tokenizer: HuggingFace tokenizer
            output_path: Output GGUF file path
            quant_type: Quantization type (q4_k_m, q8_0, etc.)
            model_name: Optional model name for metadata
            
        Returns:
            Path to the generated GGUF file
        """
        # Normalize quant type
        quant_type = quant_type.lower()
        if quant_type in QUANT_ALIASES:
            quant_type = QUANT_ALIASES[quant_type]
        
        if quant_type not in ALLOWED_QUANTS:
            raise ValueError(
                f"Invalid quantization type: {quant_type}\n"
                f"Supported types: {list(ALLOWED_QUANTS.keys())}"
            )
        
        self._log(f"Exporting to GGUF with {quant_type.upper()} quantization...")
        
        # Ensure llama.cpp is installed
        if not (self._quantize_bin or self._use_python_lib):
            if self.auto_install:
                if not self.install_llama_cpp():
                    raise RuntimeError(
                        "Failed to install llama.cpp. Please install manually:\n"
                        "  pip install llama-cpp-python\n"
                        "  OR\n"
                        "  git clone https://github.com/ggerganov/llama.cpp\n"
                        "  cd llama.cpp && make -j"
                    )
            else:
                raise RuntimeError(
                    "llama.cpp not found. Set auto_install=True or install manually."
                )
        
        # Create temp directory for model
        with tempfile.TemporaryDirectory(prefix="quantllm_gguf_") as temp_dir:
            # Save model and tokenizer
            self._log("Saving model to temporary directory...")
            
            if hasattr(model, 'save_pretrained'):
                # It's a HuggingFace model
                model.save_pretrained(temp_dir, safe_serialization=True)
                tokenizer.save_pretrained(temp_dir)
            elif isinstance(model, str) and os.path.isdir(model):
                # It's a path to a saved model
                temp_dir = model
            else:
                raise ValueError("model must be a HuggingFace model or path to saved model")
            
            # Step 1: Convert to FP16 GGUF
            fp16_path = os.path.join(temp_dir, "model-fp16.gguf")
            self._log("Converting to FP16 GGUF...")
            
            success = self._convert_to_gguf(temp_dir, fp16_path)
            if not success:
                raise RuntimeError(
                    "GGUF conversion failed. Please check if your model architecture is supported."
                )
            
            # Step 2: Quantize if needed
            if quant_type not in ["f16", "f32", "bf16"]:
                self._log(f"Quantizing to {quant_type.upper()}...")
                success = self._quantize_gguf(fp16_path, output_path, quant_type)
                if not success:
                    raise RuntimeError(f"Failed to quantize to {quant_type}")
                # Clean up FP16 file
                try:
                    os.remove(fp16_path)
                except:
                    pass
            else:
                shutil.copy(fp16_path, output_path)
        
        self._log(f"GGUF export complete: {output_path}", "success")
        return output_path
    
    def _convert_to_gguf(self, input_dir: str, output_path: str) -> bool:
        """Convert HF model to GGUF format."""
        # Try using the convert script from llama.cpp
        convert_script = None
        
        if self.llama_cpp_path:
            script_path = os.path.join(self.llama_cpp_path, "convert_hf_to_gguf.py")
            if os.path.exists(script_path):
                convert_script = script_path
        
        if not convert_script:
            # Try to find convert script in common locations
            for path in ["llama.cpp/convert_hf_to_gguf.py", "convert_hf_to_gguf.py"]:
                if os.path.exists(path):
                    convert_script = path
                    break
        
        if convert_script:
            # Use the official convert script
            cmd = [
                sys.executable, convert_script,
                input_dir,
                "--outfile", output_path,
                "--outtype", "f16",
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return True
            else:
                self._log(f"Convert script failed: {result.stderr[:500]}", "warning")
        
        # Fallback: Try using gguf Python package directly
        try:
            return self._convert_with_gguf_python(input_dir, output_path)
        except Exception as e:
            self._log(f"Python GGUF conversion failed: {e}", "warning")
        
        return False
    
    def _convert_with_gguf_python(self, input_dir: str, output_path: str) -> bool:
        """Convert using the gguf Python package directly."""
        try:
            import gguf
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            import numpy as np
            
            self._log("Using pure Python GGUF conversion...")
            
            # Load model config
            config = AutoConfig.from_pretrained(input_dir, trust_remote_code=True)
            
            # Determine architecture
            arch = config.architectures[0] if config.architectures else "LlamaForCausalLM"
            
            # Map to GGUF architecture
            gguf_arch = self._get_gguf_architecture(arch)
            
            # Create GGUF writer
            writer = gguf.GGUFWriter(output_path, gguf_arch)
            
            # Add metadata
            writer.add_name(config._name_or_path or "model")
            writer.add_context_length(getattr(config, 'max_position_embeddings', 4096))
            writer.add_embedding_length(getattr(config, 'hidden_size', 4096))
            writer.add_feed_forward_length(getattr(config, 'intermediate_size', 11008))
            writer.add_block_count(getattr(config, 'num_hidden_layers', 32))
            writer.add_head_count(getattr(config, 'num_attention_heads', 32))
            writer.add_head_count_kv(getattr(config, 'num_key_value_heads', 32))
            writer.add_layer_norm_rms_eps(getattr(config, 'rms_norm_eps', 1e-5))
            
            # Load and add tensors
            model = AutoModelForCausalLM.from_pretrained(
                input_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            
            for name, param in model.named_parameters():
                tensor_name = self._convert_tensor_name(name)
                data = param.detach().cpu().numpy().astype(np.float16)
                writer.add_tensor(tensor_name, data)
            
            # Write file
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            
            # Cleanup
            del model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return True
            
        except Exception as e:
            self._log(f"Pure Python GGUF failed: {e}", "error")
            return False
    
    def _get_gguf_architecture(self, hf_arch: str) -> str:
        """Map HuggingFace architecture to GGUF architecture."""
        mapping = {
            "LlamaForCausalLM": "llama",
            "MistralForCausalLM": "llama",
            "Qwen2ForCausalLM": "qwen2",
            "Phi3ForCausalLM": "phi3",
            "PhiForCausalLM": "phi2",
            "GemmaForCausalLM": "gemma",
            "Gemma2ForCausalLM": "gemma2",
            "StableLmForCausalLM": "stablelm",
            "GPT2LMHeadModel": "gpt2",
            "GPTNeoXForCausalLM": "gptneox",
            "FalconForCausalLM": "falcon",
            "MptForCausalLM": "mpt",
            "BloomForCausalLM": "bloom",
            "StarcoderForCausalLM": "starcoder",
        }
        return mapping.get(hf_arch, "llama")
    
    def _convert_tensor_name(self, name: str) -> str:
        """Convert HuggingFace tensor name to GGUF format."""
        # Basic conversion rules
        name = name.replace("model.", "")
        name = name.replace("layers.", "blk.")
        name = name.replace("self_attn.q_proj", "attn_q")
        name = name.replace("self_attn.k_proj", "attn_k")
        name = name.replace("self_attn.v_proj", "attn_v")
        name = name.replace("self_attn.o_proj", "attn_output")
        name = name.replace("mlp.gate_proj", "ffn_gate")
        name = name.replace("mlp.up_proj", "ffn_up")
        name = name.replace("mlp.down_proj", "ffn_down")
        name = name.replace("input_layernorm", "attn_norm")
        name = name.replace("post_attention_layernorm", "ffn_norm")
        name = name.replace("embed_tokens", "token_embd")
        name = name.replace("lm_head", "output")
        name = name.replace("norm", "output_norm")
        name = name.replace(".weight", ".weight")
        return name
    
    def _quantize_gguf(self, input_path: str, output_path: str, quant_type: str) -> bool:
        """Quantize GGUF file."""
        # Method 1: Use binary
        if self._quantize_bin:
            cmd = [self._quantize_bin, input_path, output_path, quant_type.upper()]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                 return True
            else:
                 self._log(f"Binary quantization failed: {result.stderr[:200]}", "warning")
        
        # Method 2: Use python library
        if self._use_python_lib:
            try:
                # Warning: calling quantization via dll/so not officially exposed in high level API
                # but we can assume if this library is present, user can use it manually.
                # However, for now we just show a warning if binary is missing.
                # NOTE: llama-cpp-python > 0.2.x exposes `llama_model_quantize`
                import llama_cpp
                # Mapping string type to int is required, which is internal.
                # For safety, since binary conversion is preferred, we return False here
                # unless we reverse engineer the params. 
                self._log("Quantization via python library not fully implemented - installing binary is recommended", "warning")
            except ImportError:
                pass
                
        return False


def export_to_gguf(
    model,
    tokenizer,
    output_path: str,
    quant_type: str = "q4_k_m",
    auto_install: bool = True,
    verbose: bool = True,
) -> str:
    """
    One-line GGUF export function.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer  
        output_path: Output GGUF file path
        quant_type: Quantization type (q4_k_m, q8_0, f16, etc.)
        auto_install: Automatically install llama.cpp if needed
        verbose: Print progress messages
        
    Returns:
        Path to the generated GGUF file
        
    Example:
        >>> from quantllm import turbo
        >>> model = turbo("meta-llama/Llama-3-8B")
        >>> export_to_gguf(model.model, model.tokenizer, "llama3.gguf")
    """
    exporter = GGUFExporter(auto_install=auto_install, verbose=verbose)
    return exporter.export(model, tokenizer, output_path, quant_type)


def print_quantization_methods():
    """Print all available quantization methods with descriptions."""
    print("\n📦 Available GGUF Quantization Methods:\n")
    print("-" * 60)
    for method, description in ALLOWED_QUANTS.items():
        print(f"  {method:<12} - {description}")
    print("-" * 60)
    print("\nRecommended: q4_k_m (good balance) or q8_0 (higher quality)\n")


# Export
__all__ = [
    "GGUFExporter",
    "export_to_gguf",
    "print_quantization_methods",
    "ALLOWED_QUANTS",
]
