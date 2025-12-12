"""
Modern GGUF Converter for QuantLLM v2.0

Supports all major open-source LLM architectures with proper
llama.cpp integration for GGUF conversion.

Supported Models:
- Llama 2/3 family
- Mistral/Mixtral
- Qwen/Qwen2
- Phi-1/2/3
- Gemma/Gemma2
- Falcon
- StarCoder/StarCoder2
- MPT
- GPT-NeoX/Pythia
- StableLM
- BLOOM
- OPT
- DeepSeek
- InternLM
- Baichuan
- Yi
- ChatGLM
- And more...
"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import torch
from transformers import (
    PreTrainedModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
)

from ..utils.logger import logger


# Comprehensive model architecture to llama.cpp type mapping
MODEL_TYPE_MAPPING = {
    # Llama family
    "llama": "llama",
    "llama2": "llama",
    "llama3": "llama",
    "codellama": "llama",
    
    # Mistral family
    "mistral": "llama",  # Mistral uses llama architecture
    "mixtral": "llama",
    
    # Qwen family
    "qwen": "qwen",
    "qwen2": "qwen2",
    "qwen2_moe": "qwen2",
    
    # Phi family
    "phi": "phi2",
    "phi2": "phi2", 
    "phi3": "phi3",
    "phi-msft": "phi2",
    
    # Gemma family
    "gemma": "gemma",
    "gemma2": "gemma2",
    
    # Falcon
    "falcon": "falcon",
    "refinedweb": "falcon",
    
    # StarCoder
    "starcoder": "starcoder",
    "starcoder2": "starcoder2",
    "codegen": "starcoder",
    
    # MPT
    "mpt": "mpt",
    
    # GPT-NeoX/Pythia
    "gpt_neox": "gptneox",
    "pythia": "gptneox",
    
    # StableLM
    "stablelm": "stablelm",
    "stablelm_epoch": "stablelm",
    
    # BLOOM
    "bloom": "bloom",
    "bloomz": "bloom",
    
    # OPT
    "opt": "opt",
    
    # GPT2
    "gpt2": "gpt2",
    
    # DeepSeek
    "deepseek": "llama",
    "deepseek_v2": "llama",
    
    # InternLM
    "internlm": "internlm",
    "internlm2": "internlm2",
    
    # Baichuan
    "baichuan": "baichuan",
    "baichuan2": "baichuan",
    
    # Yi
    "yi": "llama",  # Yi uses llama architecture
    
    # ChatGLM
    "chatglm": "chatglm",
    "chatglm2": "chatglm",
    "chatglm3": "chatglm",
    "glm": "chatglm",
    
    # Command-R
    "cohere": "command-r",
    "command-r": "command-r",
    
    # Orion
    "orion": "orion",
    
    # Jamba
    "jamba": "jamba",
    
    # Default fallback
    "default": "llama",
}

# GGUF quantization types with descriptions
GGUF_QUANT_TYPES = {
    # 2-bit
    "Q2_K": {"bits": 2, "desc": "2-bit with mixed K-quant"},
    "IQ2_XXS": {"bits": 2, "desc": "Tiny 2-bit importance quant"},
    "IQ2_XS": {"bits": 2, "desc": "Small 2-bit importance quant"},
    "IQ2_S": {"bits": 2, "desc": "Standard 2-bit importance quant"},
    "IQ2_M": {"bits": 2, "desc": "Medium 2-bit importance quant"},
    
    # 3-bit
    "Q3_K_S": {"bits": 3, "desc": "3-bit small K-quant"},
    "Q3_K_M": {"bits": 3, "desc": "3-bit medium K-quant"},
    "Q3_K_L": {"bits": 3, "desc": "3-bit large K-quant"},
    "IQ3_XXS": {"bits": 3, "desc": "Tiny 3-bit importance quant"},
    "IQ3_XS": {"bits": 3, "desc": "Extra small 3-bit importance quant"},
    "IQ3_S": {"bits": 3, "desc": "Small 3-bit importance quant"},
    "IQ3_M": {"bits": 3, "desc": "Medium 3-bit importance quant"},
    
    # 4-bit (most popular)
    "Q4_0": {"bits": 4, "desc": "Original 4-bit quant"},
    "Q4_1": {"bits": 4, "desc": "4-bit with improved accuracy"},
    "Q4_K_S": {"bits": 4, "desc": "4-bit small K-quant"},
    "Q4_K_M": {"bits": 4, "desc": "4-bit medium K-quant (recommended)"},
    "IQ4_NL": {"bits": 4, "desc": "4-bit non-linear importance quant"},
    "IQ4_XS": {"bits": 4, "desc": "Extra small 4-bit importance quant"},
    
    # 5-bit
    "Q5_0": {"bits": 5, "desc": "5-bit quant"},
    "Q5_1": {"bits": 5, "desc": "5-bit with higher accuracy"},
    "Q5_K_S": {"bits": 5, "desc": "5-bit small K-quant"},
    "Q5_K_M": {"bits": 5, "desc": "5-bit medium K-quant"},
    
    # 6-bit
    "Q6_K": {"bits": 6, "desc": "6-bit K-quant"},
    
    # 8-bit
    "Q8_0": {"bits": 8, "desc": "8-bit quant (highest quality)"},
    
    # Float types
    "F16": {"bits": 16, "desc": "Float16 (no quantization)"},
    "F32": {"bits": 32, "desc": "Float32 (no quantization)"},
    "BF16": {"bits": 16, "desc": "BFloat16"},
}


class GGUFConverter:
    """
    Modern GGUF converter with support for all major open-source LLMs.
    
    Uses the llama.cpp ecosystem for conversion with automatic
    tool detection and fallback options.
    
    Example:
        >>> converter = GGUFConverter()
        >>> converter.convert(
        ...     model="meta-llama/Llama-3-8B",
        ...     output_path="llama3-8b-q4_k_m.gguf",
        ...     quant_type="Q4_K_M"
        ... )
    """
    
    def __init__(self):
        self._convert_script: Optional[str] = None
        self._quantize_bin: Optional[str] = None
        self._llama_cpp_path: Optional[str] = None
        self._find_llama_cpp_tools()
    
    def _find_llama_cpp_tools(self) -> bool:
        """Find llama.cpp conversion tools."""
        # Method 1: Check if llama-cpp-python is installed
        try:
            import llama_cpp
            pkg_path = Path(llama_cpp.__file__).parent
            
            # Look for convert scripts
            possible_convert = [
                pkg_path / "convert.py",
                pkg_path / "convert_hf_to_gguf.py",
                pkg_path.parent / "scripts" / "convert.py",
            ]
            
            for script in possible_convert:
                if script.exists():
                    self._convert_script = str(script)
                    self._llama_cpp_path = str(pkg_path)
                    break
            
            # Look for quantize binary
            possible_quantize = [
                pkg_path / "quantize",
                pkg_path / "quantize.exe",
                pkg_path / "llama-quantize",
                pkg_path / "llama-quantize.exe",
            ]
            
            for qbin in possible_quantize:
                if qbin.exists():
                    self._quantize_bin = str(qbin)
                    break
                    
        except ImportError:
            pass
        
        # Method 2: Check PATH for llama.cpp tools
        if not self._quantize_bin:
            for name in ["llama-quantize", "quantize"]:
                path = shutil.which(name)
                if path:
                    self._quantize_bin = path
                    break
        
        # Method 3: Check common installation locations
        common_paths = [
            Path.home() / "llama.cpp",
            Path.home() / ".local" / "llama.cpp",
            Path("/usr/local/llama.cpp"),
            Path("C:/llama.cpp") if sys.platform == "win32" else None,
        ]
        
        for base in filter(None, common_paths):
            if base and base.exists():
                if not self._convert_script:
                    conv = base / "convert_hf_to_gguf.py"
                    if conv.exists():
                        self._convert_script = str(conv)
                        self._llama_cpp_path = str(base)
                
                if not self._quantize_bin:
                    for name in ["llama-quantize", "quantize", "llama-quantize.exe", "quantize.exe"]:
                        qbin = base / name
                        if qbin.exists():
                            self._quantize_bin = str(qbin)
                            break
        
        return self._convert_script is not None or self._quantize_bin is not None
    
    @property
    def tools_available(self) -> Dict[str, bool]:
        """Check which conversion tools are available."""
        return {
            "convert_script": self._convert_script is not None,
            "quantize_binary": self._quantize_bin is not None,
            "llama_cpp_path": self._llama_cpp_path,
        }
    
    @staticmethod
    def get_model_type(model: Union[str, PreTrainedModel]) -> str:
        """Detect the model architecture type."""
        if isinstance(model, str):
            try:
                config = AutoConfig.from_pretrained(model, trust_remote_code=True)
                model_type = getattr(config, 'model_type', 'unknown').lower()
            except Exception:
                model_type = 'unknown'
        else:
            model_type = getattr(model.config, 'model_type', 'unknown').lower()
        
        # Map to llama.cpp type
        return MODEL_TYPE_MAPPING.get(model_type, MODEL_TYPE_MAPPING['default'])
    
    @staticmethod
    def list_supported_models() -> List[str]:
        """List all supported model architectures."""
        return list(set(MODEL_TYPE_MAPPING.keys()) - {'default'})
    
    @staticmethod
    def list_quant_types(bits: Optional[int] = None) -> Dict[str, Dict]:
        """List available quantization types."""
        if bits is None:
            return GGUF_QUANT_TYPES
        return {k: v for k, v in GGUF_QUANT_TYPES.items() if v['bits'] == bits}
    
    def convert(
        self,
        model: Union[str, PreTrainedModel],
        output_path: str,
        quant_type: str = "Q4_K_M",
        *,
        use_imatrix: bool = False,
        imatrix_data: Optional[str] = None,
        keep_temp_files: bool = False,
        verbose: bool = True,
    ) -> str:
        """
        Convert a HuggingFace model to GGUF format.
        
        Args:
            model: HuggingFace model name or PreTrainedModel instance
            output_path: Output path for the GGUF file
            quant_type: GGUF quantization type (Q4_K_M, Q5_K_M, etc.)
            use_imatrix: Use importance matrix for better quality
            imatrix_data: Path to calibration data for imatrix
            keep_temp_files: Keep intermediate files for debugging
            verbose: Print progress information
            
        Returns:
            Path to the generated GGUF file
        """
        if quant_type not in GGUF_QUANT_TYPES:
            raise ValueError(f"Unknown quant type: {quant_type}. Use list_quant_types() to see options.")
        
        temp_dir = None
        temp_f16_gguf = None
        
        try:
            # Step 1: Save model to HF format if needed
            if verbose:
                print(f"Converting to GGUF ({quant_type})...")
            
            if isinstance(model, str):
                model_path = model
                model_type = self.get_model_type(model)
                
                # Check if it's a local path
                if not os.path.exists(model_path):
                    # It's a HF hub model, need to download
                    if verbose:
                        print(f"  Loading model from HuggingFace: {model}")
                    
                    loaded_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    
                    # Save to temp directory
                    temp_dir = tempfile.mkdtemp(prefix="quantllm_gguf_")
                    model_path = temp_dir
                    
                    if verbose:
                        print(f"  Saving to temporary directory...")
                    loaded_model.save_pretrained(model_path, safe_serialization=True)
                    
                    # Also save tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                    tokenizer.save_pretrained(model_path)
                    
                    del loaded_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
            else:
                # Model is already loaded
                model_type = self.get_model_type(model)
                temp_dir = tempfile.mkdtemp(prefix="quantllm_gguf_")
                model_path = temp_dir
                
                if verbose:
                    print(f"  Saving model to temporary directory...")
                model.save_pretrained(model_path, safe_serialization=True)
                
                # Try to save tokenizer if available
                if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            model.config._name_or_path, 
                            trust_remote_code=True
                        )
                        tokenizer.save_pretrained(model_path)
                    except Exception:
                        pass  # Tokenizer not required for conversion
            
            # Step 2: Convert to F16 GGUF
            if verbose:
                print(f"  Converting to FP16 GGUF (model type: {model_type})...")
            
            # Determine output type (F16 for intermediate, final type for direct conversion)
            intermediate_type = "f16" if quant_type not in ["F16", "F32", "BF16"] else quant_type.lower()
            
            if quant_type in ["F16", "F32", "BF16"]:
                # No quantization needed, convert directly
                temp_f16_gguf = output_path
            else:
                temp_f16_gguf = output_path.replace(".gguf", "_f16.gguf")
                if temp_f16_gguf == output_path:
                    temp_f16_gguf = output_path + "_f16.gguf"
            
            # Use Python-based conversion
            success = self._convert_hf_to_gguf(
                model_path, 
                temp_f16_gguf, 
                model_type,
                outtype=intermediate_type,
                verbose=verbose
            )
            
            if not success:
                raise RuntimeError("Failed to convert model to GGUF format")
            
            # Step 3: Quantize if needed
            if quant_type not in ["F16", "F32", "BF16"]:
                if verbose:
                    print(f"  Applying {quant_type} quantization...")
                
                success = self._quantize_gguf(
                    temp_f16_gguf,
                    output_path,
                    quant_type,
                    use_imatrix=use_imatrix,
                    imatrix_data=imatrix_data,
                    verbose=verbose
                )
                
                if not success:
                    raise RuntimeError(f"Failed to quantize model to {quant_type}")
            
            if verbose:
                file_size = os.path.getsize(output_path) / (1024**3)
                print(f"  Done! Output: {output_path} ({file_size:.2f} GB)")
            
            return output_path
            
        finally:
            # Cleanup
            if not keep_temp_files:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                if temp_f16_gguf and temp_f16_gguf != output_path and os.path.exists(temp_f16_gguf):
                    os.remove(temp_f16_gguf)
    
    def _convert_hf_to_gguf(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        outtype: str = "f16",
        verbose: bool = True,
    ) -> bool:
        """Convert HuggingFace model to GGUF using available tools."""
        
        # Method 1: Use convert script if available
        if self._convert_script:
            try:
                cmd = [
                    sys.executable,
                    self._convert_script,
                    model_path,
                    "--outfile", output_path,
                    "--outtype", outtype,
                ]
                
                if verbose:
                    print(f"    Running: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                )
                
                if result.returncode == 0 and os.path.exists(output_path):
                    return True
                else:
                    if verbose:
                        print(f"    Convert script failed: {result.stderr[:500]}")
                        
            except Exception as e:
                if verbose:
                    print(f"    Convert script error: {e}")
        
        # Method 2: Use transformers' built-in GGUF export
        try:
            return self._convert_with_transformers(model_path, output_path, outtype, verbose)
        except Exception as e:
            if verbose:
                print(f"    Transformers export failed: {e}")
        
        # Method 3: Use our custom converter
        try:
            return self._convert_manual(model_path, output_path, model_type, outtype, verbose)
        except Exception as e:
            if verbose:
                print(f"    Manual conversion failed: {e}")
        
        return False
    
    def _convert_with_transformers(
        self,
        model_path: str,
        output_path: str,
        outtype: str,
        verbose: bool,
    ) -> bool:
        """Try to use transformers' GGUF export if available."""
        try:
            from transformers import GGUFConfig
            from transformers.integrations import gguf
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            
            # Export to GGUF
            model.save_pretrained_gguf(output_path)
            
            return os.path.exists(output_path)
            
        except ImportError:
            raise ImportError("transformers GGUF export not available")
    
    def _convert_manual(
        self,
        model_path: str,
        output_path: str,
        model_type: str,
        outtype: str,
        verbose: bool,
    ) -> bool:
        """Manual GGUF conversion as last resort."""
        # This is a placeholder for a pure-Python GGUF writer
        # In practice, you'd implement the GGUF format specification here
        raise NotImplementedError(
            "Pure-Python GGUF conversion not implemented. "
            "Please install llama.cpp tools: https://github.com/ggerganov/llama.cpp"
        )
    
    def _quantize_gguf(
        self,
        input_path: str,
        output_path: str,
        quant_type: str,
        use_imatrix: bool = False,
        imatrix_data: Optional[str] = None,
        verbose: bool = True,
    ) -> bool:
        """Quantize GGUF file to target type."""
        
        if not self._quantize_bin:
            raise RuntimeError(
                "llama-quantize binary not found. "
                "Install llama.cpp or set LLAMA_CPP_PATH environment variable."
            )
        
        cmd = [self._quantize_bin, input_path, output_path, quant_type]
        
        if use_imatrix and imatrix_data:
            cmd.extend(["--imatrix", imatrix_data])
        
        if verbose:
            print(f"    Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True
            else:
                if verbose:
                    print(f"    Quantization failed: {result.stderr[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            if verbose:
                print("    Quantization timed out after 1 hour")
            return False
        except Exception as e:
            if verbose:
                print(f"    Quantization error: {e}")
            return False


# Convenience function
def convert_to_gguf(
    model: Union[str, PreTrainedModel],
    output_path: str,
    quant_type: str = "Q4_K_M",
    **kwargs,
) -> str:
    """
    Convert any model to GGUF format.
    
    This is the simplest way to create GGUF files for llama.cpp,
    Ollama, LM Studio, and other GGUF-compatible runtimes.
    
    Args:
        model: HuggingFace model name or loaded model
        output_path: Output GGUF file path
        quant_type: Quantization type (Q4_K_M, Q5_K_M, etc.)
        **kwargs: Additional arguments passed to GGUFConverter.convert()
        
    Returns:
        Path to the generated GGUF file
        
    Example:
        >>> from quantllm.quant import convert_to_gguf
        >>> 
        >>> # From HuggingFace
        >>> convert_to_gguf("meta-llama/Llama-3-8B", "llama3-q4.gguf")
        >>> 
        >>> # From loaded model
        >>> model = AutoModelForCausalLM.from_pretrained(...)
        >>> convert_to_gguf(model, "model-q4.gguf", quant_type="Q4_K_M")
    """
    converter = GGUFConverter()
    return converter.convert(model, output_path, quant_type, **kwargs)
