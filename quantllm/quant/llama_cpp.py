"""
llama.cpp Integration for QuantLLM
Handles installation, detection, and usage of llama.cpp tools

Optimizations:
- Uses shallow clone (--depth 1) for faster download
- Parallel compilation with max CPU usage
- Caches compiled binaries
- Only compiles necessary targets
"""

import os
import sys
import subprocess
import shutil
import psutil
from pathlib import Path
from typing import Optional, Tuple, List
import tempfile
from ..utils import logger, print_success, print_error, print_warning, print_info

# llama.cpp specific targets - only what we need
LLAMA_CPP_TARGETS = [
    "llama-quantize",  # Main quantization tool
]

# Optional targets (not compiled by default for speed)
LLAMA_CPP_OPTIONAL_TARGETS = [
    "llama-export-lora", 
    "llama-cli",
]

# Determine environment
def detect_environment():
    """Detect if running in special environments."""
    keynames = "\n" + "\n".join(os.environ.keys())
    is_colab = "\nCOLAB_" in keynames
    is_kaggle = "\nKAGGLE_" in keynames
    return is_colab, is_kaggle

IS_COLAB_ENVIRONMENT, IS_KAGGLE_ENVIRONMENT = detect_environment()


def get_llama_cpp_dir() -> str:
    """Get the llama.cpp directory path (stored in user cache)."""
    # Use user cache directory to avoid polluting CWD and nesting issues
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "quantllm")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "llama.cpp")


def check_llama_cpp() -> Tuple[Optional[str], Optional[str]]:
    """
    Check if llama.cpp is installed and return paths to executables.
    
    Returns:
        Tuple of (quantizer_path, converter_path)
        
    Raises:
        FileNotFoundError: If llama.cpp tools are not found
    """
    llama_dir = get_llama_cpp_dir()
    
    # Check for quantizer executable
    quantizer_candidates = [
        os.path.join(llama_dir, "llama-quantize"),
        os.path.join(llama_dir, "llama-quantize.exe"),
        os.path.join(llama_dir, "quantize"),
        os.path.join(llama_dir, "quantize.exe"),
        os.path.join(llama_dir, "build", "bin", "llama-quantize"),
        os.path.join(llama_dir, "build", "bin", "quantize"),
        # Common linux install locations if system-wide
        "/usr/local/bin/llama-quantize",
        "/usr/bin/llama-quantize",
    ]
    
    quantizer_path = None
    for candidate in quantizer_candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            quantizer_path = candidate
            break
    
    # Check for converter script
    converter_candidates = [
        os.path.join(llama_dir, "convert_hf_to_gguf.py"),
        os.path.join(llama_dir, "convert-hf-to-gguf.py"),
        os.path.join(llama_dir, "examples", "convert_legacy_llama.py"),
    ]
    
    converter_path = None
    for candidate in converter_candidates:
        if os.path.exists(candidate):
            converter_path = candidate
            break
    
    if not quantizer_path:
        raise FileNotFoundError(
            f"llama-quantize not found in {llama_dir}. llama.cpp may not be properly installed."
        )
    
    if not converter_path:
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found in {llama_dir}. llama.cpp may not be properly installed."
        )
    
    return quantizer_path, converter_path


def install_llama_cpp(
    gpu_support: bool = False,
    print_output: bool = False,
    force_reinstall: bool = False
) -> Tuple[str, str]:
    """
    Install llama.cpp by cloning and compiling.
    
    Args:
        gpu_support: Whether to compile with CUDA support
        print_output: Whether to print compilation output
        force_reinstall: Force reinstallation even if already exists
        
    Returns:
        Tuple of (quantizer_path, converter_path)
    """
    from ..utils import QuantLLMProgress, print_step
    
    llama_dir = get_llama_cpp_dir()
    
    # Check if already installed and working
    if not force_reinstall:
        try:
            return check_llama_cpp()
        except FileNotFoundError:
            pass  # Continue with installation
    
    # Remove existing directory if force reinstall or broken
    if os.path.exists(llama_dir):
        if force_reinstall:
            print_info("Removing existing llama.cpp installation...")
            shutil.rmtree(llama_dir, ignore_errors=True)
        else:
            print_warning("Found broken llama.cpp installation. Reinstalling...")
            shutil.rmtree(llama_dir, ignore_errors=True)
    
    total_steps = 4
    
    # Step 1: Clone llama.cpp
    os.makedirs(os.path.dirname(llama_dir), exist_ok=True)
    
    if not os.path.exists(llama_dir):
        print_step(1, total_steps, "Cloning llama.cpp repository...")
        try:
            with QuantLLMProgress(style="spinner") as progress:
                task = progress.add_task("Cloning llama.cpp...", total=None)
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", llama_dir],
                    check=True,
                    stdout=subprocess.DEVNULL if not print_output else None,
                    stderr=subprocess.STDOUT if not print_output else None,
                )
            print_success("llama.cpp cloned successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone llama.cpp: {e}")
    
    # Step 2: Install Python dependencies
    print_step(2, total_steps, "Installing Python dependencies...")
    try:
        with QuantLLMProgress(style="spinner") as progress:
            task = progress.add_task("Installing gguf, protobuf...", total=None)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "gguf>=0.10.0", "protobuf"],
                check=True,
                stdout=subprocess.DEVNULL if not print_output else None,
            )
        print_success("Dependencies installed")
    except subprocess.CalledProcessError as e:
        print_warning(f"Failed to install some dependencies: {e}")
    
    # Step 3: Compile llama.cpp
    print_step(3, total_steps, "Compiling llama.cpp (this may take 3-5 minutes)...")
    
    n_cpus = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    n_jobs = max(int(n_cpus * 1.5), 1)
    
    print_info(f"Using {n_jobs} parallel jobs for compilation")
    if gpu_support:
        print_info("CUDA support enabled")
    
    # Try CMAKE first (newer method)
    try:
        with QuantLLMProgress(style="spinner") as progress:
            task = progress.add_task("Compiling with CMAKE...", total=None)
            _compile_with_cmake(llama_dir, n_jobs, gpu_support, print_output)
        print_success("llama.cpp compiled successfully with CMAKE")
    except Exception as e:
        print_warning(f"CMAKE compilation failed: {e}")
        print_info("Trying MAKE compilation...")
        
        # Fallback to MAKE (older method)
        try:
            with QuantLLMProgress(style="spinner") as progress:
                task = progress.add_task("Compiling with MAKE...", total=None)
                _compile_with_make(llama_dir, n_jobs, gpu_support, print_output)
            print_success("llama.cpp compiled successfully with MAKE")
        except Exception as e2:
            raise RuntimeError(f"Both CMAKE and MAKE compilation failed. CMAKE: {e}, MAKE: {e2}")
    
    # Step 4: Verify installation
    print_step(4, total_steps, "Verifying installation...")
    try:
        quantizer_path, converter_path = check_llama_cpp()
        print_success("llama.cpp installed successfully!")
        print_info(f"Quantizer: {quantizer_path}")
        print_info(f"Converter: {converter_path}")
        return quantizer_path, converter_path
    except FileNotFoundError as e:
        raise RuntimeError(f"llama.cpp compiled but executables not found: {e}")


def _compile_with_cmake(
    llama_dir: str,
    n_jobs: int,
    gpu_support: bool,
    print_output: bool
) -> bool:
    """
    Compile llama.cpp using CMAKE.
    
    Optimizations:
    - Only builds llama-quantize target (faster)
    - Uses Release mode for speed
    - Parallelizes with maximum CPU cores
    """
    # Check if already compiled
    quantizer_path = os.path.join(llama_dir, "llama-quantize")
    if os.path.exists(quantizer_path) and os.access(quantizer_path, os.X_OK):
        return True  # Already compiled
    
    build_dir = os.path.join(llama_dir, "build")
    
    # Only clean if build exists but is broken
    if os.path.exists(build_dir):
        # Check if build is valid
        bin_path = os.path.join(build_dir, "bin", "llama-quantize")
        if not os.path.exists(bin_path):
            shutil.rmtree(build_dir, ignore_errors=True)
    
    # Configure CMAKE with speed optimizations
    cuda_flag = "-DGGML_CUDA=ON" if gpu_support else "-DGGML_CUDA=OFF"
    
    configure_cmd = [
        "cmake",
        llama_dir,
        "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=Release",  # Release for speed
        "-DBUILD_SHARED_LIBS=OFF",
        cuda_flag,
        "-DLLAMA_CURL=OFF",  # Skip curl to speed up compilation
        "-DLLAMA_BUILD_TESTS=OFF",  # Skip tests
        "-DLLAMA_BUILD_EXAMPLES=OFF",  # Skip examples (we only need quantize)
    ]
    
    subprocess.run(
        configure_cmd,
        check=True,
        stdout=subprocess.DEVNULL if not print_output else None,
        stderr=subprocess.DEVNULL if not print_output else None,
    )
    
    # Build only the quantize target (faster than building everything)
    build_cmd = [
        "cmake",
        "--build", build_dir,
        "--config", "Release",
        f"-j{n_jobs}",
        "--target", "llama-quantize",  # Only build what we need
    ]
    
    subprocess.run(
        build_cmd,
        check=True,
        stdout=subprocess.DEVNULL if not print_output else None,
        stderr=subprocess.DEVNULL if not print_output else None,
    )
    
    # Copy executable to main directory
    bin_dir = os.path.join(build_dir, "bin")
    if os.path.exists(bin_dir):
        for target in LLAMA_CPP_TARGETS:
            src = os.path.join(bin_dir, target)
            if os.path.exists(src):
                shutil.copy2(src, llama_dir)
    
    return True


def _compile_with_make(
    llama_dir: str,
    n_jobs: int,
    gpu_support: bool,
    print_output: bool
):
    """Compile llama.cpp using MAKE (fallback)."""
    # Clean
    subprocess.run(
        ["make", "clean", "-C", llama_dir],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    # Build
    env = os.environ.copy()
    if gpu_support:
        env["LLAMA_CUDA"] = "1"
    
    subprocess.run(
        ["make", "all", f"-j{n_jobs}", "-C", llama_dir],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL if not print_output else None,
        stderr=subprocess.STDOUT if not print_output else None,
    )


def quantize_gguf(
    input_gguf: str,
    output_gguf: str,
    quant_type: str,
    quantizer_location: Optional[str] = None,
    print_output: bool = True,
) -> str:
    """
    Quantize a GGUF file using llama-quantize.
    
    Args:
        input_gguf: Path to input GGUF file
        output_gguf: Path to output quantized GGUF file
        quant_type: Quantization type (e.g., "q4_k_m", "q8_0")
        quantizer_location: Path to llama-quantize executable (auto-detect if None)
        print_output: Whether to print quantization progress
        
    Returns:
        Path to output GGUF file
        
    Supported quantization types:
        - q2_k, q3_k_s, q3_k_m, q3_k_l (very small)
        - q4_k_s, q4_k_m (recommended balance)
        - q5_k_s, q5_k_m (higher quality)
        - q6_k, q8_0 (near full quality)
        - f16, f32 (full precision)
    """
    from ..utils import QuantLLMProgress, print_info
    import re
    
    # Get quantizer
    if quantizer_location is None:
        quantizer_location, _ = check_llama_cpp()
    
    if not os.path.exists(input_gguf):
        raise FileNotFoundError(f"Input GGUF file not found: {input_gguf}")
    
    # Normalize quant type
    quant_type = quant_type.upper()
    
    # Get CPU count for threading
    n_threads = psutil.cpu_count() or 1
    
    # Get input file size for progress estimation
    input_size_mb = os.path.getsize(input_gguf) / (1024 * 1024)
    
    # Build command
    cmd = [
        quantizer_location,
        input_gguf,
        output_gguf,
        quant_type,
        str(n_threads),
    ]
    
    if print_output:
        print_info(f"Quantizing to {quant_type} using {n_threads} threads...")
        print_info(f"Input: {input_gguf} ({input_size_mb:.1f} MB)")
    
    # Run quantization with progress tracking
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        
        output_lines = []
        last_progress = 0
        
        # Pattern to detect progress from llama-quantize output
        # Typically looks like: "[123/456] ..." or percentage patterns
        progress_pattern = re.compile(r'\[(\d+)/(\d+)\]')
        percent_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*%')
        
        if print_output:
            with QuantLLMProgress() as progress:
                task = progress.add_task(f"Quantizing to {quant_type}...", total=100)
                
                for line in process.stdout:
                    output_lines.append(line)
                    
                    # Try to parse progress
                    match = progress_pattern.search(line)
                    if match:
                        current, total = int(match.group(1)), int(match.group(2))
                        pct = (current / total) * 100
                        progress.update(task, completed=pct)
                        last_progress = pct
                    else:
                        # Try percentage pattern
                        match = percent_pattern.search(line)
                        if match:
                            pct = float(match.group(1))
                            progress.update(task, completed=pct)
                            last_progress = pct
                
                progress.update(task, completed=100)
        else:
            for line in process.stdout:
                output_lines.append(line)
        
        process.wait()
        
        if process.returncode != 0:
            error_output = ''.join(output_lines[-20:])
            raise RuntimeError(
                f"Quantization failed with exit code {process.returncode}\n"
                f"Output:\n{error_output}"
            )
        
        if print_output:
            # Show compression stats
            if os.path.exists(output_gguf):
                output_size_mb = os.path.getsize(output_gguf) / (1024 * 1024)
                compression = (1 - output_size_mb / input_size_mb) * 100
                print_success(f"Quantized to {output_gguf}")
                print_info(f"Output size: {output_size_mb:.1f} MB (compressed {compression:.1f}%)")
        
        return output_gguf
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Quantization failed: {e}")
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Unexpected error during quantization: {e}")


def convert_to_gguf(
    model_name: str,
    input_folder: str,
    model_dtype: str,
    quantization_type: str,
    converter_location: Optional[str] = None,
    supported_text_archs: Optional[List[str]] = None,
    supported_vision_archs: Optional[List[str]] = None,
    is_vlm: bool = False,
    is_gpt_oss: bool = False,
    max_shard_size: str = "50GB",
    print_output: bool = True,
) -> Tuple[List[str], bool]:
    """
    Convert HuggingFace model to GGUF using llama.cpp converter.
    
    Args:
        model_name: Name of the model
        input_folder: Folder containing the model
        model_dtype: Data type (f16, bf16)
        quantization_type: Initial quantization type
        converter_location: Path to converter script (auto-detect if None)
        supported_text_archs: List of supported text architectures
        supported_vision_archs: List of supported vision architectures
        is_vlm: Whether this is a vision-language model
        is_gpt_oss: Whether this is a GPT-OSS model
        max_shard_size: Maximum shard size
        print_output: Whether to print conversion progress
        
    Returns:
        Tuple of (list of output files, is_vlm_updated)
        
    Raises:
        RuntimeError: If conversion fails (includes detailed error message)
        FileNotFoundError: If converter script is not found
    """
    # Get converter
    if converter_location is None:
        _, converter_location = check_llama_cpp()
    
    if not os.path.exists(converter_location):
        raise FileNotFoundError(f"Converter not found: {converter_location}")
    
    # Validate input folder contains required files
    config_path = os.path.join(input_folder, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json not found in {input_folder}. "
            "Make sure the model is properly saved in HuggingFace format."
        )
    
    # Check for BitsAndBytes quantized models (they cannot be directly converted)
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'quantization_config' in config:
        quant_config = config['quantization_config']
        quant_method = quant_config.get('quant_method', '')
        if quant_method in ['bitsandbytes', 'bnb'] or quant_config.get('load_in_4bit') or quant_config.get('load_in_8bit'):
            raise RuntimeError(
                "Cannot convert BitsAndBytes quantized model directly to GGUF.\n"
                "The model weights are in BNB format which is incompatible with llama.cpp.\n\n"
                "Solutions:\n"
                "1. Load the model in full precision first:\n"
                "   model = TurboModel.from_pretrained('model_name', quantize=False)\n"
                "   model.export('gguf', quantization='Q4_K_M')\n\n"
                "2. Or use a pre-trained model without BitsAndBytes quantization."
            )
    
    # Determine output filename
    if quantization_type and quantization_type.lower() != "none":
        output_file = f"{model_name}.{quantization_type.upper()}.gguf"
    else:
        output_file = f"{model_name}.{model_dtype.upper()}.gguf"
    
    # Normalize model_dtype for llama.cpp
    dtype_map = {
        "float16": "f16",
        "bfloat16": "bf16",
        "float32": "f32",
        "torch.float16": "f16",
        "torch.bfloat16": "bf16",
        "torch.float32": "f32",
    }
    model_dtype = dtype_map.get(str(model_dtype).lower(), model_dtype)
    
    # Build conversion command
    cmd = [
        sys.executable,
        converter_location,
        input_folder,
        "--outfile", output_file,
        "--outtype", model_dtype,
    ]
    
    if print_output:
        print_info(f"Converting to GGUF format...")
        print_info(f"Output: {output_file}")
    
    # Import progress utilities
    from ..utils import QuantLLMProgress
    import re
    
    # Run conversion with proper error capture and progress tracking
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )
        
        stdout_lines = []
        stderr_lines = []
        
        # Progress patterns for convert_hf_to_gguf.py output
        # Typically shows: "Loading model..." "Writing tensor [X/Y]..."
        tensor_pattern = re.compile(r'\[(\d+)/(\d+)\]')
        writing_pattern = re.compile(r'Writing\s+(\d+)\s+tensors')
        
        if print_output:
            with QuantLLMProgress() as progress:
                task = progress.add_task("Converting to GGUF...", total=100)
                total_tensors = None
                
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    line = line.strip()
                    stdout_lines.append(line)
                    
                    # Try to find total tensors
                    if total_tensors is None:
                        match = writing_pattern.search(line)
                        if match:
                            total_tensors = int(match.group(1))
                    
                    # Try to parse tensor progress
                    match = tensor_pattern.search(line)
                    if match:
                        current, total = int(match.group(1)), int(match.group(2))
                        pct = (current / total) * 100
                        progress.update(task, completed=pct, description=f"Converting tensor [{current}/{total}]...")
                    elif "Loading" in line:
                        progress.update(task, description="Loading model...")
                    elif "Writing" in line:
                        progress.update(task, description="Writing GGUF...")
                
                progress.update(task, completed=100, description="Conversion complete!")
                
                # Read any remaining stderr
                stderr_output = process.stderr.read()
                if stderr_output:
                    stderr_lines = stderr_output.strip().split('\n')
        else:
            stdout, stderr = process.communicate()
            if stdout:
                stdout_lines = stdout.strip().split('\n')
            if stderr:
                stderr_lines = stderr.strip().split('\n')
        
        process.wait()
        
        if process.returncode != 0:
            # Collect all error information
            error_details = []
            if stdout_lines:
                error_details.append("STDOUT:\n" + '\n'.join(stdout_lines[-20:]))
            if stderr_lines:
                error_details.append("STDERR:\n" + '\n'.join(stderr_lines[-20:]))
            
            error_msg = f"GGUF conversion failed with exit code {process.returncode}\n"
            if error_details:
                error_msg += "\n--- Conversion Output ---\n" + '\n'.join(error_details)
            else:
                error_msg += "\nNo error output captured. Try running with print_output=True for more details."
            
            raise RuntimeError(error_msg)
        
        if print_output:
            print_success(f"Converted to {output_file}")
        
        return [output_file], is_vlm
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Conversion failed: {e}")
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Unexpected error during GGUF conversion: {e}")


def use_local_gguf():
    """Context manager to use local GGUF package if needed."""
    class LocalGGUF:
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    return LocalGGUF()


def _download_convert_hf_to_gguf() -> Tuple[str, List[str], List[str]]:
    """
    Ensure converter script exists and return paths.
    
    Returns:
        Tuple of (converter_path, supported_text_archs, supported_vision_archs)
    """
    llama_dir = get_llama_cpp_dir()
    
    # Look for converter
    converter_candidates = [
        os.path.join(llama_dir, "convert_hf_to_gguf.py"),
        os.path.join(llama_dir, "convert-hf-to-gguf.py"),
    ]
    
    converter_path = None
    for candidate in converter_candidates:
        if os.path.exists(candidate):
            converter_path = candidate
            break
    
    if not converter_path:
        raise FileNotFoundError(
            "convert_hf_to_gguf.py not found in llama.cpp directory. "
            "Please reinstall llama.cpp."
        )
    
    # Supported architectures (approximate - llama.cpp supports many)
    supported_text = [
        "llama", "mistral", "mixtral", "phi", "phi3", "gemma", "gemma2",
        "qwen", "qwen2", "gpt2", "falcon", "mpt", "bloom"
    ]
    
    supported_vision = [
        "llava", "clip", "siglip"
    ]
    
    return converter_path, supported_text, supported_vision


def install_llama_cpp_blocking(use_cuda: bool = False, print_output: bool = False):
    """
    Blocking installation of llama.cpp (for backwards compatibility).
    
    Args:
        use_cuda: Whether to enable CUDA support
        print_output: Whether to print compilation output
    """
    return install_llama_cpp(gpu_support=use_cuda, print_output=print_output)


# Convenience function
def ensure_llama_cpp_installed(force_reinstall: bool = False) -> Tuple[str, str]:
    """
    Ensure llama.cpp is installed, install if not.
    
    Args:
        force_reinstall: Force reinstallation even if already exists
        
    Returns:
        Tuple of (quantizer_path, converter_path)
    """
    try:
        if not force_reinstall:
            return check_llama_cpp()
    except FileNotFoundError:
        pass
    
    logger.info("llama.cpp not found. Installing...")
    return install_llama_cpp(gpu_support=False, print_output=True, force_reinstall=force_reinstall)