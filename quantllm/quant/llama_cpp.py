"""
llama.cpp Integration for QuantLLM
Handles installation, detection, and usage of llama.cpp tools
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

# llama.cpp specific targets
LLAMA_CPP_TARGETS = [
    "llama-quantize",
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
    """Get the llama.cpp directory path."""
    return os.path.join(os.getcwd(), "llama.cpp")


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
            "llama-quantize not found. llama.cpp may not be properly installed."
        )
    
    if not converter_path:
        raise FileNotFoundError(
            "convert_hf_to_gguf.py not found. llama.cpp may not be properly installed."
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
        gpu_support: Whether to compile with CUDA support (Note: conversion works better without GPU)
        print_output: Whether to print compilation output
        force_reinstall: Force reinstallation even if already exists
        
    Returns:
        Tuple of (quantizer_path, converter_path)
    """
    llama_dir = get_llama_cpp_dir()
    
    # Check if already installed and working
    if not force_reinstall:
        try:
            return check_llama_cpp()
        except FileNotFoundError:
            pass  # Continue with installation
    
    # Remove existing directory if force reinstall
    if force_reinstall and os.path.exists(llama_dir):
        logger.info("Removing existing llama.cpp installation...")
        shutil.rmtree(llama_dir, ignore_errors=True)
    
    # Step 1: Clone llama.cpp
    if not os.path.exists(llama_dir):
        logger.info("Cloning llama.cpp repository...")
        try:
            subprocess.run(
                ["git", "clone", "--recursive", "https://github.com/ggerganov/llama.cpp"],
                check=True,
                stdout=subprocess.DEVNULL if not print_output else None,
                stderr=subprocess.STDOUT if not print_output else None,
            )
            print_success("✓ llama.cpp cloned successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone llama.cpp: {e}")
    
    # Step 2: Install Python dependencies
    logger.info("Installing Python dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "gguf", "protobuf"],
            check=True,
            stdout=subprocess.DEVNULL if not print_output else None,
        )
        print_success("✓ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print_warning(f"Failed to install some dependencies: {e}")
    
    # Step 3: Compile llama.cpp
    logger.info("Compiling llama.cpp (this may take 3-5 minutes)...")
    
    n_cpus = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    n_jobs = max(int(n_cpus * 1.5), 1)
    
    # Try CMAKE first (newer method)
    try:
        is_cmake = _compile_with_cmake(llama_dir, n_jobs, gpu_support, print_output)
        print_success("✓ llama.cpp compiled successfully with CMAKE")
    except Exception as e:
        logger.warning(f"CMAKE compilation failed: {e}")
        logger.info("Trying MAKE compilation...")
        
        # Fallback to MAKE (older method)
        try:
            _compile_with_make(llama_dir, n_jobs, gpu_support, print_output)
            print_success("✓ llama.cpp compiled successfully with MAKE")
        except Exception as e2:
            raise RuntimeError(f"Both CMAKE and MAKE compilation failed. CMAKE: {e}, MAKE: {e2}")
    
    # Step 4: Verify installation
    try:
        quantizer_path, converter_path = check_llama_cpp()
        print_success(f"✓ llama.cpp installed successfully!")
        logger.info(f"  Quantizer: {quantizer_path}")
        logger.info(f"  Converter: {converter_path}")
        return quantizer_path, converter_path
    except FileNotFoundError as e:
        raise RuntimeError(f"llama.cpp compiled but executables not found: {e}")


def _compile_with_cmake(
    llama_dir: str,
    n_jobs: int,
    gpu_support: bool,
    print_output: bool
) -> bool:
    """Compile llama.cpp using CMAKE."""
    # Clean previous builds
    build_dir = os.path.join(llama_dir, "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir, ignore_errors=True)
    
    # Configure CMAKE
    cuda_flag = "-DGGML_CUDA=ON" if gpu_support else "-DGGML_CUDA=OFF"
    
    configure_cmd = [
        "cmake",
        llama_dir,
        "-B", build_dir,
        "-DBUILD_SHARED_LIBS=OFF",
        cuda_flag,
        "-DLLAMA_CURL=ON",
    ]
    
    subprocess.run(
        configure_cmd,
        check=True,
        stdout=subprocess.DEVNULL if not print_output else None,
        stderr=subprocess.STDOUT if not print_output else None,
    )
    
    # Build
    build_cmd = [
        "cmake",
        "--build", build_dir,
        "--config", "Release",
        f"-j{n_jobs}",
        "--clean-first",
        "--target",
    ] + LLAMA_CPP_TARGETS
    
    subprocess.run(
        build_cmd,
        check=True,
        stdout=subprocess.DEVNULL if not print_output else None,
        stderr=subprocess.STDOUT if not print_output else None,
    )
    
    # Copy executables to main directory
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
    """
    # Get quantizer
    if quantizer_location is None:
        quantizer_location, _ = check_llama_cpp()
    
    if not os.path.exists(input_gguf):
        raise FileNotFoundError(f"Input GGUF file not found: {input_gguf}")
    
    # Normalize quant type
    quant_type = quant_type.upper()
    
    # Get CPU count for threading
    n_threads = psutil.cpu_count() or 1
    
    # Build command
    cmd = [
        quantizer_location,
        input_gguf,
        output_gguf,
        quant_type,
        str(n_threads),
    ]
    
    if print_output:
        logger.info(f"Quantizing with {quant_type}...")
        logger.info(f"Command: {' '.join(cmd)}")
    
    # Run quantization
    try:
        if print_output:
            # Stream output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            
            for line in process.stdout:
                print(line, end='', flush=True)
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        else:
            # Silent mode
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        
        if print_output:
            print_success(f"✓ Quantized to {output_gguf}")
        
        return output_gguf
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Quantization failed: {e}")


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
    """
    # Get converter
    if converter_location is None:
        _, converter_location = check_llama_cpp()
    
    if not os.path.exists(converter_location):
        raise FileNotFoundError(f"Converter not found: {converter_location}")
    
    # Determine output filename
    if quantization_type and quantization_type.lower() != "none":
        output_file = f"{model_name}.{quantization_type.upper()}.gguf"
    else:
        output_file = f"{model_name}.{model_dtype.upper()}.gguf"
    
    # Build conversion command
    cmd = [
        sys.executable,
        converter_location,
        input_folder,
        "--outfile", output_file,
        "--outtype", model_dtype,
    ]
    
    if print_output:
        logger.info(f"Converting to GGUF format...")
        logger.info(f"Output: {output_file}")
    
    # Run conversion
    try:
        if print_output:
            # Stream output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            
            for line in process.stdout:
                print(line, end='', flush=True)
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        else:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        
        if print_output:
            print_success(f"✓ Converted to {output_file}")
        
        return [output_file], is_vlm
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Conversion failed: {e}")


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