"""
QuantLLM CLI - Simple command-line interface

Usage:
    quantllm version     Show version
    quantllm info        Show system info
    quantllm convert     Convert model to GGUF
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="quantllm",
        description="QuantLLM - Ultra-fast LLM Quantization",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Version command
    subparsers.add_parser("version", help="Show version")
    
    # Info command
    subparsers.add_parser("info", help="Show system info")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model to GGUF")
    convert_parser.add_argument("model", help="Model name or path")
    convert_parser.add_argument("-o", "--output", required=True, help="Output GGUF file")
    convert_parser.add_argument("-q", "--quant", default="q4_0", help="Quantization type (default: q4_0)")
    
    args = parser.parse_args()
    
    if args.command == "version":
        cmd_version()
    elif args.command == "info":
        cmd_info()
    elif args.command == "convert":
        cmd_convert(args.model, args.output, args.quant)
    else:
        parser.print_help()


def cmd_version():
    """Show version."""
    from quantllm import __version__
    print(f"QuantLLM v{__version__}")


def cmd_info():
    """Show system info."""
    import torch
    
    print("\n" + "="*50)
    print(" QuantLLM System Info ".center(50, "="))
    print("="*50)
    
    # PyTorch
    print(f"\n📦 PyTorch: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"🎮 CUDA: {torch.version.cuda}")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   Memory: {mem_gb:.1f} GB")
    else:
        print("🎮 CUDA: Not available")
    
    # Flash Attention
    try:
        import flash_attn
        print(f"⚡ Flash Attention: {flash_attn.__version__}")
    except ImportError:
        print("⚡ Flash Attention: Not installed")
    
    print("\n" + "="*50 + "\n")


def cmd_convert(model: str, output: str, quant: str):
    """Convert model to GGUF."""
    print(f"\n🚀 Converting {model} to GGUF...")
    
    from quantllm import turbo
    
    # Load model
    print(f"📦 Loading model...")
    m = turbo(model)
    
    # Export
    print(f"📦 Exporting with {quant} quantization...")
    m.export("gguf", output, quantization=quant)
    
    print(f"✅ Done! Created: {output}\n")


if __name__ == "__main__":
    main()