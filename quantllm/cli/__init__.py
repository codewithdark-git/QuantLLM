"""
QuantLLM CLI - Command line interface for QuantLLM v2.0.

Provides both simple info commands and advanced training/quantization commands.
"""

import argparse
import sys

# Import legacy commands 
try:
    from .commands import train, evaluate, quantize, serve
    from .parser import create_parser
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False


def main():
    """Main CLI entry point for QuantLLM."""
    parser = argparse.ArgumentParser(
        description="QuantLLM v2.0 - Ultra-fast LLM quantization and deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quantllm version        Show version info
  quantllm info           Show system info
  quantllm train          Train a model (legacy)
  quantllm quantize       Quantize a model (legacy)
  
Python API (recommended):
  from quantllm import turbo
  model = turbo("meta-llama/Llama-3-8B")
  model.generate("Hello!")
  model.finetune("data.json")
  model.export("gguf", "model.gguf")
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system info")
    
    # Legacy commands if available
    if LEGACY_AVAILABLE:
        train_parser = subparsers.add_parser("train", help="Train a model")
        train_parser.set_defaults(func=train)
        
        quant_parser = subparsers.add_parser("quantize", help="Quantize a model")
        quant_parser.set_defaults(func=quantize)
        
        eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
        eval_parser.set_defaults(func=evaluate)
        
        serve_parser = subparsers.add_parser("serve", help="Serve a model")
        serve_parser.set_defaults(func=serve)
    
    args = parser.parse_args()
    
    if args.command == "version":
        _show_version()
    elif args.command == "info":
        _show_info()
    elif hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def _show_version():
    """Show version information."""
    try:
        from quantllm import __version__
        print(f"QuantLLM v{__version__}")
    except ImportError:
        print("QuantLLM (version unknown)")


def _show_info():
    """Show system information."""
    try:
        from quantllm import __version__
        import torch
        
        print(f"QuantLLM v{__version__}")
        print("-" * 40)
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"GPU Memory: {props.total_memory / 1024**3:.1f} GB")
        
        # Check optional dependencies
        print("-" * 40)
        print("Optional dependencies:")
        
        try:
            import triton
            print(f"  Triton: {triton.__version__}")
        except ImportError:
            print("  Triton: Not installed")
        
        try:
            import flash_attn
            print(f"  Flash Attention: {flash_attn.__version__}")
        except ImportError:
            print("  Flash Attention: Not installed")
        
        try:
            from hf_lifecycle import HFManager
            print("  hf_lifecycle: Installed")
        except ImportError:
            print("  hf_lifecycle: Not installed")
            
    except Exception as e:
        print(f"Error getting info: {e}")


__all__ = ["main"]

if __name__ == "__main__":
    main()