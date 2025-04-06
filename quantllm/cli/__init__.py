"""Command-line interface for QuantLLM package."""

from .commands import train, evaluate, quantize, serve
from .parser import create_parser

__all__ = ["main", "train", "evaluate", "quantize", "serve"]

def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()