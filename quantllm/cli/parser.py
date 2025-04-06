import argparse
from typing import Any
from .commands import train, evaluate, quantize, serve

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI commands."""
    parser = argparse.ArgumentParser(description="QuantLLM: LLM Quantization and Fine-tuning")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train command
    train_parser = subparsers.add_parser("train", help="Fine-tune a model")
    train_parser.add_argument("--model", required=True, help="Model name or path")
    train_parser.add_argument("--dataset", required=True, help="Dataset name or path")
    train_parser.add_argument("--output-dir", required=True, help="Output directory")
    train_parser.add_argument("--quantization", choices=["4bit", "8bit"], help="Quantization mode")
    train_parser.add_argument("--use-lora", action="store_true", help="Use LoRA for training")
    train_parser.set_defaults(func=train)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--model", required=True, help="Model name or path")
    eval_parser.add_argument("--dataset", required=True, help="Dataset name or path")
    eval_parser.add_argument("--output-file", help="Output file for evaluation results")
    eval_parser.set_defaults(func=evaluate)

    # Quantize command
    quant_parser = subparsers.add_parser("quantize", help="Quantize a model")
    quant_parser.add_argument("--model", required=True, help="Model name or path")
    quant_parser.add_argument("--output-dir", required=True, help="Output directory")
    quant_parser.add_argument("--bits", type=int, choices=[4, 8], required=True, help="Bits for quantization")
    quant_parser.set_defaults(func=quantize)


    return parser