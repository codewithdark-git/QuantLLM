"""
QuantLLM CLI v2.2 -- Expanded command-line interface.

Subcommands:
    version         Show version
    info            Show system info
    convert         Convert model to GGUF
    finetune        Fine-tune a model
    export          Export to any format
    serve           Start inference server
    bench           Run benchmarks
    compare         Hardware auto-benchmark
    models          Model registry (list, search, info, register)
    register        Register a model on HF Hub
"""

from __future__ import annotations

import argparse
import sys

__version__ = "2.2.0.dev"


def _add_verbose(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")


def _add_model_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="Model name or path")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="quantllm", description="QuantLLM - Ultra-fast LLM Quantization, Fine-tuning & Deployment"
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("version", help="Show version")
    sub.add_parser("info", help="Show system info")

    cp = sub.add_parser("convert", help="Convert model to GGUF")
    _add_model_arg(cp)
    cp.add_argument("-o", "--output", required=True, help="Output GGUF file")
    cp.add_argument("-q", "--quant", default="q4_k_m", help="Quantization type")
    _add_verbose(cp)

    fp = sub.add_parser("finetune", help="Fine-tune a model")
    _add_model_arg(fp)
    fp.add_argument("--dataset", required=True, help="Dataset name or path")
    fp.add_argument("--output-dir", default="./fine-tuned")
    fp.add_argument("--epochs", type=int, default=3)
    fp.add_argument("--lr", type=float, default=2e-4)
    _add_verbose(fp)

    ep = sub.add_parser("export", help="Export model to any format")
    _add_model_arg(ep)
    ep.add_argument("-f", "--format", default="safetensors", choices=["gguf", "safetensors", "onnx", "mlx"])
    ep.add_argument("-o", "--output")
    ep.add_argument("-q", "--quant")
    _add_verbose(ep)

    sp = sub.add_parser("serve", help="Start inference server")
    _add_model_arg(sp)
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--max-seq-length", type=int, default=4096)
    sp.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])

    bp = sub.add_parser("bench", help="Run benchmarks")
    _add_model_arg(bp)
    bp.add_argument("--prompt", default="The meaning of life is")
    bp.add_argument("--max-tokens", type=int, default=128)
    bp.add_argument("--save", action="store_true")
    _add_verbose(bp)

    sub.add_parser("compare", help="Hardware auto-benchmark")

    mp = sub.add_parser("models", help="Model registry commands")
    ms = mp.add_subparsers(dest="models_command")
    ms.add_parser("list", help="List curated models").add_argument("--family")
    ms.add_parser("search", help="Search models").add_argument("query")
    ms.add_parser("info", help="Show model details").add_argument("name")

    rp = sub.add_parser("register", help="Register a model on HF Hub")
    rp.add_argument("model_id", help="HF model ID (e.g. 'QuantLLM/MyModel')")
    rp.add_argument("--family", default="llama")
    rp.add_argument("--params", type=float, default=0.0)
    rp.add_argument("--quality", type=float, default=3.0)
    rp.add_argument("--speed", type=float, default=3.0)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    _route(args)


def _route(args: argparse.Namespace) -> None:
    cmd = args.command
    if cmd == "version":
        print(f"QuantLLM v{__version__}")
        sys.exit(0)
    elif cmd == "info":
        import torch

        print("\n" + "=" * 50 + "\n QuantLLM System Info ".center(50, "=") + "\n" + "=" * 50)
        print(f"\n  QuantLLM: {__version__}\n  PyTorch:  {torch.__version__}")
        if torch.cuda.is_available():
            print(
                f"  CUDA:     {torch.version.cuda}\n  Device:   {torch.cuda.get_device_name(0)}\n  Memory:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
            )
        else:
            print("  CUDA:     Not available")
        print("\n" + "=" * 50)
        sys.exit(0)
    elif cmd == "convert":
        from quantllm.core import turbo

        m = turbo(args.model, verbose=args.verbose)
        m.export("gguf", output_path=args.output, quantization=args.quant)
        print(f"Done: {args.output}")
    elif cmd == "finetune":
        from quantllm.core import turbo

        m = turbo(args.model, verbose=args.verbose)
        m.finetune(args.dataset, epochs=args.epochs, learning_rate=args.lr, output_dir=args.output_dir)
    elif cmd == "export":
        from quantllm.core import turbo

        m = turbo(args.model, verbose=args.verbose)
        print(m.export(args.format, output_path=args.output, quantization=args.quant))
    elif cmd == "serve":
        from quantllm.server import ServerConfig
        from quantllm.server.app import run_server

        run_server(
            ServerConfig.from_cli(
                model=args.model, host=args.host, port=args.port, max_seq_length=args.max_seq_length, dtype=args.dtype
            )
        )
    elif cmd == "bench":
        from quantllm.benchmark import run_benchmark_cli

        run_benchmark_cli(
            args.model, prompt=args.prompt, max_new_tokens=args.max_tokens, save=args.save, verbose=args.verbose
        )
    elif cmd == "compare":
        from quantllm.benchmark.auto_bench import HardwareAutoBench

        print(HardwareAutoBench().generate_report())
    elif cmd == "models":
        sub = args.models_command
        if sub is None:
            print("Usage: quantllm models {list,search,info}")
            return
        from quantllm.registry import list_models, search_models, model_info

        print(
            list_models(family=args.family)
            if sub == "list"
            else search_models(args.query)
            if sub == "search"
            else model_info(args.name)
        )
    elif cmd == "register":
        from quantllm.registry import register_model

        ok = register_model(
            args.model_id, family=args.family, params=args.params, quality_score=args.quality, speed_score=args.speed
        )
        print(f"Model {'registered' if ok else 'registration failed'}: {args.model_id}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
