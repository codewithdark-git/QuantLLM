"""
QuantLLM v2.0 - Export Example

Shows how to export models to multiple formats.
"""

from quantllm import turbo
from quantllm.core import UniversalExporter, ExportFormat


def main():
    # ============================================
    # LOAD MODEL
    # ============================================
    print("Loading model...")
    model = turbo("microsoft/phi-2", bits=4)
    
    # ============================================
    # EXPORT TO DIFFERENT FORMATS
    # ============================================
    
    # 1. GGUF (for llama.cpp, Ollama, LM Studio)
    print("\n--- GGUF Export ---")
    print("Supported quant types: Q4_K_M, Q5_K_M, Q2_K, Q8_0, etc.")
    # model.export("gguf", "phi2-q4.gguf", quantization="Q4_K_M")
    
    # 2. SafeTensors (for HuggingFace)
    print("\n--- SafeTensors Export ---")
    # model.export("safetensors", "./phi2-safetensors/")
    
    # 3. ONNX (for ONNX Runtime, TensorRT)
    print("\n--- ONNX Export ---")
    # model.export("onnx", "phi2.onnx")
    
    # 4. MLX (for Apple Silicon Macs)
    print("\n--- MLX Export ---")
    # model.export("mlx", "./phi2-mlx/", quantization="4bit")
    
    # ============================================
    # USING UNIVERSAL EXPORTER DIRECTLY
    # ============================================
    print("\n--- Universal Exporter ---")
    print("Available formats:", UniversalExporter.list_formats())
    
    # exporter = UniversalExporter(model.model, model.tokenizer)
    # exporter.export(ExportFormat.GGUF, "model.gguf", quantization="Q4_K_M")
    
    print("\nDone! Uncomment the export lines to save models.")


if __name__ == "__main__":
    main()
