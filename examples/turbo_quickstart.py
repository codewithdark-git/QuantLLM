"""
QuantLLM v2.0 - Turbo API Quickstart

This example demonstrates the ultra-simple turbo API.
"""

from quantllm import turbo


def main():
    # ============================================
    # ONE LINE - Load and quantize any model
    # ============================================
    print("Loading model with auto-optimization...")
    
    # This automatically:
    # - Detects your GPU and capabilities
    # - Chooses optimal quantization (4-bit on most GPUs)
    # - Enables Flash Attention if available
    # - Configures memory management
    model = turbo(
        "microsoft/phi-2",  # Use a small model for demo
        bits=4,  # Optional: override quantization bits
    )
    
    print(model)
    
    # ============================================
    # GENERATE TEXT
    # ============================================
    print("\n--- Text Generation ---")
    
    response = model.generate(
        "Explain the concept of machine learning in simple terms:",
        max_new_tokens=100,
        temperature=0.7,
    )
    
    print(response)
    
    # ============================================
    # CHAT FORMAT
    # ============================================
    print("\n--- Chat Mode ---")
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is Python good for?"},
    ]
    
    response = model.chat(messages, max_new_tokens=100)
    print(response)
    
    # ============================================
    # EXPORT (Optional)
    # ============================================
    print("\n--- Export to GGUF ---")
    
    # Uncomment to export:
    # model.export("gguf", "phi2-q4.gguf", quantization="Q4_K_M")
    # model.export("safetensors", "./phi2-output/")
    
    print("Done! Run the export lines to save your model.")


if __name__ == "__main__":
    main()
