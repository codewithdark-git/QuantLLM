#!/usr/bin/env python3
"""
QuantLLM Framework Integration Examples

This script demonstrates how to integrate QuantLLM with popular ML frameworks
and libraries for various use cases.
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
from quantllm import QuantLLM

def setup_logging():
    """Set up logging for the examples."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def langchain_integration_example():
    """
    Example: Integrating QuantLLM with LangChain
    
    LangChain is a popular framework for building LLM applications.
    """
    print("\n" + "="*60)
    print("LANGCHAIN INTEGRATION EXAMPLE")
    print("="*60)
    
    try:
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        from typing import Optional, List, Any
    except ImportError:
        print("LangChain not installed. Install with: pip install langchain")
        return None
    
    # First, quantize a model
    model_name = "gpt2"
    output_dir = "./integration_models/langchain"
    
    print(f"Quantizing {model_name} for LangChain integration...")
    
    result = QuantLLM.quantize(
        model=model_name,
        method="gguf",
        bits=4,
        optimization_target="balanced",
        output_dir=output_dir
    )
    
    # Create custom LangChain LLM wrapper
    class QuantLLMLangChain(LLM):
        """Custom LangChain wrapper for QuantLLM models."""
        
        model_path: str
        max_tokens: int = 100
        temperature: float = 0.7
        
        def __init__(self, model_path: str, **kwargs):
            super().__init__(**kwargs)
            self.model_path = model_path
            self._load_model()
        
        def _load_model(self):
            """Load the quantized model."""
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        @property
        def _llm_type(self) -> str:
            return "quantllm"
        
        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            """Generate text using the quantized model."""
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=len(inputs.input_ids[0]) + self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from the generated text
            generated_text = generated_text[len(prompt):].strip()
            
            # Handle stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
            
            return generated_text
    
    # Initialize the LangChain LLM
    llm = QuantLLMLangChain(model_path=output_dir)
    
    # Example usage with LangChain
    print("\nTesting LangChain integration...")
    
    # Simple generation
    prompt = "The future of artificial intelligence is"
    response = llm(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # Save integration example
    integration_code = f'''#!/usr/bin/env python3
"""
LangChain integration example for QuantLLM
"""

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QuantLLMLangChain(LLM):
    """Custom LangChain wrapper for QuantLLM models."""
    
    model_path: str = "{output_dir}"
    max_tokens: int = 100
    temperature: float = 0.7
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @property
    def _llm_type(self) -> str:
        return "quantllm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs.input_ids[0]) + self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()
        
        if stop:
            for stop_seq in stop:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
        
        return generated_text

# Usage example
if __name__ == "__main__":
    llm = QuantLLMLangChain()
    response = llm("The future of AI is")
    print(response)
'''
    
    with open(f"{output_dir}/langchain_integration.py", 'w') as f:
        f.write(integration_code)
    
    print(f"LangChain integration example saved to: {output_dir}/langchain_integration.py")
    
    return result

def gradio_interface_example():
    """
    Example: Creating a Gradio interface for QuantLLM models
    
    Gradio provides an easy way to create web interfaces for ML models.
    """
    print("\n" + "="*60)
    print("GRADIO INTERFACE EXAMPLE")
    print("="*60)
    
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        return None
    
    # Quantize model for Gradio interface
    model_name = "gpt2"
    output_dir = "./integration_models/gradio"
    
    print(f"Quantizing {model_name} for Gradio interface...")
    
    result = QuantLLM.quantize(
        model=model_name,
        method="gguf",
        bits=4,
        optimization_target="speed",  # Optimize for interactive use
        output_dir=output_dir
    )
    
    # Create Gradio interface code
    gradio_code = f'''#!/usr/bin/env python3
"""
Gradio interface for QuantLLM models
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class QuantLLMGradio:
    def __init__(self, model_path="{output_dir}"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the quantized model."""
        print("Loading quantized model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully!")
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """Generate text using the quantized model."""
        if not prompt.strip():
            return "Please enter a prompt."
        
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=len(inputs.input_ids[0]) + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt
            response = generated_text[len(prompt):].strip()
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Add generation info
            info = f"\\n\\n---\\nGeneration time: {{generation_time:.2f}}s"
            
            return response + info
            
        except Exception as e:
            return f"Error generating text: {{str(e)}}"

# Initialize the model
quantllm_gradio = QuantLLMGradio()

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="QuantLLM Text Generation") as interface:
        gr.Markdown("# QuantLLM Text Generation Interface")
        gr.Markdown("Generate text using a quantized language model")
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                
                with gr.Row():
                    max_length_slider = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Max Length"
                    )
                    
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                
                generate_btn = gr.Button("Generate Text", variant="primary")
                clear_btn = gr.Button("Clear")
            
            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=10,
                    max_lines=20
                )
        
        # Event handlers
        generate_btn.click(
            fn=quantllm_gradio.generate_text,
            inputs=[prompt_input, max_length_slider, temperature_slider],
            outputs=output_text
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[prompt_input, output_text]
        )
        
        # Example prompts
        gr.Examples(
            examples=[
                ["The future of artificial intelligence is"],
                ["In a world where technology advances rapidly,"],
                ["The most important aspect of machine learning is"],
                ["Once upon a time, in a distant galaxy,"],
                ["The benefits of model quantization include"]
            ],
            inputs=prompt_input
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        debug=True
    )
'''
    
    with open(f"{output_dir}/gradio_interface.py", 'w') as f:
        f.write(gradio_code)
    
    # Create requirements file for Gradio
    requirements = '''gradio>=4.0.0
torch>=2.0.0
transformers>=4.35.0
quantllm
'''
    
    with open(f"{output_dir}/requirements.txt", 'w') as f:
        f.write(requirements)
    
    print(f"Gradio interface example saved to: {output_dir}/gradio_interface.py")
    print("To run the interface:")
    print(f"  cd {output_dir}")
    print("  pip install -r requirements.txt")
    print("  python gradio_interface.py")
    
    return result

def main():
    """Run framework integration examples."""
    logger = setup_logging()
    
    print("QuantLLM Framework Integration Examples")
    print("=" * 60)
    print("This script demonstrates integration with popular ML frameworks")
    
    # Create output directories
    os.makedirs("./integration_models", exist_ok=True)
    
    try:
        # Run integration examples
        logger.info("Starting LangChain integration example...")
        langchain_result = langchain_integration_example()
        
        logger.info("Starting Gradio interface example...")
        gradio_result = gradio_interface_example()
        
        print("\n" + "="*60)
        print("INTEGRATION EXAMPLES COMPLETED!")
        print("="*60)
        print("\nGenerated integrations:")
        print("- ./integration_models/langchain/ - LangChain integration")
        print("- ./integration_models/gradio/ - Gradio web interface")
        
        print("\nIntegration highlights:")
        print("1. LangChain: Custom LLM wrapper for chain-based applications")
        print("2. Gradio: Quick interactive web interfaces")
        
    except Exception as e:
        logger.error(f"Integration example failed: {e}")
        raise

if __name__ == "__main__":
    main()