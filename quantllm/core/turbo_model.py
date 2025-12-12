"""
TurboModel - The Ultra-Simple QuantLLM API.

Load, quantize, fine-tune, and export LLMs with one line each.
"""

import os
from typing import Optional, Dict, Any, Union, List
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
)

from .smart_config import SmartConfig
from .hardware import HardwareProfiler


class TurboModel:
    """
    High-performance LLM with the simplest possible API.
    
    Features:
        - One-line loading with automatic quantization
        - One-line fine-tuning with LoRA
        - One-line export to multiple formats
        - Automatic hardware optimization
    
    Example:
        >>> # Load any model in one line
        >>> model = TurboModel.from_pretrained("meta-llama/Llama-3-8B")
        >>> 
        >>> # Generate text
        >>> response = model.generate("Explain quantum computing")
        >>> 
        >>> # Fine-tune with your data
        >>> model.finetune("my_data.json", epochs=3)
        >>> 
        >>> # Export to GGUF
        >>> model.export("gguf", "my_model.gguf")
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: SmartConfig,
    ):
        """
        Initialize TurboModel. Use from_pretrained() instead of direct init.
        
        Args:
            model: The loaded/quantized model
            tokenizer: Associated tokenizer
            config: Configuration used for this model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._is_quantized = False
        self._is_finetuned = False
        self._lora_applied = False
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        # Simple overrides (all optional with smart defaults)
        bits: Optional[int] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        # Advanced options
        trust_remote_code: bool = True,
        quantize: bool = True,
        config_override: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> "TurboModel":
        """
        Load a model with automatic optimization.
        
        This is the main entry point for QuantLLM. It automatically:
        - Detects your hardware capabilities
        - Chooses optimal quantization settings
        - Configures memory management
        - Enables speed optimizations
        
        Args:
            model_name: HuggingFace model name or local path
            bits: Override quantization bits (default: auto-detect)
            max_length: Override max sequence length (default: from model)
            device: Override device (default: best available GPU)
            dtype: Override dtype (default: bf16 if available, else fp16)
            trust_remote_code: Trust remote code in model
            quantize: Whether to quantize the model
            config_override: Dict to override any auto-detected settings
            verbose: Print loading progress
            
        Returns:
            TurboModel ready for inference or fine-tuning
            
        Example:
            >>> # Simplest usage - everything automatic
            >>> model = TurboModel.from_pretrained("meta-llama/Llama-3-8B")
            >>> 
            >>> # With specific bits
            >>> model = TurboModel.from_pretrained("mistral-7b", bits=4)
            >>>
            >>> # For long context
            >>> model = TurboModel.from_pretrained("Qwen2-72B", max_length=32768)
        """
        if verbose:
            print(f"🚀 QuantLLM: Loading {model_name}")
            print("=" * 50)
        
        # Auto-configure everything
        smart_config = SmartConfig.detect(
            model_name,
            bits=bits,
            max_seq_length=max_length,
            device=device,
            dtype=dtype,
        )
        
        # Apply user overrides
        if config_override:
            for key, value in config_override.items():
                if hasattr(smart_config, key):
                    setattr(smart_config, key, value)
        
        if verbose:
            smart_config.print_summary()
        
        # Load tokenizer
        if verbose:
            print("\n📝 Loading tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model with optimizations
        if verbose:
            print(f"📦 Loading model ({smart_config.bits}-bit)...")
        
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": smart_config.dtype,
        }
        
        # Apply quantization if requested
        if quantize and smart_config.bits < 16:
            model_kwargs.update(cls._get_quantization_kwargs(smart_config))
        
        # Device map for memory management
        if smart_config.cpu_offload:
            model_kwargs["device_map"] = "auto"
            model_kwargs["offload_folder"] = "offload"
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = {"": smart_config.device}
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        
        # Apply additional optimizations
        if smart_config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                if verbose:
                    print("✓ Gradient checkpointing enabled")
        
        # Enable Flash Attention if available
        if smart_config.use_flash_attention:
            cls._enable_flash_attention(model, verbose)
        
        # Compile model if beneficial
        if smart_config.compile_model:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                if verbose:
                    print("✓ torch.compile enabled")
            except Exception as e:
                if verbose:
                    print(f"⚠ torch.compile failed: {e}")
        
        if verbose:
            print("\n✅ Model loaded successfully!")
            print("=" * 50)
        
        instance = cls(model, tokenizer, smart_config)
        instance._is_quantized = quantize and smart_config.bits < 16
        
        return instance
    
    @staticmethod
    def _get_quantization_kwargs(config: SmartConfig) -> Dict[str, Any]:
        """Get kwargs for quantized model loading."""
        try:
            from transformers import BitsAndBytesConfig
            
            if config.bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=config.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif config.bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                # For other bit widths, use 4-bit as base
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=config.dtype,
                    bnb_4bit_quant_type="nf4",
                )
            
            return {"quantization_config": quantization_config}
            
        except ImportError:
            print("⚠ bitsandbytes not installed, loading without quantization")
            return {}
    
    @staticmethod
    def _enable_flash_attention(model: PreTrainedModel, verbose: bool = True) -> None:
        """Enable Flash Attention if available."""
        try:
            # Try to use native Flash Attention 2
            if hasattr(model, 'config'):
                model.config._attn_implementation = "flash_attention_2"
                if verbose:
                    print("✓ Flash Attention 2 enabled")
        except Exception:
            if verbose:
                print("⚠ Flash Attention not available")
    
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample (False = greedy)
            stream: Whether to stream output (not yet implemented)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Example:
            >>> response = model.generate("Explain quantum computing in simple terms")
            >>> print(response)
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - max_new_tokens,
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        
        # Decode, removing the prompt
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return response
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Generate response for chat-format messages.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            **kwargs: Additional generation parameters
            
        Returns:
            Assistant's response
            
        Example:
            >>> response = model.chat([
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Hello!"}
            ... ])
        """
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback to simple formatting
            prompt = "\n".join([
                f"{m['role'].upper()}: {m['content']}" for m in messages
            ])
            prompt += "\nASSISTANT:"
        
        return self.generate(prompt, **kwargs)
    
    def finetune(
        self,
        data: Union[str, List[Dict[str, str]], Any],
        *,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Fine-tune the model with LoRA.
        
        Args:
            data: Training data - file path, list of dicts, or HF dataset
            epochs: Number of training epochs (default: auto)
            learning_rate: Learning rate (default: auto based on model size)
            batch_size: Batch size (default: auto based on GPU memory)
            lora_r: LoRA rank (default: auto)
            lora_alpha: LoRA alpha (default: 2x lora_r)
            output_dir: Where to save (default: ./output/{model_name})
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
            
        Example:
            >>> # Simple fine-tuning
            >>> model.finetune("my_data.json", epochs=3)
            >>> 
            >>> # With custom settings
            >>> model.finetune(my_dataset, lora_r=32, learning_rate=1e-4)
        """
        print("🎯 Starting fine-tuning...")
        
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            raise ImportError("peft is required for fine-tuning. Install with: pip install peft")
        
        # Prepare model for training
        if self._is_quantized:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Auto-configure LoRA
        r = lora_r or (16 if self.model.num_parameters() < 10e9 else 64)
        alpha = lora_alpha or (r * 2)
        
        # Determine target modules based on model type
        target_modules = self._get_lora_target_modules()
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self._lora_applied = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✓ LoRA applied: {trainable:,} trainable params ({100*trainable/total:.2f}%)")
        
        # Load and prepare data
        train_dataset = self._prepare_dataset(data)
        
        # Auto-configure training
        epochs = epochs or 3
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or 2e-4
        output_dir = output_dir or f"./output/{self.model.config._name_or_path.split('/')[-1]}"
        
        # Training loop
        try:
            from transformers import TrainingArguments, Trainer
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=learning_rate,
                fp16=self.config.dtype == torch.float16,
                bf16=self.config.dtype == torch.bfloat16,
                logging_steps=10,
                save_strategy="epoch",
                warmup_ratio=0.03,
                lr_scheduler_type="cosine",
                gradient_checkpointing=self.config.gradient_checkpointing,
                **kwargs,
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer,
            )
            
            result = trainer.train()
            self._is_finetuned = True
            
            print(f"\n✅ Training complete! Model saved to {output_dir}")
            
            return {
                "train_loss": result.training_loss,
                "epochs": epochs,
                "output_dir": output_dir,
            }
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def _get_lora_target_modules(self) -> List[str]:
        """Get appropriate LoRA target modules for the model."""
        model_type = getattr(self.model.config, 'model_type', '').lower()
        
        # Common patterns by model type
        target_patterns = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        
        return target_patterns.get(model_type, ["q_proj", "v_proj"])
    
    def _prepare_dataset(self, data: Union[str, List[Dict], Any]) -> Any:
        """Prepare dataset for training."""
        from datasets import Dataset, load_dataset
        
        if isinstance(data, str):
            # Load from file
            if data.endswith('.json') or data.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=data)['train']
            else:
                dataset = load_dataset(data)['train']
        elif isinstance(data, list):
            dataset = Dataset.from_list(data)
        else:
            dataset = data  # Assume it's already a Dataset
        
        # Tokenize
        def tokenize_function(examples):
            # Handle different data formats
            if 'text' in examples:
                texts = examples['text']
            elif 'instruction' in examples and 'output' in examples:
                texts = [
                    f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                    for inst, out in zip(examples['instruction'], examples['output'])
                ]
            else:
                # Try to concatenate all string fields
                keys = [k for k in examples.keys() if isinstance(examples[k][0], str)]
                texts = [' '.join([examples[k][i] for k in keys]) for i in range(len(examples[keys[0]]))]
            
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        return tokenized
    
    def export(
        self,
        format: str,
        output_path: Optional[str] = None,
        *,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Export model to various formats.
        
        Supported formats:
            - "gguf": For llama.cpp, Ollama, LM Studio
            - "safetensors": For HuggingFace ecosystem
            - "onnx": For ONNX Runtime
        
        Args:
            format: Target format (gguf, safetensors, onnx)
            output_path: Output file/directory path
            quantization: For GGUF: Q4_K_M, Q5_K_M, Q2_K, etc.
            **kwargs: Format-specific options
            
        Returns:
            Path to exported model
            
        Example:
            >>> model.export("gguf")  # Uses auto name
            >>> model.export("gguf", "my_model.gguf", quantization="Q4_K_M")
            >>> model.export("safetensors", "./my_model/")
        """
        format = format.lower()
        
        # Merge LoRA if applied
        if self._lora_applied:
            print("🔗 Merging LoRA weights...")
            self.model = self.model.merge_and_unload()
            self._lora_applied = False
        
        # Auto-generate output path
        if output_path is None:
            model_name = self.model.config._name_or_path.split('/')[-1]
            if format == "gguf":
                quant = quantization or self.config.quant_type
                output_path = f"{model_name}-{quant}.gguf"
            elif format == "safetensors":
                output_path = f"./{model_name}-quantllm/"
            elif format == "onnx":
                output_path = f"./{model_name}.onnx"
        
        exporters = {
            "gguf": self._export_gguf,
            "safetensors": self._export_safetensors,
            "onnx": self._export_onnx,
        }
        
        if format not in exporters:
            raise ValueError(f"Unknown format: {format}. Supported: {list(exporters.keys())}")
        
        print(f"📦 Exporting to {format.upper()}...")
        result = exporters[format](output_path, quantization=quantization, **kwargs)
        print(f"✅ Exported to: {result}")
        
        return result
    
    def _export_gguf(
        self, 
        output_path: str, 
        quantization: Optional[str] = None,
        **kwargs
    ) -> str:
        """Export to GGUF format using the modern converter."""
        from ..quant import convert_to_gguf, GGUF_QUANT_TYPES
        
        quant_type = quantization or self.config.quant_type
        
        # Validate quant type
        if quant_type not in GGUF_QUANT_TYPES:
            available = list(GGUF_QUANT_TYPES.keys())
            raise ValueError(f"Unknown quant type: {quant_type}. Available: {available[:10]}...")
        
        # Use the new modern converter
        return convert_to_gguf(
            self.model,
            output_path,
            quant_type=quant_type,
            verbose=True,
            **kwargs
        )
    
    def _export_safetensors(
        self,
        output_path: str,
        **kwargs
    ) -> str:
        """Export to safetensors format."""
        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path, safe_serialization=True)
        self.tokenizer.save_pretrained(output_path)
        return output_path
    
    def _export_onnx(
        self,
        output_path: str,
        **kwargs
    ) -> str:
        """Export to ONNX format."""
        try:
            from transformers.onnx import export
            export(
                self.model,
                self.tokenizer,
                output_path,
            )
            return output_path
        except ImportError:
            raise ImportError("ONNX export requires: pip install onnx onnxruntime")
    
    def __repr__(self) -> str:
        params = self.model.num_parameters() / 1e9
        return (
            f"TurboModel(\n"
            f"  model={self.model.config._name_or_path},\n"
            f"  params={params:.2f}B,\n"
            f"  bits={self.config.bits},\n"
            f"  quantized={self._is_quantized},\n"
            f"  finetuned={self._is_finetuned}\n"
            f")"
        )


def turbo(
    model: str,
    *,
    bits: Optional[int] = None,
    max_length: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    **kwargs,
) -> TurboModel:
    """
    Load and quantize any LLM in one line.
    
    This is the simplest way to use QuantLLM. Everything is
    automatically configured based on your hardware.
    
    Args:
        model: HuggingFace model name or local path
        bits: Override quantization bits (default: auto)
        max_length: Override max sequence length (default: auto)
        device: Override device (default: best GPU)
        dtype: Override dtype (default: bf16/fp16)
        **kwargs: Additional options passed to from_pretrained
        
    Returns:
        TurboModel ready for use
        
    Examples:
        >>> # Simplest usage - everything automatic
        >>> model = turbo("meta-llama/Llama-3-8B")
        >>> 
        >>> # Override quantization
        >>> model = turbo("mistralai/Mistral-7B", bits=4)
        >>> 
        >>> # For long context
        >>> model = turbo("Qwen/Qwen2-72B", max_length=32768)
        >>> 
        >>> # Generate text
        >>> print(model.generate("Hello, world!"))
        >>> 
        >>> # Fine-tune
        >>> model.finetune("my_data.json")
        >>> 
        >>> # Export
        >>> model.export("gguf")
    """
    return TurboModel.from_pretrained(
        model,
        bits=bits,
        max_length=max_length,
        device=device,
        dtype=dtype,
        **kwargs,
    )
