"""
TurboModel - The Ultra-Simple QuantLLM API.

Load, quantize, fine-tune, and export LLMs with one line each.
"""

import os
import shutil
import tempfile
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
from ..utils import logger, print_header, print_success, print_error, print_info, print_warning, QuantLLMProgress
from transformers.utils.logging import disable_progress_bar as disable_hf_progress_bar
from datasets.utils.logging import disable_progress_bar as disable_ds_progress_bar


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
        verbose: bool = False,
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
        self._is_quantized = False
        self._is_finetuned = False
        self._lora_applied = False
        self.verbose = verbose
    
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
        # Disable default progress bars
        disable_hf_progress_bar()
        disable_ds_progress_bar()
        
        if verbose:
            print_header(f"Loading {model_name}")
        
        # Auto-configure everything
        if verbose:
            logger.info("🚀 Detecting hardware and configuration...")

        smart_config = SmartConfig.detect(
            model_name,
            bits=bits,
            max_seq_length=max_length,
            device=device,
            dtype=dtype,
        )
        
        from dataclasses import asdict
        
        # Apply user overrides
        if config_override:
            # Handle SmartConfig objects
            if isinstance(config_override, SmartConfig):
                override_dict = asdict(config_override)
            else:
                override_dict = config_override
                
            for key, value in override_dict.items():
                if hasattr(smart_config, key):
                    setattr(smart_config, key, value)
        
        if verbose:
            smart_config.print_summary()
        
        # Load tokenizer
        if verbose:
            logger.info("📝 Loading tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model with optimizations
        # Load model with optimizations
        if verbose:
            logger.info(f"📦 Loading model ({smart_config.bits}-bit)...")
        
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": smart_config.dtype,
        }
        
        # Check if model is already quantized to prevent conflicts
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            
            existing_quant = getattr(hf_config, "quantization_config", None)
            if existing_quant:
                allow_requantize = False
                
                # Allow 8-bit -> 4-bit re-quantization (if B&B)
                is_bnb = "BitsAndBytesConfig" in existing_quant.__class__.__name__
                is_8bit = getattr(existing_quant, "load_in_8bit", False)
                
                if is_bnb and is_8bit and smart_config.bits == 4:
                    allow_requantize = True
                if is_bnb and is_8bit and smart_config.bits == 4:
                    allow_requantize = True
                    if verbose:
                        logger.info("  ℹ️ Re-quantizing 8-bit model to 4-bit")
                
                if not allow_requantize:
                    if verbose:
                        logger.warning(f"⚠️ Model is already quantized ({existing_quant.__class__.__name__}). Disabling dynamic quantization.")
                    quantize = False
                
        except Exception:
            pass # Ignore config loading errors, proceed with defaults

        # Apply quantization if requested
        if quantize and smart_config.bits < 16:
            model_kwargs.update(cls._get_quantization_kwargs(smart_config))
        
        # Device map for memory management
        if smart_config.cpu_offload:
            model_kwargs["device_map"] = "auto"
            model_kwargs["offload_folder"] = "offload"
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = {"": smart_config.device}
        
        # Load the model with progress spinner
        with QuantLLMProgress() as p:
            if verbose:
                task = p.add_task("Downloading & Loading model...", total=None)
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
            
            if verbose:
                p.update(task, completed=100)
        
        # Apply additional optimizations
        if smart_config.gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                if verbose:
                    logger.info("  ✓ Gradient checkpointing enabled")
        
        # Enable Flash Attention if available
        if smart_config.use_flash_attention:
            cls._enable_flash_attention(model, verbose)
        
        # Compile model if beneficial
        if smart_config.compile_model:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                if verbose:
                    logger.info("  ✓ torch.compile enabled")
            except Exception as e:
                if verbose:
                    print_warning(f"torch.compile failed: {e}")
        
        if verbose:
            print_success("Model loaded successfully!")
            logger.info("")
        
        instance = cls(model, tokenizer, smart_config)
        instance._is_quantized = quantize and smart_config.bits < 16
        
        return instance
    
    @classmethod
    def from_gguf(
        cls,
        model_id: str,
        filename: Optional[str] = None,
        *,
        device: Optional[str] = None,
        verbose: bool = True,
        **kwargs
    ) -> "TurboModel":
        """
        Load a GGUF model directly from HuggingFace Hub or local path.
        
        This uses transformers' native GGUF support (requires transformers>=4.36.0).
        
        Args:
            model_id: HuggingFace repo ID (e.g., "TheBloke/Llama-2-7B-GGUF") or local directory
            filename: GGUF filename (e.g., "llama-2-7b.Q4_K_M.gguf"). 
                      If None, tries to auto-find. Use list_gguf_files() to see available options.
            device: Override device (default: auto-detect best GPU)
            verbose: Print progress
            **kwargs: Additional args for AutoModelForCausalLM.from_pretrained
            
        Returns:
            TurboModel with loaded GGUF model
            
        Example:
            >>> # List available GGUF files in a repo
            >>> files = TurboModel.list_gguf_files("TheBloke/Llama-2-7B-GGUF")
            >>> print(files)
            >>> 
            >>> # Load specific quantization
            >>> model = TurboModel.from_gguf(
            ...     "TheBloke/Llama-2-7B-GGUF", 
            ...     filename="llama-2-7b.Q4_K_M.gguf"
            ... )
            >>> 
            >>> # Generate text
            >>> model.generate("Hello!")
        """
        if verbose:
            print_header(f"Loading GGUF: {model_id}")
            
        # Check for GGUF package
        try:
            import gguf
            gguf_version = getattr(gguf, '__version__', 'unknown')
            if verbose:
                print_info(f"Using gguf version: {gguf_version}")
        except ImportError:
            print_error("Missing 'gguf' package!")
            raise ImportError(
                "Loading GGUF models requires the 'gguf' package.\n"
                "Please run: pip install gguf>=0.10.0"
            )
        
        # If no filename specified, try to find one
        if filename is None:
            if verbose:
                print_info("No filename specified, searching for GGUF files...")
            try:
                available_files = cls.list_gguf_files(model_id)
                if available_files:
                    # Prefer Q4_K_M if available, otherwise take first
                    q4_files = [f for f in available_files if 'q4_k_m' in f.lower()]
                    filename = q4_files[0] if q4_files else available_files[0]
                    if verbose:
                        print_info(f"Found {len(available_files)} GGUF files, using: {filename}")
            except Exception as e:
                if verbose:
                    print_warning(f"Could not list GGUF files: {e}")
        
        if verbose and filename:
            print_info(f"Loading: {filename}")
            
        smart_config = SmartConfig.detect(model_id, device=device)
        smart_config.quant_type = "GGUF"
        
        with QuantLLMProgress() as progress:
            if verbose:
                task = progress.add_task("Loading GGUF model...", total=None)
                 
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    gguf_file=filename,
                    torch_dtype=smart_config.dtype,
                    trust_remote_code=True,
                    **kwargs
                )
            except ImportError as e:
                if "gguf" in str(e).lower():
                    raise ImportError(
                        "transformers requires a newer version of 'gguf'.\n"
                        "Please run: pip install --upgrade gguf>=0.10.0"
                    ) from e
                raise
            except Exception as e:
                if "gguf" in str(e).lower() and "not found" in str(e).lower():
                    available = cls.list_gguf_files(model_id)
                    raise FileNotFoundError(
                        f"GGUF file '{filename}' not found in {model_id}.\n"
                        f"Available files: {available}"
                    ) from e
                raise
            
            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
            except Exception:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                except Exception:
                    # Some GGUF repos might not have tokenizer, try base model
                    if verbose:
                        print_warning("Could not load tokenizer from GGUF repo, using default")
                    tokenizer = None
                    
            if verbose:
                print_success("GGUF Model loaded successfully!")
                
                # Print model info
                if hasattr(model, 'num_parameters'):
                    params = model.num_parameters() / 1e9
                    print_info(f"Parameters: {params:.2f}B")
             
        instance = cls(model, tokenizer, smart_config, verbose=verbose)
        instance._is_quantized = True
        return instance
    
    @staticmethod
    def list_gguf_files(model_id: str) -> List[str]:
        """
        List available GGUF files in a HuggingFace repository.
        
        Args:
            model_id: HuggingFace repo ID (e.g., "TheBloke/Llama-2-7B-GGUF")
            
        Returns:
            List of GGUF filenames available in the repository
            
        Example:
            >>> files = TurboModel.list_gguf_files("TheBloke/Llama-2-7B-GGUF")
            >>> print(files)
            ['llama-2-7b.Q2_K.gguf', 'llama-2-7b.Q4_K_M.gguf', ...]
        """
        try:
            from huggingface_hub import list_repo_files
            
            all_files = list_repo_files(model_id)
            gguf_files = [f for f in all_files if f.endswith('.gguf')]
            
            # Sort by quantization quality (Q4_K_M before Q2_K, etc.)
            def quant_sort_key(name):
                name_lower = name.lower()
                # Higher number = better quality, listed first
                if 'f32' in name_lower: return 0
                if 'f16' in name_lower: return 1
                if 'q8' in name_lower: return 2
                if 'q6' in name_lower: return 3
                if 'q5_k_m' in name_lower: return 4
                if 'q5_k_s' in name_lower: return 5
                if 'q4_k_m' in name_lower: return 6
                if 'q4_k_s' in name_lower: return 7
                if 'q3_k' in name_lower: return 8
                if 'q2_k' in name_lower: return 9
                return 10
            
            return sorted(gguf_files, key=quant_sort_key)
            
        except Exception as e:
            # If it's a local path, list directory
            if os.path.isdir(model_id):
                return [f for f in os.listdir(model_id) if f.endswith('.gguf')]
            raise ValueError(f"Could not list GGUF files from {model_id}: {e}")

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
                    logger.info("  ✓ Flash Attention 2 enabled")
        except Exception:
            if verbose:
                logger.warning("  ⚠ Flash Attention not available")
    
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
        repetition_penalty: float = 1.1,
        stop_strings: Optional[List[str]] = None,
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
            stream: Whether to stream output token by token
            repetition_penalty: Penalty for repeating tokens (>1.0 = less repetition)
            stop_strings: List of strings that stop generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Example:
            >>> response = model.generate("Explain quantum computing")
            >>> 
            >>> # With streaming
            >>> response = model.generate("Tell me a story", stream=True)
        """
        import sys
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - max_new_tokens,
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Default stop strings
        stop_strings = stop_strings or []
        
        # Streaming generation
        if stream:
            return self._generate_streaming(
                inputs, max_new_tokens, temperature, top_p, top_k,
                do_sample, repetition_penalty, stop_strings, **kwargs
            )
        
        # Non-streaming generation
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
        
        # Decode, removing the prompt
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # Check for stop strings and truncate
        for stop in stop_strings:
            if stop in response:
                response = response.split(stop)[0]
                break
        
        return response.strip()
    
    def _generate_streaming(
        self,
        inputs: Dict,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: float,
        stop_strings: List[str],
        **kwargs,
    ) -> str:
        """Generate with streaming output."""
        import sys
        
        try:
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True,
                skip_special_tokens=True,
            )
            
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature if do_sample else 1.0,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": repetition_penalty,
                "streamer": streamer,
                **kwargs,
            }
            
            # Run generation in background thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream output
            generated_text = []
            for new_text in streamer:
                sys.stdout.write(new_text)
                sys.stdout.flush()
                generated_text.append(new_text)
                
                # Check stop strings
                full_text = "".join(generated_text)
                should_stop = False
                for stop in stop_strings:
                    if stop in full_text:
                        should_stop = True
                        break
                if should_stop:
                    break
            
            thread.join()
            print()  # New line after streaming
            
            response = "".join(generated_text)
            for stop in stop_strings:
                if stop in response:
                    response = response.split(stop)[0]
            
            return response.strip()
            
        except ImportError:
            # Fallback to non-streaming
            print("(Streaming not available, using batch generation)")
            return self.generate(
                self.tokenizer.decode(inputs["input_ids"][0]),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stream=False,
            )
    
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
        # Try to apply chat template
        prompt = None
        
        # First, try native chat template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        
        # If no template, use a sensible default
        if prompt is None:
            # Default chat format (works for most models)
            parts = []
            for m in messages:
                role = m.get('role', 'user')
                content = m.get('content', '')
                if role == 'system':
                    parts.append(f"System: {content}\n")
                elif role == 'user':
                    parts.append(f"User: {content}\n")
                elif role == 'assistant':
                    parts.append(f"Assistant: {content}\n")
            parts.append("Assistant:")
            prompt = "".join(parts)
        
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
        hub_manager: Optional[Any] = None,
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
            hub_manager: QuantLLMHubManager instance for auto-tracking
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        print_header("Starting Fine-tuning")

        # 1. Memory & Environment Optimizations
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if "WANDB_DISABLED" not in os.environ:
            os.environ["WANDB_DISABLED"] = "true"
            
        # Suppress noise
        import warnings
        warnings.filterwarnings("ignore", module="peft")
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        except ImportError:
            raise ImportError("peft is required for fine-tuning. Install with: pip install peft")
        
        # 2. Prepare model for training
        if self._is_quantized:
            self.model = prepare_model_for_kbit_training(self.model)
            
        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # 3. Auto-configure LoRA
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
        
        # Apply LoRA if not already applied
        if not self._lora_applied:
            self.model = get_peft_model(self.model, lora_config)
            self._lora_applied = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print_info(f"LoRA applied: {trainable:,} trainable params ({100*trainable/total:.2f}%)")
        
        # 4. Load and prepare data
        train_dataset = self._prepare_dataset(data)
        
        # 5. Auto-configure training settings
        epochs = epochs or 3
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or 2e-4
        output_dir = output_dir or f"./output/{self.model.config._name_or_path.split('/')[-1]}"
        
        # Auto-track parameters if hub_manager provided
        if hub_manager:
            hub_manager.track_hyperparameters({
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "lora_r": r,
                "lora_alpha": alpha,
                "base_model": getattr(self.config, "model_name", "unknown"),
                "output_dir": output_dir
            })
        
        # Training loop
        try:
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
                optim="paged_adamw_32bit" if self.config.bits <= 4 else "adamw_torch",
                torch_compile=self.config.compile_model,
                **kwargs,
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer, # Use new argument name
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            
            result = trainer.train()
            self._is_finetuned = True
            
            print_success(f"Training complete! Model saved to {output_dir}")
            
            return {
                "train_loss": result.training_loss,
                "epochs": epochs,
                "output_dir": output_dir,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "lora_r": r
            }
            
        except Exception as e:
            print_error(f"Training failed: {e}")
            raise
            # Hint about OOM
            if "out of memory" in str(e).lower():
                print_info("Tip: Try reducing batch_size or enabling gradient_checkpointing in config.")
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
                padding=False, # Use DataCollator for dynamic padding
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False, # Avoid hash warnings
            desc="Tokenizing dataset",
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
            - "gguf": For llama.cpp, Ollama, LM Studio (Q4_K_M, Q5_K_M, etc.)
            - "safetensors": For HuggingFace ecosystem
            - "onnx": For ONNX Runtime, TensorRT
            - "mlx": For Apple Silicon Macs
        
        Args:
            format: Target format (gguf, safetensors, onnx, mlx)
            output_path: Output file/directory path
            quantization: Format-specific quantization:
                - GGUF: Q4_K_M, Q5_K_M, Q8_0, etc.
                - ONNX: dynamic, int8
                - MLX: 4bit, 8bit
            **kwargs: Format-specific options
            
        Returns:
            Path to exported model
            
        Example:
            >>> model.export("gguf")  # Uses auto name
            >>> model.export("gguf", "my_model.gguf", quantization="Q4_K_M")
            >>> model.export("onnx", "./my_model_onnx/")
            >>> model.export("mlx", "./my_model_mlx/", quantization="4bit")
        """
        format = format.lower()
        
        # Merge LoRA if applied
        if self._lora_applied:
            if self.verbose:
                print_info("Merging LoRA weights before export...")
            self.model = self.model.merge_and_unload()
            self._lora_applied = False
        
        # Auto-generate output path
        if output_path is None:
            model_name = self.model.config._name_or_path.split('/')[-1]
            if format == "gguf":
                quant = quantization or self.config.quant_type or "q4_k_m"
                output_path = f"{model_name}.{quant.upper()}.gguf"
            elif format == "safetensors":
                output_path = f"./{model_name}-quantllm/"
            elif format == "onnx":
                output_path = f"./{model_name}-onnx/"
            elif format == "mlx":
                output_path = f"./{model_name}-mlx/"
            else:
                output_path = f"./{model_name}-{format}/"
        
        exporters = {
            "gguf": self._export_gguf,
            "safetensors": self._export_safetensors,
            "onnx": self._export_onnx,
            "mlx": self._export_mlx,
        }
        if format not in exporters:
            raise ValueError(f"Unknown format: {format}. Supported: {list(exporters.keys())}")
        
        print_header(f"Exporting to {format.upper()}")
        result = exporters[format](output_path, quantization=quantization, **kwargs)
        print_success(f"Exported to: {result}")
        
        return result

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        format: str = "safetensors",
        quantization: Optional[str] = None,
        commit_message: str = "Upload model via QuantLLM",
        **kwargs
    ):
        """
        Push model to HuggingFace Hub.
        
        Args:
            repo_id: Repository ID (e.g. "username/model")
            token: HF Token
            format: "safetensors", "gguf", "onnx", or "mlx"
            quantization: Quantization type (for gguf/onnx)
            commit_message: Commit message
            **kwargs: Arguments for export
            
        Supported formats:
            - safetensors: Standard HuggingFace format
            - gguf: For llama.cpp, Ollama, LM Studio
            - onnx: For ONNX Runtime, TensorRT
            - mlx: For Apple Silicon (requires macOS)
        """
        from ..hub import QuantLLMHubManager
        
        format_lower = format.lower()
        model_name = self.model.config._name_or_path.split('/')[-1]
        
        print_header(f"Pushing to {repo_id}")
        print_info(f"Format: {format_lower.upper()}")
        
        manager = QuantLLMHubManager(repo_id=repo_id, hf_token=token)
        
        if format_lower == "gguf":
            # Export GGUF directly to staging
            quant_label = quantization or (self.config.quant_type if self.config.quant_type != "GGUF" else "q4_k_m") or "q4_k_m"
            filename = f"{model_name}.{quant_label.upper()}.gguf"
            save_path = os.path.join(manager.staging_dir, filename)
            
            self.export(format="gguf", output_path=save_path, quantization=quant_label, **kwargs)
            
            manager.track_hyperparameters({
                "format": "gguf",
                "quantization": quant_label.upper(),
                "base_model": model_name
            })
            manager._generate_model_card()
            
        elif format_lower == "onnx":
            # Export to ONNX format
            print_info("Exporting to ONNX format...")
            save_path = manager.staging_dir
            
            self._export_onnx(save_path, quantization=quantization, **kwargs)
            
            manager.track_hyperparameters({
                "format": "onnx",
                "quantization": quantization or "none",
                "base_model": model_name
            })
            manager._generate_model_card()
            
        elif format_lower == "mlx":
            # Export to MLX format
            print_info("Exporting to MLX format...")
            save_path = manager.staging_dir
            
            self._export_mlx(save_path, quantization=quantization, **kwargs)
            
            manager.track_hyperparameters({
                "format": "mlx",
                "quantization": quantization or "none",
                "base_model": model_name
            })
            manager._generate_model_card()
            
        else:
            # SafeTensors or PyTorch format
            manager.save_final_model(self, format=format)
            
        manager.push(commit_message=commit_message)
    
    # Alias for convenience
    push = push_to_hub
    
    def _export_gguf(
        self, 
        output_path: str, 
        quantization: Optional[str] = None,
        fast_mode: bool = False,
        **kwargs
    ) -> str:
        """
        Export to GGUF format using optimized llama.cpp converter.
        
        Automatically installs and configures llama.cpp tools.
        Handles BitsAndBytes quantized models by dequantizing first.
        
        Flow:
            1. Save model to temp directory (dequantize if needed)
            2. Convert to F16 GGUF using convert_hf_to_gguf.py
            3. Quantize to target format (Q4_K_M, Q5_K_M, etc.) using llama-quantize
            
        Args:
            output_path: Output file path for GGUF
            quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)
            fast_mode: Skip intermediate F16 step for faster export (slightly less optimal)
        """
        from ..quant import convert_to_gguf, quantize_gguf, ensure_llama_cpp_installed, GGUF_QUANT_TYPES
        from ..utils import QuantLLMProgress, format_time, format_size
        import time
        
        start_time = time.time()
        
        quant_type = quantization or self.config.quant_type or "q4_k_m"
        quant_type_upper = quant_type.upper()
        quant_type_lower = quant_type.lower()
        
        # Check if this is a passthrough format (f16, bf16, f32 - no quantization needed)
        passthrough_types = {'f16', 'f32', 'bf16', 'float16', 'float32', 'bfloat16'}
        needs_quantization = quant_type_lower not in passthrough_types
        
        if self.verbose:
            print_info(f"Target quantization: {quant_type_upper}")
            if fast_mode:
                print_info("Fast mode enabled")
        
        # Ensure llama.cpp
        if self.verbose:
            print_info("Checking llama.cpp installation...")
        ensure_llama_cpp_installed()
        
        # Check if model is BitsAndBytes quantized and needs dequantization
        model_to_save = self.model
        is_bnb_quantized = self._is_bnb_quantized()
        
        if is_bnb_quantized:
            if self.verbose:
                print_warning("Model is BitsAndBytes quantized. Dequantizing for GGUF export...")
                print_info("This may use significant memory. For large models, consider loading with quantize=False.")
            
            model_to_save = self._dequantize_model()
            if self.verbose:
                print_success("Model dequantized successfully!")
        
        # Determine dtype for initial conversion (always F16 for best quality)
        model_dtype = "f16"
        
        # Get model name for file naming
        model_name = self.model.config._name_or_path.split('/')[-1]
        
        # Create temp dir for conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Save model to temp directory
            if self.verbose:
                print_header("Step 1/3: Saving Model", icon="💾")
                print_info(f"Staging model to {temp_dir}...")
            
            with QuantLLMProgress() as progress:
                task = progress.add_task("Saving model weights...", total=None)
                try:
                    model_to_save.save_pretrained(temp_dir, safe_serialization=True)
                except Exception as e:
                    if self.verbose:
                        print_warning(f"SafeTensors save failed ({e}), using PyTorch format...")
                    model_to_save.save_pretrained(temp_dir, safe_serialization=False)
                
                self.tokenizer.save_pretrained(temp_dir)
                progress.update(task, completed=100)
            
            if self.verbose:
                print_success("Model saved to staging area!")
            
            # Step 2: Convert to F16 GGUF
            if self.verbose:
                print_header("Step 2/3: Converting to GGUF", icon="🔄")
            
            # F16 intermediate file (or final if no quantization needed)
            if needs_quantization:
                f16_gguf_file = os.path.join(temp_dir, f"{model_name}.F16.gguf")
            else:
                f16_gguf_file = f"{model_name}.{quant_type_upper}.gguf"
            
            output_files, _ = convert_to_gguf(
                model_name=model_name,
                input_folder=temp_dir,
                model_dtype=model_dtype,
                quantization_type="f16" if needs_quantization else quant_type_lower,
                print_output=self.verbose
            )
            
            if not output_files:
                raise RuntimeError("GGUF conversion failed to produce output file.")
            
            f16_file = output_files[0]
            
            # If conversion produced a different name, use that
            if os.path.exists(f16_file):
                f16_gguf_file = f16_file
            
            if self.verbose:
                print_success(f"F16 GGUF created: {f16_gguf_file}")
            
            # Step 3: Apply quantization if needed
            if needs_quantization:
                if self.verbose:
                    print_header(f"Step 3/3: Quantizing to {quant_type_upper}", icon="⚡")
                    print_info(f"Applying {quant_type_upper} quantization...")
                
                # Final quantized output
                quantized_file = f"{model_name}.{quant_type_upper}.gguf"
                
                quantize_gguf(
                    input_gguf=f16_gguf_file,
                    output_gguf=quantized_file,
                    quant_type=quant_type_upper,
                    print_output=self.verbose
                )
                
                final_file = quantized_file
                
                # Clean up intermediate F16 file
                if os.path.exists(f16_gguf_file) and f16_gguf_file != quantized_file:
                    os.remove(f16_gguf_file)
                
                if self.verbose:
                    print_success(f"Quantization complete: {quantized_file}")
            else:
                final_file = f16_gguf_file
                if self.verbose:
                    print_info("No quantization needed (already in target format)")
            
            # Move to output path if different
            if os.path.abspath(final_file) != os.path.abspath(output_path):
                if self.verbose:
                    print_info(f"Moving {final_file} → {output_path}")
                shutil.move(final_file, output_path)
        
        # Clean up dequantized model if created
        if is_bnb_quantized and model_to_save is not self.model:
            del model_to_save
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Print final summary
        elapsed = time.time() - start_time
        if self.verbose:
            file_size_bytes = os.path.getsize(output_path)
            print_header("Export Complete! 🎉", icon="✅")
            print_info(f"Output: {output_path}")
            print_info(f"Format: GGUF {quant_type_upper}")
            print_info(f"Size: {format_size(file_size_bytes)}")
            print_info(f"Time: {format_time(elapsed)}")
                
        return output_path
    
    def _is_bnb_quantized(self) -> bool:
        """Check if model is BitsAndBytes quantized."""
        # Check config for quantization_config
        if hasattr(self.model, 'config'):
            quant_config = getattr(self.model.config, 'quantization_config', None)
            if quant_config:
                # Check if it's BitsAndBytes
                quant_method = getattr(quant_config, 'quant_method', None)
                if quant_method in ['bitsandbytes', 'bnb']:
                    return True
                if getattr(quant_config, 'load_in_4bit', False):
                    return True
                if getattr(quant_config, 'load_in_8bit', False):
                    return True
        
        # Check for BNB linear layers in the model
        try:
            import bitsandbytes as bnb
            for module in self.model.modules():
                if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                    return True
        except ImportError:
            pass
        
        return False
    
    def _dequantize_model(self) -> nn.Module:
        """
        Dequantize a BitsAndBytes model to full precision for GGUF export.
        
        Returns:
            Dequantized model in float16/bfloat16
        """
        import gc
        
        # Get the model name for reloading
        model_name = getattr(self.model.config, '_name_or_path', None)
        
        if model_name:
            # Best approach: Reload model in full precision
            if self.verbose:
                print_info(f"Reloading {model_name} in full precision...")
            
            # Determine target dtype
            target_dtype = self.config.dtype if self.config.dtype in [torch.float16, torch.bfloat16] else torch.float16
            
            try:
                dequant_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=target_dtype,
                    trust_remote_code=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                )
                return dequant_model
            except Exception as e:
                if self.verbose:
                    print_warning(f"Failed to reload model: {e}")
                    print_info("Attempting in-place dequantization...")
        
        # Fallback: In-place dequantization (less reliable but works for some models)
        try:
            import bitsandbytes as bnb
            
            target_dtype = self.config.dtype if self.config.dtype in [torch.float16, torch.bfloat16] else torch.float16
            
            # Create a copy of the model state dict with dequantized weights
            dequant_model = AutoModelForCausalLM.from_config(
                self.model.config,
                torch_dtype=target_dtype,
            )
            
            # Copy and dequantize weights
            with torch.no_grad():
                for name, module in self.model.named_modules():
                    if isinstance(module, bnb.nn.Linear4bit):
                        # Dequantize 4-bit weights
                        target_module = dict(dequant_model.named_modules()).get(name)
                        if target_module is not None and hasattr(target_module, 'weight'):
                            # Get dequantized weight
                            weight = module.weight
                            if hasattr(weight, 'dequantize'):
                                dequant_weight = weight.dequantize()
                            else:
                                # Manual dequantization for older versions
                                dequant_weight = bnb.functional.dequantize_4bit(
                                    weight.data, weight.quant_state
                                )
                            target_module.weight.data.copy_(dequant_weight.to(target_dtype))
                            
                            if module.bias is not None:
                                target_module.bias.data.copy_(module.bias.data.to(target_dtype))
                    
                    elif isinstance(module, bnb.nn.Linear8bitLt):
                        # Dequantize 8-bit weights
                        target_module = dict(dequant_model.named_modules()).get(name)
                        if target_module is not None and hasattr(target_module, 'weight'):
                            weight = module.weight
                            if hasattr(weight, 'dequantize'):
                                dequant_weight = weight.dequantize()
                            else:
                                dequant_weight = weight.data.to(target_dtype)
                            target_module.weight.data.copy_(dequant_weight.to(target_dtype))
                            
                            if module.bias is not None:
                                target_module.bias.data.copy_(module.bias.data.to(target_dtype))
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return dequant_model
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to dequantize BitsAndBytes model: {e}\n\n"
                "To export to GGUF, please reload your model without quantization:\n"
                "  model = TurboModel.from_pretrained('your-model', quantize=False)\n"
                "  model.export('gguf', quantization='Q4_K_M')"
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
        quantization: Optional[str] = None,
        opset_version: int = 14,
        **kwargs
    ) -> str:
        """
        Export to ONNX format with proper structure.
        
        Args:
            output_path: Output directory for ONNX files
            quantization: ONNX quantization type (dynamic, static, int8)
            opset_version: ONNX opset version (default: 14)
        """
        from ..utils import QuantLLMProgress
        
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            HAS_OPTIMUM = True
        except ImportError:
            HAS_OPTIMUM = False
        
        os.makedirs(output_path, exist_ok=True)
        model_name = self.model.config._name_or_path
        
        if HAS_OPTIMUM:
            # Use Optimum for best ONNX export
            if self.verbose:
                print_info("Using Optimum for ONNX export (recommended)...")
            
            with QuantLLMProgress() as progress:
                task = progress.add_task("Exporting to ONNX...", total=None)
                
                # Check if model is quantized - need to dequantize first
                if self._is_bnb_quantized():
                    if self.verbose:
                        print_warning("BNB quantized model detected. Exporting from original model...")
                    
                    # Export directly from HuggingFace
                    ort_model = ORTModelForCausalLM.from_pretrained(
                        model_name,
                        export=True,
                        trust_remote_code=True,
                    )
                else:
                    # Save model first, then convert
                    temp_path = os.path.join(output_path, "_temp_hf")
                    os.makedirs(temp_path, exist_ok=True)
                    self.model.save_pretrained(temp_path)
                    self.tokenizer.save_pretrained(temp_path)
                    
                    ort_model = ORTModelForCausalLM.from_pretrained(
                        temp_path,
                        export=True,
                    )
                    
                    # Clean temp
                    shutil.rmtree(temp_path, ignore_errors=True)
                
                # Save ONNX model
                ort_model.save_pretrained(output_path)
                self.tokenizer.save_pretrained(output_path)
                
                progress.update(task, completed=100)
            
            # Apply quantization if requested
            if quantization:
                if self.verbose:
                    print_info(f"Applying {quantization} quantization...")
                self._quantize_onnx_model(output_path, quantization)
                
        else:
            # Fallback to basic torch.onnx export
            if self.verbose:
                print_warning("Optimum not found. Using basic ONNX export.")
                print_info("For better results: pip install optimum[onnxruntime]")
            
            try:
                import onnx
            except ImportError:
                raise ImportError("ONNX export requires: pip install onnx onnxruntime optimum[onnxruntime]")
            
            # Basic export using torch.onnx
            onnx_path = os.path.join(output_path, "model.onnx")
            
            # Create dummy input
            dummy_input = self.tokenizer(
                "Hello world",
                return_tensors="pt",
                padding=True,
            )
            dummy_input = {k: v.to(self.model.device) for k, v in dummy_input.items()}
            
            self.model.eval()
            
            with QuantLLMProgress() as progress:
                task = progress.add_task("Exporting to ONNX...", total=None)
                
                torch.onnx.export(
                    self.model,
                    tuple(dummy_input.values()),
                    onnx_path,
                    input_names=list(dummy_input.keys()),
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch", 1: "sequence"},
                        "attention_mask": {0: "batch", 1: "sequence"},
                        "logits": {0: "batch", 1: "sequence"},
                    },
                    opset_version=opset_version,
                    do_constant_folding=True,
                )
                
                progress.update(task, completed=100)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_path)
        
        if self.verbose:
            print_success(f"ONNX model exported to {output_path}")
        
        return output_path
    
    def _quantize_onnx_model(self, model_path: str, quant_type: str) -> None:
        """Apply ONNX quantization."""
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            
            quantizer = ORTQuantizer.from_pretrained(model_path)
            
            if quant_type.lower() in ["dynamic", "int8"]:
                qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
            else:
                qconfig = AutoQuantizationConfig.avx2(is_static=False)
            
            quantizer.quantize(save_dir=model_path, quantization_config=qconfig)
            
        except ImportError:
            print_warning("Optimum quantization not available. Skipping ONNX quantization.")
    
    def _export_mlx(
        self,
        output_path: str,
        quantization: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export to MLX format for Apple Silicon.
        
        Args:
            output_path: Output directory
            quantization: MLX quantization (4bit, 8bit)
        """
        from ..utils import QuantLLMProgress
        import subprocess
        import sys
        
        # Check platform
        import platform
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            print_warning("MLX export is optimized for Apple Silicon Macs.")
            print_info("The model will be saved but may not run efficiently on this system.")
        
        try:
            import mlx
            HAS_MLX = True
        except ImportError:
            HAS_MLX = False
        
        os.makedirs(output_path, exist_ok=True)
        model_name = self.model.config._name_or_path
        
        if HAS_MLX:
            try:
                from mlx_lm import convert
                
                if self.verbose:
                    print_info("Using mlx-lm for conversion...")
                
                with QuantLLMProgress() as progress:
                    task = progress.add_task("Converting to MLX...", total=None)
                    
                    # Save HF model first if quantized
                    if self._is_bnb_quantized():
                        # Use original model name
                        source_path = model_name
                    else:
                        source_path = os.path.join(output_path, "_temp_hf")
                        os.makedirs(source_path, exist_ok=True)
                        self.model.save_pretrained(source_path)
                        self.tokenizer.save_pretrained(source_path)
                    
                    # Build convert command
                    convert_args = {
                        "hf_path": source_path,
                        "mlx_path": output_path,
                    }
                    
                    if quantization:
                        if "4" in quantization:
                            convert_args["quantize"] = True
                            convert_args["q_bits"] = 4
                        elif "8" in quantization:
                            convert_args["quantize"] = True
                            convert_args["q_bits"] = 8
                    
                    # Run conversion
                    convert(**convert_args)
                    
                    # Clean temp
                    if not self._is_bnb_quantized():
                        shutil.rmtree(source_path, ignore_errors=True)
                    
                    progress.update(task, completed=100)
                    
            except Exception as e:
                print_error(f"MLX conversion failed: {e}")
                raise
        else:
            # Fallback: save as HF format with instructions
            if self.verbose:
                print_warning("mlx-lm not installed. Saving as HuggingFace format.")
                print_info("To convert to MLX: pip install mlx-lm && python -m mlx_lm.convert ...")
            
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            # Create README with conversion instructions
            readme_path = os.path.join(output_path, "CONVERT_TO_MLX.md")
            with open(readme_path, "w") as f:
                f.write("# Convert to MLX\n\n")
                f.write("This model was saved in HuggingFace format.\n")
                f.write("To convert to MLX format on Apple Silicon:\n\n")
                f.write("```bash\n")
                f.write("pip install mlx-lm\n")
                f.write(f"python -m mlx_lm.convert --hf-path {output_path} --mlx-path ./mlx_model\n")
                f.write("```\n")
        
        if self.verbose:
            print_success(f"MLX model exported to {output_path}")
        
        return output_path
    
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


    def optimize_inference(self, backend: str = "triton", bits: int = 4):
        """
        Optimize model for inference using high-performance kernels.
        
        Args:
            backend: Optimization backend ("triton")
            bits: Quantization bits (4 or 8)
        """
        if backend == "triton":
            from ..kernels.triton import TritonQuantizedLinear, is_triton_available
            if not is_triton_available():
                print_warning("Triton is not available or no GPU detected. Skipping optimization.")
                return
            
            if self.verbose:
                print_header("Optimizing with Triton Kernels ⚡")
                
            count = self._replace_with_triton(self.model, bits)
            
            if self.verbose:
                print_success(f"Optimized {count} layers with Triton fused kernels!")
                
    def _replace_with_triton(self, module: nn.Module, bits: int) -> int:
        """Recursively replace Linear layers with TritonQuantizedLinear."""
        from ..kernels.triton import TritonQuantizedLinear
        count = 0
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with Triton Linear
                if self.verbose:
                    print_info(f"Quantizing {name}...")
                
                quantized = TritonQuantizedLinear(
                    child.in_features, 
                    child.out_features, 
                    bits=bits, 
                    bias=child.bias is not None,
                    group_size=128
                )
                quantized.to(child.weight.device)
                quantized.quantize_from(child)
                
                setattr(module, name, quantized)
                count += 1
            else:
                count += self._replace_with_triton(child, bits)
        return count


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
