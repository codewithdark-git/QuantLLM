from typing import Optional, Dict, Any, Union, Tuple
import torch
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from ..quant.gguf import GGUFQuantizer, SUPPORTED_GGUF_BITS, SUPPORTED_GGUF_TYPES
from ..utils.logger import logger
import psutil
import math
import os

def get_gpu_memory():
    """Get available GPU memory in GB."""
    if torch.cuda.is_available():
        gpu_mem = []
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            gpu_mem.append(total - allocated)
        return gpu_mem
    return []

def get_system_memory():
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024**3)

def estimate_model_size(model_name: Union[str, PreTrainedModel]) -> float:
    """Estimate model size in GB."""
    try:
        if isinstance(model_name, PreTrainedModel):
            params = sum(p.numel() for p in model_name.parameters())
            return (params * 2) / (1024**3)  # Assuming FP16
        else:
            config = AutoConfig.from_pretrained(model_name)
            if hasattr(config, 'num_parameters'):
                return (config.num_parameters * 2) / (1024**3)  # Assuming FP16
            elif hasattr(config, 'n_params'):
                return (config.n_params * 2) / (1024**3)  # Assuming FP16
            # Estimate based on common architectures
            elif hasattr(config, 'hidden_size') and hasattr(config, 'num_hidden_layers'):
                # More accurate estimation for transformer models
                hidden_size = config.hidden_size
                num_layers = config.num_hidden_layers
                vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else 32000
                
                # Calculate main components
                attention_params = 4 * num_layers * hidden_size * hidden_size  # Q,K,V,O matrices
                ffn_params = 8 * num_layers * hidden_size * hidden_size  # FFN layers
                embedding_params = vocab_size * hidden_size  # Input embeddings
                
                total_params = attention_params + ffn_params + embedding_params
                return (total_params * 2) / (1024**3)  # Assuming FP16
            
            # If no size info available, estimate based on model name
            if "llama" in model_name.lower():
                if "7b" in model_name.lower():
                    return 13.0
                elif "13b" in model_name.lower():
                    return 24.0
                elif "70b" in model_name.lower():
                    return 130.0
                elif "3b" in model_name.lower():
                    return 6.0
            return 7.0  # Default assumption
    except Exception as e:
        logger.log_warning(f"Error estimating model size: {e}. Using default size.")
        return 7.0  # Default assumption

class QuantLLM:
    """High-level API for GGUF model quantization."""
    
    @staticmethod
    def list_quant_types(bits: Optional[int] = None) -> Dict[str, str]:
        """
        List available quantization types and their descriptions.
        
        Args:
            bits: Optional bit width to filter quantization types
            
        Returns:
            Dictionary mapping quantization types to their descriptions
        """
        quant_types = {}
        
        if bits is not None:
            if bits not in SUPPORTED_GGUF_BITS:
                raise ValueError(f"Unsupported bits: {bits}. Supported values: {SUPPORTED_GGUF_BITS}")
            types = SUPPORTED_GGUF_TYPES[bits]
            for qtype, config in types.items():
                quant_types[qtype] = config["description"]
        else:
            for bits_val, types in SUPPORTED_GGUF_TYPES.items():
                for qtype, config in types.items():
                    quant_types[f"{qtype} ({bits_val}-bit)"] = config["description"]
        
        return quant_types
    
    @staticmethod
    def get_recommended_quant_type(
        model_size_gb: float,
        target_size_gb: Optional[float] = None,
        priority: str = "balanced"
    ) -> Tuple[int, str]:
        """
        Get recommended quantization type based on model size and requirements.
        
        Args:
            model_size_gb: Original model size in gigabytes
            target_size_gb: Target model size in gigabytes (optional)
            priority: Optimization priority ("speed", "quality", or "balanced")
            
        Returns:
            Tuple of (bits, quant_type)
        """
        if priority not in ["speed", "quality", "balanced"]:
            raise ValueError("Priority must be 'speed', 'quality', or 'balanced'")
        
        # Calculate compression ratio if target size is specified
        if target_size_gb:
            required_ratio = model_size_gb / target_size_gb
            
            if required_ratio <= 2:
                bits, qtype = (8, "Q8_0") if priority == "quality" else (6, "Q6_K")
            elif required_ratio <= 4:
                if priority == "quality":
                    bits, qtype = (5, "Q5_1")
                elif priority == "speed":
                    bits, qtype = (4, "Q4_K_S")
                else:
                    bits, qtype = (4, "Q4_K_M")
            elif required_ratio <= 8:
                if priority == "quality":
                    bits, qtype = (4, "Q4_1")
                elif priority == "speed":
                    bits, qtype = (3, "Q3_K_S")
                else:
                    bits, qtype = (3, "Q3_K_M")
            else:
                bits, qtype = (2, "Q2_K")
        else:
            # Without target size, recommend based on model size and priority
            if model_size_gb <= 2:
                bits, qtype = (5, "Q5_1") if priority == "quality" else (4, "Q4_K_M")
            elif model_size_gb <= 7:
                bits, qtype = (5, "Q5_1") if priority == "quality" else (4, "Q4_K_M")
            elif model_size_gb <= 13:
                bits, qtype = (4, "Q4_K_M") if priority != "speed" else (4, "Q4_K_S")
            else:
                bits, qtype = (3, "Q3_K_M")
        
        return bits, qtype
    
    @staticmethod
    def quantize_from_pretrained(
        model_name: Union[str, PreTrainedModel],
        bits: int = 4,
        group_size: int = 128,
        quant_type: Optional[str] = None,
        use_packed: bool = True,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: torch.dtype = torch.float16,
        bnb_4bit_use_double_quant: bool = True,
        use_gradient_checkpointing: bool = True,
        device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]] = "auto",
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        offload_folder: Optional[str] = None,
        offload_state_dict: bool = False,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        auto_device: bool = True,
        optimize_for: str = "balanced",
        cpu_offload: bool = False
    ) -> PreTrainedModel:
        """
        Quantize a model using GGUF format with optimized resource handling.
        
        Args:
            model_name: Model identifier or instance
            bits: Number of bits for GGUF quantization
            group_size: Size of quantization groups
            quant_type: GGUF quantization type
            use_packed: Whether to use packed format
            device: Target device for quantization
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
            bnb_4bit_quant_type: BitsAndBytes 4-bit quantization type
            bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
            bnb_4bit_use_double_quant: Whether to use double quantization
            use_gradient_checkpointing: Whether to use gradient checkpointing
            device_map: Device mapping strategy
            max_memory: Maximum memory configuration
            offload_folder: Folder for offloading
            offload_state_dict: Whether to offload state dict
            torch_dtype: Default torch dtype
            auto_device: Automatically determine optimal device
            optimize_for: Optimization priority ("speed", "quality", or "balanced")
            cpu_offload: Whether to use CPU offloading
            
        Returns:
            Quantized model
        """
        try:
            logger.log_info(f"Starting GGUF quantization with {bits} bits")
            
            if bits not in SUPPORTED_GGUF_BITS:
                raise ValueError(f"Unsupported bits: {bits}. Supported values: {SUPPORTED_GGUF_BITS}")
            if quant_type and quant_type not in SUPPORTED_GGUF_TYPES.get(bits, {}):
                raise ValueError(f"Unsupported quant_type: {quant_type} for {bits} bits")
                
            # Estimate model size and available resources
            model_size_gb = estimate_model_size(model_name)
            gpu_mem = get_gpu_memory()
            system_mem = get_system_memory()
            
            logger.log_info(f"Estimated model size: {model_size_gb:.2f} GB")
            logger.log_info(f"Available GPU memory: {gpu_mem}")
            logger.log_info(f"Available system memory: {system_mem:.2f} GB")
            
            # Auto-configure resources
            if auto_device:
                if torch.cuda.is_available() and gpu_mem:
                    max_gpu_mem = max(gpu_mem)
                    if model_size_gb > max_gpu_mem:
                        logger.log_info("Insufficient GPU memory. Using CPU offloading.")
                        device = "cpu"
                        cpu_offload = True
                        device_map = "cpu"
                        max_memory = None
                    else:
                        device = "cuda"
                        # Calculate memory distribution
                        if device_map == "auto":
                            max_memory = {
                                i: f"{int(mem * 0.8)}GB"  # Use 80% of available memory
                                for i, mem in enumerate(gpu_mem)
                            }
                            max_memory["cpu"] = f"{int(system_mem * 0.5)}GB"  # Use 50% of system RAM
                else:
                    device = "cpu"
                    cpu_offload = True
                    device_map = "cpu"
                    max_memory = None
                logger.log_info(f"Auto-selected device: {device}")
                
            # Configure BitsAndBytes for 4-bit quantization
            if load_in_4bit:
                compute_dtype = bnb_4bit_compute_dtype or torch.float16
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    llm_int8_enable_fp32_cpu_offload=cpu_offload
                )
            else:
                bnb_config = None
            
            # If no quant_type specified, use recommended type
            if not quant_type:
                bits, quant_type = QuantLLM.get_recommended_quant_type(
                    model_size_gb=model_size_gb,
                    priority=optimize_for
                )
                logger.log_info(f"Selected quantization type: {quant_type} ({bits}-bit)")
            
            # Create and store quantizer
            quantizer = GGUFQuantizer(
                model_name=model_name,
                bits=bits,
                group_size=group_size,
                quant_type=quant_type,
                use_packed=use_packed,
                device=device,
                quantization_config=bnb_config,
                use_gradient_checkpointing=use_gradient_checkpointing,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                torch_dtype=torch_dtype,
                cpu_offload=cpu_offload
            )
            
            # Store quantizer instance in model for later use
            quantizer.model._quantizer = quantizer
            
            return quantizer.model
            
        except Exception as e:
            logger.log_error(f"Quantization failed: {str(e)}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    @staticmethod
    def save_quantized_model(
        model: PreTrainedModel,
        output_path: str,
        save_format: str = "gguf",
        save_tokenizer: bool = True,
        quant_config: Optional[Dict[str, Any]] = None,
        safe_serialization: bool = True
    ):
        """
        Save a quantized model in either GGUF or safetensors format.
        
        Args:
            model: The quantized model to save
            output_path: Path to save the model
            save_format: Format to save in ("gguf" or "safetensors")
            save_tokenizer: Whether to save the tokenizer
            quant_config: Optional quantization configuration
            safe_serialization: Whether to use safe serialization for safetensors format
        """
        try:
            logger.log_info("\n" + "="*80)
            logger.log_info(f"Starting Model Export Process ({save_format.upper()})".center(80))
            logger.log_info("="*80 + "\n")
            
            # Log model details
            total_params = sum(p.numel() for p in model.parameters())
            model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
            
            logger.log_info("📊 Model Information:")
            logger.log_info("-"*40)
            logger.log_info(f"• Architecture: {model.config.model_type}")
            logger.log_info(f"• Total Parameters: {total_params:,}")
            logger.log_info(f"• Model Size: {model_size_gb:.2f} GB")
            logger.log_info(f"• Export Format: {save_format.upper()}")
            logger.log_info("")
            
            # Get quantization info
            if not quant_config:
                if hasattr(model.config, 'quantization_config'):
                    config_dict = model.config.quantization_config
                    if isinstance(config_dict, BitsAndBytesConfig):
                        # Handle BitsAndBytesConfig
                        bits = 4 if config_dict.load_in_4bit else (8 if config_dict.load_in_8bit else 16)
                        quant_config = {
                            'bits': bits,
                            'group_size': 128,  # Default group size
                            'quant_type': f"Q{bits}_K_M" if bits <= 8 else "F16"
                        }
                        logger.log_info("📊 Quantization Configuration:")
                        logger.log_info("-"*40)
                        logger.log_info(f"• Bits: {bits}")
                        logger.log_info(f"• Quantization Type: {quant_config['quant_type']}")
                        if config_dict.load_in_4bit:
                            logger.log_info(f"• 4-bit Type: {config_dict.bnb_4bit_quant_type}")
                            logger.log_info(f"• Compute dtype: {config_dict.bnb_4bit_compute_dtype}")
                    else:
                        quant_config = config_dict
                else:
                    logger.log_info("\nUsing default 4-bit quantization settings")
                    quant_config = {
                        'bits': 4,
                        'group_size': 128,
                        'quant_type': "Q4_K_M"
                    }
            
            # Create output directory
            output_dir = os.path.dirname(output_path) or "."
            os.makedirs(output_dir, exist_ok=True)
            
            if save_format.lower() == "gguf":
                # Convert to GGUF using the converter
                from ..quant.llama_cpp_utils import LlamaCppConverter
                
                converter = LlamaCppConverter()
                gguf_path = converter.convert_to_gguf(
                    model=model,
                    output_dir=output_dir,
                    bits=quant_config['bits'],
                    group_size=quant_config.get('group_size', 128),
                    save_tokenizer=save_tokenizer
                )
                
                logger.log_info("\n✨ GGUF export completed successfully!")
                
            else:  # safetensors format
                logger.log_info("\n💾 Saving model in safetensors format:")
                logger.log_info("-"*40)
                
                # Save the model
                model.save_pretrained(
                    output_dir,
                    safe_serialization=safe_serialization
                )
                logger.log_info("• Model weights saved successfully")
                
                # Save tokenizer if requested
                if save_tokenizer and hasattr(model, 'tokenizer'):
                    logger.log_info("• Saving tokenizer...")
                    model.tokenizer.save_pretrained(output_dir)
                
                logger.log_info("\n✨ Safetensors export completed successfully!")
            
            logger.log_info("="*80)
            
        except Exception as e:
            logger.log_error(f"\n❌ Failed to save model: {str(e)}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    