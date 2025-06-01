from typing import Optional, Dict, Any, Union, Tuple
import torch
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig
from ..quant.gguf import GGUFQuantizer, SUPPORTED_GGUF_BITS, SUPPORTED_GGUF_TYPES
from ..utils.logger import logger

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
                bits, qtype = (4, "Q4_K_M") if priority != "speed" else (4, "Q4_K_S")
            elif model_size_gb <= 13:
                bits, qtype = (3, "Q3_K_M") if priority != "speed" else (3, "Q3_K_S")
            else:
                bits, qtype = (2, "Q2_K")
        
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
        optimize_for: str = "balanced"
    ) -> PreTrainedModel:
        """
        Quantize a model using GGUF format with BitsAndBytes and Accelerate for efficient loading.
        
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
            
        Returns:
            Quantized model
        """
        try:
            logger.log_info(f"Starting GGUF quantization with {bits} bits")
            
            if bits not in SUPPORTED_GGUF_BITS:
                raise ValueError(f"Unsupported bits: {bits}. Supported values: {SUPPORTED_GGUF_BITS}")
            if quant_type and quant_type not in SUPPORTED_GGUF_TYPES.get(bits, {}):
                raise ValueError(f"Unsupported quant_type: {quant_type} for {bits} bits")
                
            # Auto-determine device if requested
            if auto_device and device is None:
                if torch.cuda.is_available():
                    # Check available GPU memory
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory
                    model_size = 0
                    if isinstance(model_name, PreTrainedModel):
                        model_size = sum(p.numel() * p.element_size() for p in model_name.parameters())
                    
                    # If model is too large for GPU, use CPU offloading
                    if model_size > gpu_mem * 0.7:  # Leave 30% margin
                        logger.log_info("Model too large for GPU memory. Using CPU offloading.")
                        device = "cpu"
                        device_map = "cpu"
                        max_memory = None
                    else:
                        device = "cuda"
                else:
                    device = "cpu"
                    device_map = "cpu"
                    max_memory = None
                logger.log_info(f"Auto-selected device: {device}")
            
            # If no quant_type specified, use recommended type based on optimization priority
            if not quant_type:
                if isinstance(model_name, PreTrainedModel):
                    model_size_gb = sum(p.numel() * p.element_size() for p in model_name.parameters()) / (1024**3)
                else:
                    # Estimate model size based on common architectures
                    config = AutoConfig.from_pretrained(model_name)
                    params = config.n_params if hasattr(config, 'n_params') else None
                    if params:
                        model_size_gb = (params * 2) / (1024**3)  # Assuming FP16
                    else:
                        model_size_gb = 7  # Default assumption
                
                bits, quant_type = QuantLLM.get_recommended_quant_type(
                    model_size_gb=model_size_gb,
                    priority=optimize_for
                )
                logger.log_info(f"Selected quantization type: {quant_type} ({bits}-bit)")
            
            quantizer = GGUFQuantizer(
                model_name=model_name,
                bits=bits,
                group_size=group_size,
                quant_type=quant_type,
                use_packed=use_packed,
                device=device,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                use_gradient_checkpointing=use_gradient_checkpointing,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                torch_dtype=torch_dtype
            )
            
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
        save_tokenizer: bool = True,
        quant_config: Optional[Dict[str, Any]] = None
    ):
        """
        Save a quantized model in GGUF format.
        
        Args:
            model: Quantized model to save
            output_path: Path to save the model
            save_tokenizer: Whether to save the tokenizer
            quant_config: Optional quantization configuration
        """
        try:
            logger.log_info(f"Converting model to GGUF format: {output_path}")
            
            # Get quantization config from model if not provided
            if not quant_config and hasattr(model.config, 'quantization_config'):
                quant_config = model.config.quantization_config
            
            # Create quantizer with existing or default config
            quantizer = GGUFQuantizer(
                model_name=model,
                bits=quant_config.get('bits', 4) if quant_config else 4,
                group_size=quant_config.get('group_size', 128) if quant_config else 128,
                quant_type=quant_config.get('quant_type', None) if quant_config else None,
                use_packed=quant_config.get('use_packed', True) if quant_config else True
            )
            
            # Convert to GGUF
            quantizer.convert_to_gguf(output_path)
            logger.log_info("GGUF conversion completed successfully")
            
            # Save tokenizer if requested
            if save_tokenizer and hasattr(model, 'config'):
                if hasattr(model.config, '_name_or_path'):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            model.config._name_or_path,
                            trust_remote_code=True
                        )
                        tokenizer.save_pretrained(output_path)
                        logger.log_info("Tokenizer saved successfully")
                    except Exception as e:
                        logger.log_warning(f"Failed to save tokenizer: {e}")
            
            logger.log_info("Model saved successfully")
            
        except Exception as e:
            logger.log_error(f"Failed to save model: {str(e)}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    