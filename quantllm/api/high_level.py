from typing import Optional, Dict, Any, Union, Tuple
import torch
from transformers import PreTrainedModel, AutoTokenizer
from ..quant.gguf import GGUFQuantizer, SUPPORTED_GGUF_BITS, SUPPORTED_GGUF_TYPES
from ..utils.logger import logger
from ..utils.memory_tracker import memory_tracker
from ..utils.benchmark import QuantizationBenchmark

class QuantLLM:
    """High-level API for GGUF model quantization."""
    @staticmethod
    def quantize_from_pretrained(
        model_name_or_path: Union[str, PreTrainedModel],
        bits: int = 4,
        group_size: int = 128,
        quant_type: Optional[str] = None,
        use_packed: bool = True,
        cpu_offload: bool = False,
        desc_act: bool = False,
        desc_ten: bool = False,
        legacy_format: bool = False,
        batch_size: int = 4,
        device: Optional[str] = None,
        calibration_data: Optional[torch.Tensor] = None,
        benchmark: bool = True,
        benchmark_input_shape: Optional[Tuple[int, ...]] = None,
        benchmark_steps: int = 100,
        gradient_checkpointing: bool = False,
        chunk_size: int = 1000,
        auto_device: bool = True
    ) -> Tuple[PreTrainedModel, Any]:
        """
        Quantize a model using GGUF format with optional benchmarking and memory optimizations.
        
        Args:
            model_name_or_path: Model identifier or instance
            bits: Number of bits for quantization
            group_size: Size of quantization groups
            quant_type: GGUF quantization type
            use_packed: Whether to use packed format
            cpu_offload: Whether to offload to CPU during quantization
            desc_act: Whether to use activation descriptors
            desc_ten: Whether to use tensor descriptors
            legacy_format: Whether to use legacy format
            batch_size: Batch size for processing
            device: Target device for quantization
            calibration_data: Data for calibration
            benchmark: Whether to run benchmarks
            benchmark_input_shape: Shape for benchmark inputs
            benchmark_steps: Number of benchmark steps
            gradient_checkpointing: Whether to use gradient checkpointing
            chunk_size: Size of chunks for processing
            auto_device: Automatically determine optimal device
            
        Returns:
            Tuple of (quantized_model, benchmark_results)
        """
        try:
            logger.log_info(f"Starting GGUF quantization with {bits} bits")
            memory_tracker.log_memory("quantization_start")
            
            if bits not in SUPPORTED_GGUF_BITS:
                raise ValueError(f"Unsupported bits: {bits}. Supported values: {SUPPORTED_GGUF_BITS}")
            if quant_type and quant_type not in SUPPORTED_GGUF_TYPES.get(bits, []):
                raise ValueError(f"Unsupported quant_type: {quant_type} for {bits} bits")
                
            # Auto-determine device if requested
            if auto_device and device is None:
                if torch.cuda.is_available():
                    # Check available GPU memory
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory
                    model_size = 0
                    if isinstance(model_name_or_path, PreTrainedModel):
                        model_size = sum(p.numel() * p.element_size() for p in model_name_or_path.parameters())
                    
                    # If model is too large for GPU, use CPU offloading
                    if model_size > gpu_mem * 0.7:  # Leave 30% margin
                        logger.log_info("Model too large for GPU memory. Enabling CPU offloading.")
                        cpu_offload = True
                        device = "cpu"
                    else:
                        device = "cuda"
                else:
                    device = "cpu"
                logger.log_info(f"Auto-selected device: {device}")
            
            quantizer = GGUFQuantizer(
                model_name=model_name_or_path,
                bits=bits,
                group_size=group_size,
                quant_type=quant_type,
                use_packed=use_packed,
                cpu_offload=cpu_offload,
                desc_act=desc_act,
                desc_ten=desc_ten,
                legacy_format=legacy_format,
                batch_size=batch_size,
                device=device,
                gradient_checkpointing=gradient_checkpointing,
                chunk_size=chunk_size
            )
            
            logger.log_info("Starting quantization process")
            quantized_model = quantizer.quantize(calibration_data)
            memory_tracker.log_memory("quantization_complete")
            
            benchmark_results = {}
            if benchmark:
                logger.log_info("Running benchmarks")
                if not benchmark_input_shape:
                    if hasattr(quantized_model.config, 'max_position_embeddings'):
                        seq_len = min(32, quantized_model.config.max_position_embeddings)
                    else:
                        seq_len = 32
                    benchmark_input_shape = (1, seq_len)
                    
                benchmarker = QuantizationBenchmark(
                    model=quantized_model,
                    calibration_data=calibration_data,
                    input_shape=benchmark_input_shape,
                    num_inference_steps=benchmark_steps,
                    device=device,
                    num_warmup_steps=10
                )
                
                benchmark_results = benchmarker.run_all_benchmarks()
                memory_tracker.log_memory("benchmarking_complete")
                
                logger.log_info("Benchmark Results:")
                if hasattr(benchmark_results, 'to_dict'):
                    benchmark_results = benchmark_results.to_dict()
                for metric, value in (benchmark_results.items() if isinstance(benchmark_results, dict) else []):
                    logger.log_info(f"{metric}: {value}")
                    
            return quantized_model, benchmark_results
            
        except Exception as e:
            logger.log_error(f"Quantization failed: {str(e)}")
            raise
        finally:
            memory_tracker.clear_memory()


        
    @staticmethod
    def save_quantized_model(
        model: PreTrainedModel,
        output_path: str,
        save_tokenizer: bool = True
    ):
        """
        Save a quantized model and optionally its tokenizer.
        
        Args:
            model: Quantized model to save
            output_path: Path to save the model
            save_tokenizer: Whether to save the tokenizer
        """
        try:
            logger.log_info(f"Saving quantized model to {output_path}")
            memory_tracker.log_memory("save_start")
            
            # Save model
            model.save_pretrained(output_path)
            
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
            
            memory_tracker.log_memory("save_complete")
            logger.log_info("Model saved successfully")
            
        except Exception as e:
            logger.log_error(f"Failed to save model: {str(e)}")
            raise
        finally:
            memory_tracker.clear_memory()
    
    @staticmethod
    def convert_to_gguf(
        model: PreTrainedModel,
        output_path: str,
        quant_config: Optional[Dict[str, Any]] = None
    ):
        """
        Convert a quantized model to GGUF format.
        
        Args:
            model: Model to convert
            output_path: Path to save GGUF file
            quant_config: Optional quantization configuration
        """
        try:
            logger.log_info(f"Converting model to GGUF format: {output_path}")
            memory_tracker.log_memory("conversion_start")
            
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
            memory_tracker.log_memory("conversion_complete")
            logger.log_info("GGUF conversion completed successfully")
            
        except Exception as e:
            logger.log_error(f"GGUF conversion failed: {str(e)}")
            raise
        finally:
            memory_tracker.clear_memory() 
