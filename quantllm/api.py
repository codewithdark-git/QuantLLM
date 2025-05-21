from typing import Optional, Dict, Any, Tuple
from transformers import PreTrainedModel
from .quant.awq import AWQQuantizer
from .quant.gptq import GPTQQuantizer
from .quant.gguf import GGUFQuantizer
from .trainer.logger import TrainingLogger

class QuantizerFactory:
    @staticmethod
    def quantize_from_pretrained(
        model_name_or_path: str,
        method: str,
        quant_config_dict: Optional[Dict[str, Any]] = None,
        calibration_data: Optional[Any] = None, # Typically torch.Tensor or similar
        calibration_steps: Optional[int] = 100, # Specific to AWQ's quantize method
        device: Optional[str] = None # Explicit device control
    ) -> Tuple[PreTrainedModel, Any]: # Returns (quantized_model, tokenizer)
        """
        Loads a model from Hugging Face, quantizes it using the specified method,
        and returns the quantized model and its tokenizer.

        Args:
            model_name_or_path (str): Hugging Face model ID or local path.
            method (str): Quantization method to use ('awq', 'gptq', 'gguf').
            quant_config_dict (Optional[Dict[str, Any]]): Dictionary with quantization parameters.
                Common keys: 'bits', 'group_size', 'batch_size' (for quantizer init).
                AWQ specific: 'zero_point', 'awq_version' (maps to 'version' in AWQQuantizer).
                GPTQ specific: 'actorder', 'percdamp', 'sym'.
                GGUF specific: 'use_packed', 'cpu_offload', 'desc_act', 'desc_ten', 'legacy_format'.
            calibration_data (Optional[Any]): Calibration data required for quantization.
            calibration_steps (Optional[int]): Number of calibration steps, primarily for AWQ's
                                              quantize() method. Defaults to 100.
            device (Optional[str]): Device to run quantization on ('cpu', 'cuda', 'cuda:x'). 
                                    If None, default device selection logic in BaseQuantizer is used.
        
        Returns:
            Tuple[PreTrainedModel, Any]: The quantized model and its associated tokenizer.
        
        Raises:
            ValueError: If an unsupported quantization method is specified or essential parameters are missing.
            RuntimeError: If quantization fails for some reason.
        """
        logger = TrainingLogger() 
        if quant_config_dict is None:
            quant_config_dict = {}

        method_lower = method.lower()
        logger.log_info(f"Attempting to quantize model '{model_name_or_path}' using method: {method_lower}")

        bits = quant_config_dict.get('bits', 4)
        group_size = quant_config_dict.get('group_size', 128)
        quantizer_batch_size = quant_config_dict.get('batch_size', 4) 
        
        quantizer = None

        if method_lower == 'awq':
            awq_zero_point = quant_config_dict.get('zero_point', True)
            awq_version = quant_config_dict.get('awq_version', 'v2')

            quantizer = AWQQuantizer(
                model_or_model_name_or_path=model_name_or_path,
                bits=bits,
                group_size=group_size,
                zero_point=awq_zero_point,
                version=awq_version,
                batch_size=quantizer_batch_size,
                device=device
            )
            logger.log_info(f"Quantizing with AWQ... Bits: {bits}, Group Size: {group_size}, Zero Point: {awq_zero_point}, Version: {awq_version}")
            quantizer.quantize( # Call quantize, model is updated in place
                calibration_data=calibration_data,
                calibration_steps=calibration_steps
            )

        elif method_lower == 'gptq':
            gptq_actorder = quant_config_dict.get('actorder', True)
            gptq_percdamp = quant_config_dict.get('percdamp', 0.01)
            gptq_sym = quant_config_dict.get('sym', True)
            
            quantizer = GPTQQuantizer(
                model_or_model_name_or_path=model_name_or_path,
                bits=bits,
                group_size=group_size,
                actorder=gptq_actorder,
                percdamp=gptq_percdamp,
                sym=gptq_sym,
                batch_size=quantizer_batch_size,
                device=device
            )
            logger.log_info(f"Quantizing with GPTQ... Bits: {bits}, Group Size: {group_size}, ActOrder: {gptq_actorder}, Sym: {gptq_sym}")
            quantizer.quantize(calibration_data=calibration_data) # Model updated in place

        elif method_lower == 'gguf':
            gguf_use_packed = quant_config_dict.get('use_packed', True)
            gguf_cpu_offload = quant_config_dict.get('cpu_offload', False)
            gguf_desc_act = quant_config_dict.get('desc_act', False)
            gguf_desc_ten = quant_config_dict.get('desc_ten', False)
            gguf_legacy_format = quant_config_dict.get('legacy_format', False)

            quantizer = GGUFQuantizer(
                model_or_model_name_or_path=model_name_or_path,
                bits=bits,
                group_size=group_size,
                use_packed=gguf_use_packed,
                cpu_offload=gguf_cpu_offload,
                desc_act=gguf_desc_act,
                desc_ten=gguf_desc_ten,
                legacy_format=gguf_legacy_format,
                batch_size=quantizer_batch_size,
                device=device
            )
            logger.log_info(f"Quantizing with GGUF... Bits: {bits}, Group Size: {group_size}, Packed: {gguf_use_packed}, CPU Offload: {gguf_cpu_offload}")
            quantizer.quantize(calibration_data=calibration_data) # Model updated in place

        else:
            logger.log_error(f"Unsupported quantization method: {method}")
            raise ValueError(f"Unsupported quantization method: {method}. Supported methods are 'awq', 'gptq', 'gguf'.")

        if quantizer is None or quantizer.model is None:
             logger.log_error(f"Failed to initialize quantizer or obtain quantized model for method: {method}")
             raise RuntimeError(f"Quantization failed for method: {method}. Quantizer or model is None.")
        
        logger.log_info(f"Successfully quantized model with method: {method_lower}")
        return quantizer.model, quantizer.tokenizer
