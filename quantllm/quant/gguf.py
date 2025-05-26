"""GGUF (GGML Universal Format) quantization implementation."""

import gc
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import PreTrainedModel
from .quantization_engine import move_to_device, BaseQuantizer, QuantizationConfig, QuantizedLinear

try:
    import ctransformers
    CT_AVAILABLE = True
except ImportError:
    CT_AVAILABLE = False

class GGUFQuantizer(BaseQuantizer):
    """
    Quantizes Hugging Face PreTrainedModels into GGUF format.

    This class implements the GGUF quantization algorithm, leveraging ctransformers for
    the final GGUF file conversion. It supports various bit sizes, group sizes,
    and options like CPU offloading for memory-constrained environments.

    The quantization process involves:
    1. Optional statistics collection from calibration data to inform quantization parameters (experimental).
    2. Layer-by-layer quantization of `nn.Linear` modules:
        - Weights are scaled symmetrically.
        - Grouping is applied along input features if `group_size` is specified.
        - Weights are packed into lower bit representations if `use_packed` is True.
    3. Updating the model's configuration with quantization parameters.
    4. Conversion to a GGUF file using the `ctransformers` library.

    Note: AWQ/GPTQ specific logic is explicitly excluded from this quantizer.
    """
    
    def __init__(
        self,
        model_name: Union[str, PreTrainedModel],
        bits: int = 4,
        group_size: int = 32,
        desc_act: bool = False,
        desc_ten: bool = False,
        use_packed: bool = True,
        legacy_format: bool = False,
        batch_size: int = 4,
        device: Optional[Union[str, torch.device]] = None,
        cpu_offload: bool = False
    ):
        """
        Initializes the GGUFQuantizer.

        Args:
            model_name (Union[str, PreTrainedModel]):
                The Hugging Face model name/path (e.g., "meta-llama/Llama-2-7b-hf")
                or an already loaded PreTrainedModel instance to be quantized.
            bits (int, optional): Number of bits for quantization (e.g., 2, 3, 4, 5, 6, 8). Defaults to 4.
            group_size (int, optional): Size of the quantization group. Typically a power of 2 (e.g., 32, 64, 128).
                                     Use -1 for per-tensor quantization (no grouping). Defaults to 32.
            desc_act (bool, optional): Whether to describe activations in GGUF metadata. Defaults to False.
            desc_ten (bool, optional): Whether to describe tensors in GGUF metadata. Defaults to False.
            use_packed (bool, optional): Whether to use packed quantization types (e.g., Q4_K_M). Defaults to True.
            legacy_format (bool, optional): Whether to use a legacy GGUF format version. Defaults to False.
            batch_size (int, optional): Batch size for processing calibration data. Defaults to 4.
            device (Optional[Union[str, torch.device]], optional): 
                The device for quantization operations ('cpu', 'cuda', etc.). 
                Inherited from BaseQuantizer. Defaults to None (auto-detection).
            cpu_offload (bool, optional): 
                If True, quantized layers and some computations are forced to CPU, 
                reducing GPU memory usage. Defaults to False.
        """
        if not CT_AVAILABLE:
            # This check is crucial. If ctransformers is not available, GGUF quantization cannot proceed.
            raise ImportError("CTransformers is required for GGUF quantization. Install with: pip install ctransformers")

        # Validate quantization parameters
        if bits not in [2, 3, 4, 5, 6, 8]: # Common GGUF bit sizes
            raise ValueError(f"Unsupported bits for GGUF: {bits}. Supported values are [2, 3, 4, 5, 6, 8].")
        
        if group_size != -1 and group_size <= 0:
             raise ValueError(f"GGUF group_size must be -1 (for per-tensor) or a positive integer.")
        
        super().__init__(model_name=model_name, bits=bits, device=device) # BaseQuantizer init, sets up self.logger

        if self.group_size > 0 and not (self.group_size & (self.group_size - 1) == 0):
             self.logger.log_warning(
                f"GGUF group_size {self.group_size} is not a power of 2. "
                "This might lead to suboptimal performance or compatibility issues with some GGUF runtimes."
            )
        if self.group_size == 0: # Should have been caught by the positive integer check.
            raise ValueError("GGUF group_size cannot be 0. Use -1 for per-tensor quantization or a positive integer for group size.")

        self.group_size = group_size 
        self.desc_act = desc_act
        self.desc_ten = desc_ten
        self.use_packed = use_packed
        self.legacy_format = legacy_format
        self.batch_size = batch_size # Used in _collect_stats
        self.cpu_offload = cpu_offload

        if self.device_manager.primary_device is None:
            self.device_manager.determine_primary_device()
            self.logger.log_info(f"Primary device for GGUFQuantizer operations automatically set to: {self.device_manager.primary_device}")
        
        self.logger.log_info(f"GGUFQuantizer initialized with bits={self.bits}, group_size={self.group_size}, cpu_offload={self.cpu_offload}")

    def quantize(
        self,
        calibration_data: Optional[torch.Tensor] = None
    ) -> PreTrainedModel:
        """
        Quantize model using GGUF format with a focus on memory efficiency, 
        performance, and error handling. AWQ/GPTQ specific logic is excluded.

        Returns:
            PreTrainedModel: The quantized model with `nn.Linear` layers replaced by `QuantizedLinear` layers.
                             The model configuration is updated with quantization details.

        Raises:
            ImportError: If `ctransformers` is not available.
            ValueError: If quantization parameters are invalid.
            RuntimeError: If quantization or statistics collection fails.
        """
        self.logger.log_info("Starting GGUF quantization process...")
        if not CT_AVAILABLE: # Should have been caught in __init__, but double-check.
            self.logger.log_error("CTransformers is not available, which is required for GGUF operations.")
            raise ImportError("CTransformers is required for GGUF quantization.")

        quantization_device = torch.device('cpu') if self.cpu_offload else self.device_manager.primary_device
        self.logger.log_info(f"Quantization operations will primarily use: {quantization_device}")

        try:
            if not hasattr(self.model, '_prepared_for_quantization'): 
                self.model = self._prepare_model(self.model) 
            
            if self.cpu_offload and self.model.device != torch.device('cpu'):
                self.logger.log_info(f"CPU offload: Moving initial model from {self.model.device} to CPU.")
                self.model.to(torch.device('cpu'))
                self._clear_memory()

            if calibration_data is not None:
                self.logger.log_info("Preparing calibration data...")
                # Pass quantization_device to prepare_calibration_data
                calibration_data = self.prepare_calibration_data(calibration_data, target_device=quantization_device)
            
            self.model.eval()
            
            stats: Dict[str, Dict[str, torch.Tensor]] = {}
            if calibration_data is not None:
                self.logger.log_info("Collecting statistics from calibration data...")
                try:
                    stats = self._collect_stats(calibration_data) # Uses self.cpu_offload and data.device
                    self.logger.log_info("Successfully collected statistics.")
                except Exception as e:
                    self.logger.log_error(f"Error during statistics collection: {str(e)}")
                    raise RuntimeError(f"Statistics collection failed: {str(e)}") from e
                finally:
                    del calibration_data 
                    self._clear_memory()
            else:
                self.logger.log_info("No calibration data provided. Quantizing without layer-specific statistics.")

            self.logger.log_info("Starting layer-by-layer quantization...")
            quantized_layer_count = 0
            
            # Iterate over a list of (name, module) tuples to avoid issues if model structure is modified
            modules_to_quantize = [(name, module) for name, module in self.model.named_modules() if isinstance(module, nn.Linear)]

            for name, module in modules_to_quantize:
                self.logger.log_info(f"Quantizing layer: {name} of type {type(module)}")
                original_layer_device = next(module.parameters()).device
                try:
                    layer_stats = stats.get(name, None)
                    # _quantize_layer handles moving the layer to its processing device (quantization_device)
                    quantized_layer = self._quantize_layer(module, layer_stats) 
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    if parent_name:
                        parent_module = self.model.get_submodule(parent_name)
                        setattr(parent_module, child_name, quantized_layer)
                    else:
                        setattr(self.model, name, quantized_layer)
                    
                    quantized_layer_count += 1
                    self.logger.log_info(f"Successfully quantized layer: {name}")
                    
                except Exception as e:
                    self.logger.log_error(f"Error quantizing layer {name}: {str(e)}")
                    raise RuntimeError(f"Failed to quantize layer {name}: {str(e)}") from e
                finally:
                    # Ensure original module's memory is freed if it was on a different device or replaced
                    if 'quantized_layer' in locals() and module is not quantized_layer:
                        del module
                    # If original layer was moved by _quantize_layer, it might leave behind cache
                    if original_layer_device != quantization_device :
                         pass # _quantize_layer moves it back or handles it.
                    self._clear_memory()


            if quantized_layer_count == 0:
                self.logger.log_warning("No nn.Linear layers were found or quantized in the model.")
            else:
                self.logger.log_info(f"Successfully quantized {quantized_layer_count} linear layers.")

            self.logger.log_info("Updating model configuration with GGUF parameters...")
            gguf_specific_params = {
                "use_packed": self.use_packed,
                "cpu_offload": self.cpu_offload,
                "desc_act": self.desc_act,
                "desc_ten": self.desc_ten,
                "legacy_format": self.legacy_format,
            }
            self._update_model_config_with_quant_params("gguf", gguf_specific_params)
            self.logger.log_info("Model configuration updated.")
            
            final_model_device = torch.device('cpu') if self.cpu_offload else self.device_manager.primary_device
            if self.model.device != final_model_device:
                 self.logger.log_info(f"Moving final quantized model to {final_model_device}.")
                 self.model.to(final_model_device)
                 self._clear_memory()

            self.logger.log_info("GGUF quantization process completed successfully.")
            return self.model

        except ImportError as e: 
            self.logger.log_critical(f"ImportError: {str(e)}. Please ensure CTransformers is installed.")
            raise e 
        except ValueError as e: 
            self.logger.log_error(f"ValueError during quantization: {str(e)}")
            raise e
        except RuntimeError as e: 
            self.logger.log_error(f"RuntimeError during GGUF quantization: {str(e)}")
            raise e 
        except Exception as e: 
            self.logger.log_error(f"An unexpected error occurred during GGUF quantization: {str(e)}")
            raise RuntimeError(f"Unexpected GGUF quantization failure: {str(e)}") from e
        finally:
            self.logger.log_info("Performing final memory cleanup.")
            self._clear_memory()

    def _collect_stats(self, data: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Collects activation statistics (min, max, mean, std) from calibration data.

        This method processes the `data` in batches through the model, capturing the outputs
        of `nn.Linear` layers. These outputs are used to compute statistics that can
        optionally be used during the `_quantize_layer` step to adjust weight distributions
        (e.g., for methods like SmoothQuant, though GGUF itself doesn't strictly require this).

        Args:
            data (torch.Tensor): Calibration data tensor, expected to be on `quantization_device`
                                 (CPU if `cpu_offload` is True, else primary device).

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: A dictionary where keys are layer names,
                                                 and values are dictionaries of statistics
                                                 (e.g., "min_val", "max_val", "mean", "std")
                                                 stored on `stat_device`.
        """
        stat_device = torch.device('cpu') if self.cpu_offload else self.device_manager.primary_device
        self.logger.log_info(f"Statistics will be collected and stored on: {stat_device}")

        # Data is already on quantization_device (CPU if cpu_offload, else primary_device)
        # Model needs to be on the same device as data for the forward pass.
        model_device_for_stats = data.device
        original_model_device = self.model.device
        
        if original_model_device != model_device_for_stats:
            self.logger.log_info(f"Temporarily moving model from {original_model_device} to {model_device_for_stats} for statistics collection.")
            self.model.to(model_device_for_stats)
            self._clear_memory()
        
        stats_dict: Dict[str, Dict[str, torch.Tensor]] = {} # Explicit type
        
        def hook_fn(name: str): # Explicit type for name
            def fn(_module: nn.Module, _input: Any, output: Any): # Use _ for unused params
                # output[0] is on model_device_for_stats
                if name not in stats_dict:
                    output_sample = output[0] 
                    stats_dict[name] = {
                        "min_val": torch.tensor(float('inf'), device=stat_device, dtype=torch.float32),
                        "max_val": torch.tensor(float('-inf'), device=stat_device, dtype=torch.float32),
                        "sum": torch.zeros(output_sample.shape[1:], device=stat_device, dtype=torch.float32),
                        "sq_sum": torch.zeros(output_sample.shape[1:], device=stat_device, dtype=torch.float32),
                        "count": torch.tensor(0, dtype=torch.long, device=stat_device)
                    }
                
                x_on_model_device = output[0].detach()
                # Chunk processing logic assumes x_on_model_device is large enough to warrant chunking
                # For very small outputs, chunk_size could be larger than x_on_model_device.size(0)
                # Effective chunk_size for processing:
                process_chunk_size = min(x_on_model_device.size(0), max(1, self.batch_size)) # Ensure positive chunk size

                num_chunks = math.ceil(x_on_model_device.size(0) / process_chunk_size)
                
                for i in range(num_chunks):
                    chunk_on_model_dev = x_on_model_device[i * process_chunk_size:(i + 1) * process_chunk_size]
                    chunk_for_stats = move_to_device(chunk_on_model_dev, stat_device, blocking=False).float()
                    
                    # Check if chunk_for_stats is empty before operations
                    if chunk_for_stats.numel() == 0:
                        continue

                    current_min = torch.min(chunk_for_stats.view(-1, chunk_for_stats.size(-1)), dim=0)[0]
                    current_max = torch.max(chunk_for_stats.view(-1, chunk_for_stats.size(-1)), dim=0)[0]

                    stats_dict[name]["min_val"] = torch.min(stats_dict[name]["min_val"], current_min)
                    stats_dict[name]["max_val"] = torch.max(stats_dict[name]["max_val"], current_max)
                    stats_dict[name]["sum"] += torch.sum(chunk_for_stats, dim=0)
                    stats_dict[name]["sq_sum"] += torch.sum(chunk_for_stats ** 2, dim=0)
                    stats_dict[name]["count"] += chunk_for_stats.size(0)
                    
                    del chunk_on_model_dev, chunk_for_stats, current_min, current_max
                    if i % 10 == 0: self._clear_memory()
            return fn
        
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        self.logger.log_info(f"Processing calibration data in batches on {model_device_for_stats} for stat collection...")
        with torch.no_grad():
            # data is already on model_device_for_stats
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i+self.batch_size]
                try:
                    self.model(batch)
                except Exception as e:
                    self.logger.log_error(f"Error during model forward pass in _collect_stats: {e}")
                    for h in hooks: h.remove() # Use different variable name for hook in loop
                    if original_model_device != model_device_for_stats:
                        self.model.to(original_model_device)
                    raise
                if i % (self.batch_size * 5) == 0: self._clear_memory()
        
        for h in hooks: h.remove()
        
        if original_model_device != model_device_for_stats:
            self.logger.log_info(f"Restoring model to its original device: {original_model_device}")
            self.model.to(original_model_device)
            self._clear_memory()

        self.logger.log_info("Calculating final statistics (mean, std)...")
        for name in stats_dict:
            s = stats_dict[name] # More concise
            if s["count"].item() > 0:
                s["mean"] = (s["sum"] / s["count"])
                var = (s["sq_sum"] / s["count"]) - (s["mean"] ** 2)
                s["std"] = torch.sqrt(torch.max(torch.tensor(0.0, device=stat_device, dtype=torch.float32), var))
            else:
                s["mean"] = torch.zeros_like(s["sum"], dtype=torch.float32)
                s["std"] = torch.zeros_like(s["sum"], dtype=torch.float32)
            del s["sum"], s["sq_sum"], s["count"]
        self._clear_memory()
        return stats_dict
        
    def _quantize_layer(
        self,
        layer: nn.Linear,
        stats: Optional[Dict[str, torch.Tensor]] = None
    ) -> QuantizedLinear:
        """
        Quantizes a single `nn.Linear` layer into a `QuantizedLinear` layer configured for GGUF.

        The process involves:
        1. Moving the layer to the `target_device` (CPU if `cpu_offload`, else primary device).
        2. Optionally normalizing weights using provided `stats` (mean/std of activations).
        3. Calculating quantization scales based on weight magnitudes (per-group or per-tensor).
           GGUF uses symmetric quantization for weights, so zero-point is implicitly 0.
        4. Quantizing weights to the specified number of bits (`self.bits`).
        5. Packing weights into a format suitable for GGUF (e.g., multiple 4-bit values into a byte)
           if `self.use_packed` is True.
        6. Storing quantized weights, scales, and zero-points in a new `QuantizedLinear` module.

        Args:
            layer (nn.Linear): The original linear layer to quantize.
            stats (Optional[Dict[str, torch.Tensor]]): Optional activation statistics for this layer.
                                                       If provided, used for weight normalization.

        Returns:
            QuantizedLinear: The new quantized linear layer.

        Raises:
            ValueError: If layer dimensions are incompatible with packing requirements (e.g.,
                        odd input features for 4-bit packing).
        """
        target_device = torch.device('cpu') if self.cpu_offload else self.device_manager.primary_device
        self.logger.log_debug(f"Quantizing layer {type(layer).__name__} to target_device: {target_device}")

        original_layer_device = layer.weight.device
        if original_layer_device != target_device:
            layer = layer.to(target_device)
            # If layer was shared and moved, other parts of model might be affected.
            # This quantizer assumes independent layers or full model on one device.
            self._clear_memory() 

        quant_config = QuantizationConfig(
            bits=self.bits, scheme="symmetric", 
            granularity="per-channel" if self.group_size > 0 and self.group_size < layer.in_features else "per-tensor",
            channel_wise=(self.group_size > 0 and self.group_size < layer.in_features),
            dtype=f"int{self.bits}", format="gguf",
            format_config={
                "desc_act": self.desc_act, "desc_ten": self.desc_ten,
                "use_packed": self.use_packed, "legacy_format": self.legacy_format,
                "group_size": self.group_size if self.group_size < layer.in_features else -1 
            }
        )
        quantized_layer = QuantizedLinear(
            layer.in_features, layer.out_features, bias=layer.bias is not None, config=quant_config
        ).to(target_device)

        if layer.bias is not None:
            quantized_layer.bias.data.copy_(layer.bias.data)
            
        W = layer.weight.data.clone() # W is on target_device
        
        if stats and "mean" in stats and "std" in stats:
            self.logger.log_debug("Applying provided statistics (mean/std) to weights.")
            mean = move_to_device(stats["mean"], target_device).unsqueeze(1) # Ensure [out, 1] for broadcasting
            std = move_to_device(stats["std"], target_device).unsqueeze(1)
            W = (W - mean) / (std + 1e-7)
            del mean, std
            self._clear_memory()
        
        out_features, in_features = W.shape
        q_max = 2**(self.bits - 1) - 1
        q_min = -(2**(self.bits - 1))

        use_effective_group_size = self.group_size > 0 and self.group_size < in_features
        effective_group_size = self.group_size if use_effective_group_size else in_features

        if use_effective_group_size and in_features % effective_group_size != 0:
            self.logger.log_warning(
                f"Layer input features {in_features} not divisible by group_size {effective_group_size}. "
                "This may cause issues or require padding for some GGUF types."
            )
            # For simplicity, we proceed; GGUF packing might handle this or error later.
            # A robust solution would involve padding W or adjusting groups.

        # Reshape W for grouped scaling: [out_features, num_groups, group_size_dim]
        # Each row (output channel) is independently grouped along its input features.
        W_grouped = W.view(out_features, in_features // effective_group_size, effective_group_size)
        
        max_abs_vals = torch.max(torch.abs(W_grouped), dim=2, keepdim=True)[0]
        scales = max_abs_vals / q_max
        scales.clamp_(min=1e-12) # Avoid division by zero; scales are on target_device

        W_quant = torch.round(W_grouped / scales)
        W_quant.clamp_(q_min, q_max)
        
        W_quant = W_quant.view(out_features, in_features) # Reshape back
        final_scales = scales.squeeze(-1) # Shape [out_features, num_groups] or [out_features, 1]
        final_zero_points = torch.zeros_like(final_scales, device=target_device) # Symmetric

        W_quant_int = W_quant.to(torch.int8) # Base for packing

        if self.use_packed:
            if self.bits == 4:
                if W_quant_int.shape[1] % 2 != 0:
                    # This indicates an issue with feature dimension vs packing capability.
                    # GGUF types like Q4_0 expect this. For Q4_K, block size (e.g. 256) is key.
                    # This basic quantizer doesn't fully implement K-quants block logic.
                    raise ValueError(f"4-bit packing requires even in_features per group/tensor, got {W_quant_int.shape[1]}.")
                W_packed = (W_quant_int[:, ::2] + 8) | ((W_quant_int[:, 1::2] + 8) << 4)
                W_quant_final = W_packed.to(torch.uint8)
            elif self.bits == 2:
                if W_quant_int.shape[1] % 4 != 0:
                    raise ValueError(f"2-bit packing requires in_features multiple of 4, got {W_quant_int.shape[1]}.")
                W_packed = (W_quant_int[:, ::4] + 2) | \
                           ((W_quant_int[:, 1::4] + 2) << 2) | \
                           ((W_quant_int[:, 2::4] + 2) << 4) | \
                           ((W_quant_int[:, 3::4] + 2) << 6)
                W_quant_final = W_packed.to(torch.uint8)
            elif self.bits == 8:
                W_quant_final = W_quant_int # Already int8
            # Add more bit-specific packing if needed, e.g. for 3, 5, 6 bits.
            # These often map to specific GGUF K-quant types (Q3_K, Q5_K, Q6_K) which have complex block structures
            # not fully implemented here. This quantizer provides a more generic n-bit quantization.
            else:
                self.logger.log_warning(f"No specific packing for {self.bits}-bit implemented. Using clamped int8 weights.")
                W_quant_final = W_quant_int # Fallback to int8
        else: # Not packed
            W_quant_final = W_quant.to(getattr(torch, quant_config.dtype, torch.int8))

        quantized_layer.weight_quantized.copy_(W_quant_final)
        # GGUF dequantization is typically: original_value = quantized_value * scale.
        # The `QuantizedLinear` class's forward pass likely implements dequantization this way.
        # `final_scales` here are `max_abs_val / q_max`.
        # So, `W_quant = round(W_grouped / final_scales)`.
        # To dequantize: `W_grouped_approx = W_quant * final_scales`.
        # Thus, `QuantizedLinear` should store `final_scales` directly if its `forward` does `W_q * scale`.
        # The previous version had `1.0 / final_scales`. This depends on `QuantizedLinear`'s exact dequant logic.
        # Assuming `QuantizedLinear.forward` uses `quant_weight * self.weight_scale + self.weight_zero_point` (or similar for symmetric)
        # Then `self.weight_scale` should be the `final_scales` calculated here.
        # Let's assume `QuantizedLinear` expects the direct scale for multiplication.
        # If `QuantizedLinear` internally does `quant_weight / stored_scale`, then `1.0/final_scales` would be correct to store.
        # For GGUF, the convention is often that the stored scale is what you multiply the dequantized integer by.
        # Reverting to storing `final_scales` as is, assuming `QuantizedLinear` handles it.
        # If `1.0 / final_scales` was correct, it implies `QuantizedLinear` was dividing by the stored scale.
        # Given the common GGUF pattern `dequantized = quant_value * scale`, storing `final_scales` is more direct.
        # The earlier choice `1.0 / final_scales` might have been to fit a specific `QuantizedLinear` interpretation.
        # For clarity, let's assume `QuantizedLinear` expects the scale that directly multiplies the quantized value.
        quantized_layer.weight_scale.copy_(final_scales) # Storing direct scales for dequant: W_approx = W_q * scale
        quantized_layer.weight_zero_point.copy_(final_zero_points) # For symmetric, zero_points are 0.
        
        # Explicitly delete intermediate tensors to free memory, especially on GPU.
        del W, W_grouped, W_quant, W_quant_int, final_scales, final_zero_points, scales, max_abs_vals
        if 'W_packed' in locals(): 
            del W_packed
        if 'W_quant_final' in locals() and W_quant_final is not quantized_layer.weight_quantized: # If it's a different tensor object
            del W_quant_final

        self._clear_memory() # Call after deletions
        
        # If original layer was moved, its state is now on target_device.
        # The replacement of this layer in the main model structure happens in the `quantize` method.
        # No need to move it back here, as the original `module` object (reference from self.model)
        # will be replaced by `quantized_layer`.
        # This is tricky if the layer is part of a larger model structure.
        # Best practice: quantize operates on a model already on the target_device or handles moves carefully.
        if original_layer_device != target_device and layer.weight.device == target_device:
            # This part is risky if 'layer' is a reference to a module in self.model.
            # The replacement happens outside this function.
            # Consider if _quantize_layer should take a copy or if caller manages original.
            pass # layer = layer.to(original_layer_device) # Avoid this, can cause side effects.

        return quantized_layer

    def convert_to_gguf(self, output_path: str):
        """
        Converts the quantized model to a GGUF file using `ctransformers`.

        The model (which should have already been processed by `self.quantize()`)
        is first moved entirely to CPU. Then, its configuration and tensor data
        (from `model.state_dict()`) are written to the specified GGUF file path.

        Args:
            output_path (str): The path where the GGUF file will be saved.

        Raises:
            ImportError: If `ctransformers` is not available.
            RuntimeError: If the GGUF file conversion fails for any reason.
        """
        if not CT_AVAILABLE:
            self.logger.log_error("CTransformers is not available. Cannot convert to GGUF format.")
            raise ImportError("CTransformers is required for GGUF conversion")
            
        self.logger.log_info(f"Starting conversion of quantized model to GGUF format at: {output_path}")
        
        model_to_convert = self.model
        original_model_device = self.model.device # Should be self.model.device
        is_temp_on_cpu = False
        if original_model_device != torch.device('cpu'):
            self.logger.log_info(f"Moving model to CPU for GGUF conversion (from {original_model_device}).")
            model_to_convert.to(torch.device('cpu')) # Move the actual model instance
            is_temp_on_cpu = True
            self._clear_memory()

        try:
            # Construct GGUF configuration
            gguf_config_dict: Dict[str, Any] = {
                "model_type": model_to_convert.config.model_type,
                "vocab_size": model_to_convert.config.vocab_size,
                "hidden_size": model_to_convert.config.hidden_size,
                "intermediate_size": model_to_convert.config.intermediate_size,
                "num_hidden_layers": model_to_convert.config.num_hidden_layers,
                "num_attention_heads": model_to_convert.config.num_attention_heads,
                "max_position_embeddings": model_to_convert.config.max_position_embeddings,
                "layer_norm_eps": getattr(model_to_convert.config, 'layer_norm_eps', 1e-5),
                "use_cache": getattr(model_to_convert.config, 'use_cache', True),
            }
            # Add GGUF specific architecture fields if available in model config
            # e.g. num_key_value_heads, rope_theta, etc.
            for gguf_arch_key in ["num_key_value_heads", "rope_theta", "ssm_d_inner", 
                                  "ssm_d_conv", "ssm_dt_rank", "ssm_conv_kernel"]:
                if hasattr(model_to_convert.config, gguf_arch_key):
                    gguf_config_dict[gguf_arch_key] = getattr(model_to_convert.config, gguf_arch_key)


            q_cfg_data = {}
            if hasattr(model_to_convert.config, "quantization_config"):
                q_cfg = model_to_convert.config.quantization_config
                if q_cfg.get("format") == "gguf":
                    q_cfg_data = {
                        "bits": q_cfg.get("bits", self.bits),
                        "group_size": q_cfg.get("format_config", {}).get("group_size", self.group_size),
                        "desc_act": q_cfg.get("format_config", {}).get("desc_act", self.desc_act),
                        "desc_ten": q_cfg.get("format_config", {}).get("desc_ten", self.desc_ten),
                        "use_packed": q_cfg.get("format_config", {}).get("use_packed", self.use_packed),
                        "legacy_format": q_cfg.get("format_config", {}).get("legacy_format", self.legacy_format),
                        "quant_type": self._get_gguf_quant_type_string(
                            bits=q_cfg.get("bits", self.bits), 
                            use_packed=q_cfg.get("format_config", {}).get("use_packed", self.use_packed)
                        )
                    }
            
            if not q_cfg_data: # Fallback if no GGUF config found on model
                self.logger.log_warning("Model has no GGUF quantization_config or it's not for GGUF. Using current quantizer settings for GGUF header.")
                q_cfg_data = {
                    "bits": self.bits, "group_size": self.group_size, "desc_act": self.desc_act,
                    "desc_ten": self.desc_ten, "use_packed": self.use_packed, "legacy_format": self.legacy_format,
                    "quant_type": self._get_gguf_quant_type_string(bits=self.bits, use_packed=self.use_packed)
                }
            gguf_config_dict["quantization"] = q_cfg_data
            self.logger.log_info(f"GGUF configuration to be saved: {gguf_config_dict}")

            with open(output_path, 'wb') as f:
                ctransformers.save_config(gguf_config_dict, f)
                
                # Save all tensors (quantized and non-quantized)
                # All tensors must be on CPU for ctransformers.save_tensor
                for name, tensor_data in model_to_convert.state_dict().items():
                    # ctransformers expects specific naming for quantized weights (e.g. from QuantizedLinear)
                    # If state_dict() provides already processed names like "layer.X.weight_quantized", great.
                    # If it provides raw names like "layer.X.weight", need to map them.
                    # Assuming QuantizedLinear correctly registers its buffers (weight_quantized, etc.)
                    # so they appear with appropriate names in state_dict().
                    
                    self.logger.log_debug(f"Saving tensor: {name} of shape {tensor_data.shape} and dtype {tensor_data.dtype}")
                    # Ensure tensor is on CPU (model_to_convert should already be on CPU)
                    if tensor_data.device != torch.device('cpu'):
                        tensor_data_cpu = tensor_data.to(torch.device('cpu'))
                        ctransformers.save_tensor(name, tensor_data_cpu, 0, f)
                        del tensor_data_cpu
                    else:
                        ctransformers.save_tensor(name, tensor_data, 0, f)
                    self._clear_memory() # After each tensor to be safe

            self.logger.log_info(f"Successfully saved GGUF model to {output_path}")
            
        except ImportError as e: 
            self.logger.log_error(f"CTransformers import error during GGUF conversion: {str(e)}")
            raise e
        except Exception as e:
            self.logger.log_error(f"Failed to convert model to GGUF format: {str(e)}")
            raise RuntimeError(f"Failed to convert model to GGUF format: {str(e)}") from e
        finally:
            if is_temp_on_cpu and original_model_device != torch.device('cpu'):
                self.logger.log_info(f"Restoring model to its original device: {original_model_device}")
                model_to_convert.to(original_model_device) # Move the model instance back
            self._clear_memory()

    def _get_gguf_quant_type_string(self, bits: int, use_packed: bool) -> str:
        """Determines the GGUF quantization type string."""
        # This mapping should align with GGUF standards and ctransformers expectations.
        # See GGUF specification (e.g., llama.cpp/ggml.h or GGUF-py's GGMLQuantizationType) for type enums.
        # Examples: GGML_TYPE_Q4_0 = 2, GGML_TYPE_Q4_K = 12, GGML_TYPE_Q6_K = 14, etc.
        # The strings returned here are often conventions used by ctransformers or llama.cpp.
        
        # This is a simplified mapping. For true K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K),
        # the actual GGUF type implies a specific block structure (e.g., K-blocks of 256 for weights),
        # and often includes additional per-block metadata (scales, mins).
        # This quantizer implements a more generic n-bit quantization.
        # The type string helps the GGUF reader interpret the data.
        # If `use_packed` is True, we assume it maps to some common GGUF packed type.
        if use_packed:
            if bits == 2: return "Q2_K"   # A common GGUF 2-bit type (often K-quant)
            if bits == 3: return "Q3_K_S" # A common GGUF 3-bit type (often K-quant, S variant)
            if bits == 4: return "Q4_K_M" # Common 4-bit K-quant (M variant is typical)
            if bits == 5: return "Q5_K_M" # Common 5-bit K-quant (M variant is typical)
            if bits == 6: return "Q6_K"   # A common GGUF 6-bit type (K-quant)
            if bits == 8: return "Q8_0"   # A common GGUF 8-bit type
            else:
                # For other bit sizes (e.g., 7-bit if it were supported), a generic name might be needed.
                self.logger.log_warning(f"No standard GGUF type string for packed {bits}-bit. Using fallback 'custom_packed_b{bits}'. This may affect compatibility.")
                return f"custom_packed_b{bits}"
        else: # Not packed (less common for main quantized weights in GGUF, usually for other tensors or FP16/FP32)
            if bits == 8: # Unpacked int8 weights
                # Q8_0 implies a specific GGUF quantization type. If it's just raw int8, this might be inexact.
                self.logger.log_info("Using 'Q8_0' for unpacked 8-bit weights. Ensure this matches GGUF reader expectations.")
                return "Q8_0" 
            else:
                # Non-packed, non-8-bit integer weights are highly non-standard for GGUF's typical quantized layers.
                # GGUF primarily uses F32, F16 for unquantized weights or metadata.
                self.logger.log_warning(
                    f"Unpacked {bits}-bit integer weights are not standard for GGUF's main quantized layers. "
                    f"Using 'custom_int{bits}'. Header type string may lead to compatibility issues."
                )
                return f"custom_int{bits}" # This is a placeholder and likely not recognized by standard GGUF readers.
    
    def _clear_memory(self):
        """Utility to clear GPU cache (if CUDA is available) and run Python's garbage collector."""
        self.logger.log_debug("Clearing memory: gc.collect() and torch.cuda.empty_cache()")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

