"""
Model Analyzer for QuantLLM.

Analyzes model architecture and size to enable smart configuration
without loading the full model into memory.
"""

import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch


@dataclass
class ModelInfo:
    """Information about a model's architecture and size."""
    model_name: str
    
    # Size info
    num_params: int  # Total parameters
    size_gb: float   # Size in GB at FP32
    
    # Architecture info
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    
    # Model type
    model_type: str  # llama, mistral, qwen, etc.
    
    @property
    def size_4bit_gb(self) -> float:
        """Estimated size at 4-bit quantization."""
        return self.size_gb / 8
    
    @property
    def size_2bit_gb(self) -> float:
        """Estimated size at 2-bit quantization."""
        return self.size_gb / 16
    
    def estimated_size_at_bits(self, bits: int) -> float:
        """Estimate model size at given bit-width."""
        return self.size_gb * bits / 32


class ModelAnalyzer:
    """
    Analyze model architecture without loading weights.
    
    Uses the model config from HuggingFace to determine
    optimal quantization parameters.
    
    Example:
        >>> info = ModelAnalyzer.analyze("meta-llama/Llama-3-8B")
        >>> print(f"Model has {info.num_params / 1e9:.1f}B params")
        >>> print(f"4-bit size: {info.size_4bit_gb:.1f} GB")
    """
    
    # Cache for analyzed models
    _cache: Dict[str, ModelInfo] = {}
    
    # Known model parameter counts (for faster estimation)
    _known_models: Dict[str, int] = {
        "llama-3-8b": 8_000_000_000,
        "llama-3-70b": 70_000_000_000,
        "llama-2-7b": 7_000_000_000,
        "llama-2-13b": 13_000_000_000,
        "llama-2-70b": 70_000_000_000,
        "mistral-7b": 7_000_000_000,
        "mixtral-8x7b": 47_000_000_000,  # Sparse
        "qwen2-7b": 7_000_000_000,
        "qwen2-72b": 72_000_000_000,
        "phi-3-mini": 3_800_000_000,
        "gemma-2b": 2_000_000_000,
        "gemma-7b": 7_000_000_000,
    }
    
    @classmethod
    def analyze(cls, model_name: str, trust_remote_code: bool = True) -> ModelInfo:
        """
        Analyze a model's architecture from its config.
        
        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            
        Returns:
            ModelInfo containing architecture details
        """
        # Check cache
        if model_name in cls._cache:
            return cls._cache[model_name]
        
        try:
            from transformers import AutoConfig
            
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            
            info = cls._extract_info_from_config(model_name, config)
            cls._cache[model_name] = info
            return info
            
        except Exception as e:
            # Fallback to estimation from model name
            return cls._estimate_from_name(model_name)
    
    @staticmethod
    def _get_config_attr(config: Any, keys: list, default: Any) -> Any:
        """Get attribute from config using multiple possible keys."""
        for key in keys:
            if hasattr(config, key):
                return getattr(config, key)
        return default

    @classmethod
    def _extract_info_from_config(cls, model_name: str, config: Any) -> ModelInfo:
        """Extract model info from HuggingFace config."""
        # Get common attributes with defaults (support alternate keys for GPT/Falcon etc)
        hidden_size = cls._get_config_attr(config, ['hidden_size', 'n_embd', 'd_model'], 4096)
        num_layers = cls._get_config_attr(config, ['num_hidden_layers', 'n_layer', 'n_layers'], 32)
        num_attention_heads = cls._get_config_attr(config, ['num_attention_heads', 'n_head', 'n_heads'], 32)
        num_kv_heads = cls._get_config_attr(config, ['num_key_value_heads', 'n_head_kv', 'num_kv_heads'], num_attention_heads)
        intermediate_size = cls._get_config_attr(config, ['intermediate_size', 'n_inner'], hidden_size * 4)
        vocab_size = getattr(config, 'vocab_size', 32000)
        max_position_embeddings = cls._get_config_attr(config, ['max_position_embeddings', 'n_positions', 'seq_length'], 4096)
        model_type = getattr(config, 'model_type', 'unknown')
        
        # Calculate total parameters
        num_params = cls._calculate_params(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size
        )
        
        # Size in GB at FP32 (4 bytes per param)
        size_gb = num_params * 4 / (1024**3)
        
        return ModelInfo(
            model_name=model_name,
            num_params=num_params,
            size_gb=size_gb,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            model_type=model_type
        )
    
    @staticmethod
    def _calculate_params(
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        vocab_size: int
    ) -> int:
        """Calculate total parameters in a transformer model."""
        head_dim = hidden_size // num_attention_heads
        
        # Embedding layer
        embedding_params = vocab_size * hidden_size
        
        # Per layer:
        # - Q projection: hidden_size * hidden_size
        # - K projection: hidden_size * (num_kv_heads * head_dim)
        # - V projection: hidden_size * (num_kv_heads * head_dim)
        # - O projection: hidden_size * hidden_size
        # - MLP up: hidden_size * intermediate_size
        # - MLP gate: hidden_size * intermediate_size (for SwiGLU)
        # - MLP down: intermediate_size * hidden_size
        # - LayerNorms: 2 * hidden_size * 2
        
        kv_size = num_kv_heads * head_dim
        attention_params = (
            hidden_size * hidden_size +  # Q
            hidden_size * kv_size +       # K
            hidden_size * kv_size +       # V
            hidden_size * hidden_size     # O
        )
        
        mlp_params = hidden_size * intermediate_size * 3  # up, gate, down
        norm_params = hidden_size * 4  # 2 norms with weight and bias
        
        layer_params = attention_params + mlp_params + norm_params
        
        # Output projection
        output_params = hidden_size * vocab_size
        
        # Final layer norm
        final_norm_params = hidden_size * 2
        
        total = embedding_params + (layer_params * num_layers) + output_params + final_norm_params
        
        return total
    
    @classmethod
    def _estimate_from_name(cls, model_name: str) -> ModelInfo:
        """Estimate model info from the model name alone."""
        name_lower = model_name.lower()
        
        # Try to match known models
        num_params = None
        for pattern, params in cls._known_models.items():
            if pattern in name_lower:
                num_params = params
                break
        
        # Try to extract size from name (e.g., "7b", "70b")
        if num_params is None:
            match = re.search(r'(\d+)b', name_lower)
            if match:
                num_params = int(match.group(1)) * 1_000_000_000
        
        # Default to 7B if unknown
        if num_params is None:
            num_params = 7_000_000_000
        
        # Estimate architecture from size
        hidden_size, num_layers = cls._estimate_architecture(num_params)
        
        return ModelInfo(
            model_name=model_name,
            num_params=num_params,
            size_gb=num_params * 4 / (1024**3),
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=hidden_size // 128,
            num_kv_heads=hidden_size // 128,
            intermediate_size=hidden_size * 4,
            vocab_size=32000,
            max_position_embeddings=4096,
            model_type="estimated"
        )
    
    @staticmethod
    def _estimate_architecture(num_params: int) -> Tuple[int, int]:
        """Estimate hidden_size and num_layers from param count."""
        # Common configurations
        configs = [
            (1_000_000_000, 2048, 24),    # ~1B
            (3_000_000_000, 3072, 26),    # ~3B
            (7_000_000_000, 4096, 32),    # ~7B
            (8_000_000_000, 4096, 32),    # ~8B
            (13_000_000_000, 5120, 40),   # ~13B
            (34_000_000_000, 6144, 48),   # ~34B
            (70_000_000_000, 8192, 80),   # ~70B
            (180_000_000_000, 12288, 96), # ~180B
        ]
        
        # Find closest match
        for size, hidden, layers in configs:
            if num_params <= size * 1.2:  # Allow 20% margin
                return hidden, layers
        
        # Default for very large models
        return 8192, 80
    
    @classmethod
    def get_quantization_recommendations(cls, model_name: str, available_memory_gb: float) -> Dict[str, Any]:
        """
        Get recommended quantization settings for a model.
        
        Args:
            model_name: Model to analyze
            available_memory_gb: Available GPU memory
            
        Returns:
            Dictionary with recommended settings
        """
        info = cls.analyze(model_name)
        
        recommendations = {
            "model_name": model_name,
            "model_params_b": info.num_params / 1e9,
        }
        
        # Determine viable bit-widths
        viable_bits = []
        for bits in [8, 6, 5, 4, 3, 2]:
            estimated_size = info.estimated_size_at_bits(bits)
            # Need ~1.5x model size for inference headroom
            if estimated_size * 1.5 <= available_memory_gb:
                viable_bits.append(bits)
        
        if not viable_bits:
            recommendations["warning"] = "Model may not fit in available memory even at 2-bit"
            recommendations["recommended_bits"] = 2
            recommendations["cpu_offload"] = True
        else:
            # Prefer 4-bit for best quality/size tradeoff
            if 4 in viable_bits:
                recommendations["recommended_bits"] = 4
            else:
                recommendations["recommended_bits"] = max(viable_bits)
            recommendations["cpu_offload"] = False
        
        recommendations["viable_bits"] = viable_bits
        recommendations["estimated_sizes"] = {
            f"{bits}-bit": f"{info.estimated_size_at_bits(bits):.2f} GB"
            for bits in [2, 4, 8]
        }
        
        return recommendations
