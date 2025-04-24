from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for model loading and initialization."""
    
    # Model identification
    model_name: str
    model_type: str = "auto"
    revision: str = "main"
    trust_remote_code: bool = True
    
    # Model architecture
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    
    # Tokenizer
    tokenizer_name: Optional[str] = None
    tokenizer_revision: str = "main"
    tokenizer_trust_remote_code: bool = False
    
    # Quantization
    quantization_config: Optional[Dict[str, Any]] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA
    lora_config: Optional[Dict[str, Any]] = None
    use_lora: bool = False

    # CPU optimization
    cpu_offload: bool = False
    gradient_checkpointing: bool = False
    bf16: bool = False  # bfloat16 support for more efficient training
    max_memory: Optional[dict] = None  # For device specific memory limits

    kwargs: Optional[Dict[str, Any]] = None
    device_map: Optional[Dict[str, str]] = 'auto'  # 'auto' or specific device mapping
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name
            
        if self.quantization_config is None:
            self.quantization_config = {}
            
        if self.lora_config is None:
            self.lora_config = {}

        if self.kwargs is None:
            self.kwargs = {}
        
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization simultaneously")
        
        # Set reasonable defaults for memory management
        if self.max_memory is None:
            import torch
            if torch.cuda.is_available():
                # Leave some GPU memory free for system
                total_memory = torch.cuda.get_device_properties(0).total_memory
                self.max_memory = {0: f"{int(total_memory * 0.85 / 1024**3)}GiB"}
            else:
                # Default CPU memory limit
                self.max_memory = {"cpu": "16GiB"}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_trust_remote_code": self.tokenizer_trust_remote_code,
            "quantization_config": self.quantization_config,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "lora_config": self.lora_config,
            "use_lora": self.use_lora,
            "cpu_offload": self.cpu_offload,
            "gradient_checkpointing": self.gradient_checkpointing,
            "bf16": self.bf16,
            "max_memory": self.max_memory,
            "kwargs": self.kwargs,
            "device_map": self.device_map
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
        
    def validate(self) -> bool:
        """Validate configuration values."""
        if not self.model_name:
            raise ValueError("Model name or path is required")
            
        if self.hidden_size is not None and self.hidden_size <= 0:
            raise ValueError("Hidden size must be positive")
        if self.num_hidden_layers is not None and self.num_hidden_layers <= 0:
            raise ValueError("Number of hidden layers must be positive")
        if self.num_attention_heads is not None and self.num_attention_heads <= 0:
            raise ValueError("Number of attention heads must be positive")
        if self.intermediate_size is not None and self.intermediate_size <= 0:
            raise ValueError("Intermediate size must be positive")
        if self.hidden_dropout_prob < 0 or self.hidden_dropout_prob >= 1:
            raise ValueError("Hidden dropout probability must be between 0 and 1")
        if self.attention_probs_dropout_prob < 0 or self.attention_probs_dropout_prob >= 1:
            raise ValueError("Attention dropout probability must be between 0 and 1")
        if self.max_position_embeddings <= 0:
            raise ValueError("Max position embeddings must be positive")
        if self.type_vocab_size <= 0:
            raise ValueError("Type vocab size must be positive")
        if self.initializer_range <= 0:
            raise ValueError("Initializer range must be positive")
        if self.layer_norm_eps <= 0:
            raise ValueError("Layer norm epsilon must be positive")
            
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Cannot load model in both 8-bit and 4-bit precision")
            
        if self.device_map not in ['auto', None] and not isinstance(self.device_map, dict):
            raise ValueError("device_map must be 'auto', None, or a dictionary mapping layers to devices")
            
        return True