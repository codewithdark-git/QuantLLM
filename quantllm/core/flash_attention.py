"""
Flash Attention Integration for QuantLLM

Provides utilities to enable and configure Flash Attention 2
for maximum attention computation speed.
"""

from typing import Optional, Tuple, Any
import torch
import torch.nn as nn


# Check Flash Attention availability
_FLASH_ATTN_AVAILABLE = False
_FLASH_ATTN_VERSION = None

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_cuda
    _FLASH_ATTN_AVAILABLE = True
    try:
        import flash_attn
        _FLASH_ATTN_VERSION = getattr(flash_attn, '__version__', '2.0.0')
    except:
        _FLASH_ATTN_VERSION = '2.0.0'
except ImportError:
    pass


def is_flash_attention_available() -> bool:
    """Check if Flash Attention is available."""
    return _FLASH_ATTN_AVAILABLE


def get_flash_attention_version() -> Optional[str]:
    """Get the installed Flash Attention version."""
    return _FLASH_ATTN_VERSION


def check_flash_attention_requirements() -> dict:
    """
    Check if system meets Flash Attention requirements.
    
    Returns:
        Dict with requirement status:
        - cuda_available: CUDA is available
        - compute_capability: GPU compute capability (needs >= 8.0 for FA2)
        - flash_attn_installed: flash-attn package is installed
        - meets_requirements: All requirements met
    """
    result = {
        "cuda_available": torch.cuda.is_available(),
        "compute_capability": None,
        "flash_attn_installed": _FLASH_ATTN_AVAILABLE,
        "meets_requirements": False,
    }
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result["compute_capability"] = (props.major, props.minor)
        
        # Flash Attention 2 requires Ampere (SM80) or newer
        if result["compute_capability"] >= (8, 0):
            result["meets_requirements"] = _FLASH_ATTN_AVAILABLE
    
    return result


def enable_flash_attention_for_model(model: nn.Module) -> bool:
    """
    Enable Flash Attention for a HuggingFace model.
    
    This modifies the model's config to use Flash Attention 2
    if available and supported.
    
    Args:
        model: HuggingFace model to enable Flash Attention for
        
    Returns:
        True if Flash Attention was enabled, False otherwise
    """
    if not _FLASH_ATTN_AVAILABLE:
        return False
    
    try:
        # Check if model has config
        if not hasattr(model, 'config'):
            return False
        
        # Try to set attention implementation
        if hasattr(model.config, '_attn_implementation'):
            model.config._attn_implementation = "flash_attention_2"
            return True
        elif hasattr(model.config, 'attn_implementation'):
            model.config.attn_implementation = "flash_attention_2"
            return True
        
        # Alternative: try to set use_flash_attention_2
        if hasattr(model.config, 'use_flash_attention_2'):
            model.config.use_flash_attention_2 = True
            return True
        
        return False
        
    except Exception:
        return False


if _FLASH_ATTN_AVAILABLE:
    def flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = True,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute attention using Flash Attention 2.
        
        This is a direct wrapper around flash_attn_func for easy use.
        
        Args:
            q: Query tensor [batch, seqlen_q, num_heads, head_dim]
            k: Key tensor [batch, seqlen_k, num_heads, head_dim]
            v: Value tensor [batch, seqlen_k, num_heads, head_dim]
            softmax_scale: Scale for softmax (default: 1/sqrt(head_dim))
            causal: Whether to use causal masking
            dropout_p: Dropout probability
            
        Returns:
            Output tensor [batch, seqlen_q, num_heads, head_dim]
        """
        # flash_attn_func expects [batch, seqlen, num_heads, head_dim]
        return flash_attn_func(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=causal,
            dropout_p=dropout_p,
        )
    
    
    def flash_attention_with_kv_cache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flash Attention with KV cache for efficient autoregressive generation.
        
        Args:
            q: Query tensor [batch, 1, num_heads, head_dim]
            k_cache: Key cache [batch, max_seqlen, num_heads, head_dim]
            v_cache: Value cache [batch, max_seqlen, num_heads, head_dim]
            cache_seqlens: Actual sequence lengths [batch]
            
        Returns:
            Tuple of (output, updated_k_cache, updated_v_cache)
        """
        # For token-by-token generation, we need varlen version
        # This is a simplified implementation
        batch, _, num_heads, head_dim = q.shape
        max_seqlen = k_cache.shape[1]
        
        # Get valid parts of cache
        outputs = []
        for b in range(batch):
            seqlen = cache_seqlens[b].item()
            k = k_cache[b, :seqlen].unsqueeze(0)
            v = v_cache[b, :seqlen].unsqueeze(0)
            
            out = flash_attn_func(
                q[b:b+1], k, v,
                causal=False,  # Not causal for single token
            )
            outputs.append(out)
        
        return torch.cat(outputs, dim=0), k_cache, v_cache

else:
    def flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = True,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Fallback attention when Flash Attention is not available."""
        return _pytorch_attention(q, k, v, softmax_scale, causal, dropout_p)
    
    def flash_attention_with_kv_cache(*args, **kwargs):
        raise NotImplementedError("Flash Attention not available. Install with: pip install flash-attn")


def _pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Standard PyTorch attention implementation.
    
    Used as fallback when Flash Attention is not available.
    """
    batch, seqlen_q, num_heads, head_dim = q.shape
    seqlen_k = k.shape[1]
    
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # Reshape for batched matmul: [batch, num_heads, seqlen, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    
    # Apply causal mask
    if causal:
        mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Dropout
    if dropout_p > 0.0 and q.requires_grad:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)
    
    # Output
    output = torch.matmul(attn_weights, v)
    
    # Reshape back: [batch, seqlen, num_heads, head_dim]
    output = output.transpose(1, 2)
    
    return output


class FlashAttentionWrapper(nn.Module):
    """
    Wrapper module that uses Flash Attention when available.
    
    Provides a unified interface that falls back to standard
    PyTorch attention when Flash Attention is not available.
    """
    
    def __init__(
        self,
        softmax_scale: Optional[float] = None,
        causal: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.dropout_p = dropout_p
        self._use_flash = _FLASH_ATTN_AVAILABLE
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention.
        
        Args:
            q: Query [batch, seqlen_q, num_heads, head_dim]
            k: Key [batch, seqlen_k, num_heads, head_dim]
            v: Value [batch, seqlen_k, num_heads, head_dim]
            
        Returns:
            Output [batch, seqlen_q, num_heads, head_dim]
        """
        return flash_attention(
            q, k, v,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
    
    def extra_repr(self) -> str:
        return f"use_flash={self._use_flash}, causal={self.causal}"
