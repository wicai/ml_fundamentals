# Implement KV Cache for Autoregressive Generation
# ====================================================================
#
# Implement KV caching for efficient autoregressive text generation.
# 
# Your implementation should:
# - Cache key and value tensors from previous tokens
# - Reuse cached KV for new token generation
# - Support batched generation
# - Avoid recomputing attention for past tokens
# 
# **Function signature:**
#
# ====================================================================

from typing import Optional, Tuple
import torch
import torch.nn as nn

class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Args:
            d_model: model dimension
            num_heads: number of attention heads
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads          
        self.wq = nn.Linear(d_model, d_model)        
        self.wk = nn.Linear(d_model, d_model)        
        self.wv = nn.Linear(d_model, d_model)        
        self.wo = nn.Linear(d_model, d_model)        

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: input tokens, shape (batch, seq_len, d_model)
            kv_cache: optional tuple of (cached_keys, cached_values)
                      each of shape (batch, num_heads, prev_seq_len, d_k)
            use_cache: whether to return updated cache
        Returns:
            output: shape (batch, seq_len, d_model)
            new_kv_cache: tuple of (keys, values) if use_cache else None
        """
        Q = self.wq(x) # (batch, seq_len, d_model)
        
        batch = Q.shape[0]
        seq_len = Q.shape[1]
        d_k = self.d_model // self.num_heads
        Q = Q.reshape(batch, seq_len, self.num_heads, d_k).transpose(1, 2) # (batch, num_heads, seq_len, d_k)
        if use_cache and kv_cache is not None: # would be none on first time 
            k_cached, v_cached = kv_cache
            k_new = self.wk(x[:,-1,:]) #(batch, 1, d_model)
            k_new = k_new.reshape((batch, self.num_heads, d_k)).unsqueeze(-2) # (batch, num_heads, 1, d_k)
            K = torch.stack([k_cached, k_new], dim=2) # (batch, num_heads, seq_len, d_k)
            v_new = self.wv(x[:,-1,:]) #(batch, 1, d_model)
            v_new = v_new.reshape((batch, self.num_heads, d_k)).unsqueeze(-2) # (batch, num_heads, 1, d_k)
            V = torch.stack([v_cached, v_new], dim=2)
        else:
            K = self.wk(x)
            K = K.reshape(batch, seq_len, self.num_heads, d_k).transpose(1, 2) # (batch, num_heads, seq_len, d_k)
            V = self.wv(x)
            V = V.reshape(batch, seq_len, self.num_heads, d_k).transpose(1, 2) # (batch, num_heads, seq_len, d_k)
        # softmax(QK^T/sqrt(d_k))V 
        x = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(d_k), dim=-1) @ V # (batch, num_heads, seq_len, d_k)
        x = x.transpose(1, 2).reshape((batch, seq_len, self.d_model)) # (batch, seq_len, d_model)
        x = self.wo(x)
        return (x, (K,V) if use_cache else None)        

