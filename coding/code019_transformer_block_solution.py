# Implement Complete Transformer Block
# ====================================================================
#
# Implement a complete Transformer decoder block from scratch (like GPT).
# 
# Your implementation should include:
# - Multi-head causal self-attention
# - Feed-forward network (MLP)
# - Layer normalization
# - Residual connections
# - Pre-norm architecture (modern standard)
# 
# **Function signature:**
#
# ====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: model dimension
            num_heads: number of attention heads
            d_ff: feed-forward hidden dimension (usually 4 * d_model)
            dropout: dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)    
        self.dropout = nn.Dropout(dropout)        

    def forward(self, x, mask=None):
        """
        Args:
            x: input tensor (batch, seq_len, d_model)
            mask: optional causal mask
        Returns:
            output tensor (batch, seq_len, d_model)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        normed = self.layer_norm1(x)        
        # these are dim (batch_size, num_heads, seq_len, d_k)
        Q = self.W_Q(normed).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(-3,-2)
        K = self.W_K(normed).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(-3,-2)
        V = self.W_V(normed).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(-3,-2)
        qk = Q @ K.transpose(-2, -1) # dim (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            qk = qk.masked_fill(mask, float('-inf'))
        attn_out = torch.softmax(qk/math.sqrt(self.d_k), dim=-1) @ V # dim (batch_size, num_heads, seq_len, d_k)
        attn_out = attn_out.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)        
        attn_out = self.W_O(attn_out) # dim (batch_size, seq_len, d_model)
        attn_out_with_res = self.dropout(attn_out) + x
        # FFN
        x = self.layer_norm2(attn_out_with_res)
        x = self.ffn1(x)
        x = F.gelu(x)
        x = self.ffn2(x)
        x = self.dropout(x)
        x = attn_out_with_res + x #residual layer
        return x        

