# Implement Complete Transformer Block

**Category:** coding
**Difficulty:** 5
**Tags:** coding, transformers, architecture, end-to-end

## Question
Implement a complete Transformer decoder block from scratch (like GPT).

Your implementation should include:
- Multi-head causal self-attention
- Feed-forward network (MLP)
- Layer normalization
- Residual connections
- Pre-norm architecture (modern standard)

**Function signature:**
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: model dimension
            num_heads: number of attention heads
            d_ff: feed-forward hidden dimension (usually 4 * d_model)
            dropout: dropout rate
        """
        pass

    def forward(self, x, mask=None):
        """
        Args:
            x: input tensor (batch, seq_len, d_model)
            mask: optional causal mask
        Returns:
            output tensor (batch, seq_len, d_model)
        """
        pass
```

## Answer

**Key concepts:**
1. Pre-norm architecture: LayerNorm before attention and FFN
2. Residual connections: x + sublayer(x)
3. Causal masking for autoregressive generation
4. FFN: expand to 4*d_model, activate, project back

**Reference implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Combined QKV projection (more efficient)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # Combined QKV projection
        qkv = self.W_qkv(x)  # (batch, seq_len, 3 * d_model)

        # Split and reshape for multi-head
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        # Reshape and combine heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)

        # Final projection
        output = self.W_o(output)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Expand
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Project back
        x = self.fc2(x)
        x = self.dropout(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Pre-norm architecture
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm attention with residual
        x = x + self.dropout(self.attn(self.ln1(x), mask))

        # Pre-norm FFN with residual
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x

# Full GPT-style model
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=None,
        dropout=0.1
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights (common practice)
        self.head.weight = self.token_emb.weight

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, targets=None):
        """
        Args:
            x: token indices (batch, seq_len)
            targets: target token indices (batch, seq_len) for training
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar if targets provided, else None
        """
        batch_size, seq_len = x.shape

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

        # Token + position embeddings
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.

        Args:
            idx: starting token indices (batch, seq_len)
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
        Returns:
            generated token indices (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.pos_emb.num_embeddings else idx[:, -self.pos_emb.num_embeddings:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
```

**Testing:**
```python
# Test single transformer block
batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
d_ff = 2048

block = TransformerBlock(d_model, num_heads, d_ff)
x = torch.randn(batch_size, seq_len, d_model)

# Create causal mask
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

output = block(x, mask)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Shape preserved: {output.shape == x.shape}")

# Count parameters
num_params = sum(p.numel() for p in block.parameters())
print(f"Parameters in block: {num_params:,}")

# Test full GPT model
vocab_size = 50257  # GPT-2 vocab size
max_seq_len = 1024
model = GPT(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    d_model=768,
    num_heads=12,
    num_layers=12
)

# Forward pass
tokens = torch.randint(0, vocab_size, (2, 100))
logits, _ = model(tokens)

print(f"\nGPT output shape: {logits.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test generation
start_tokens = torch.tensor([[1, 2, 3]])
generated = model.generate(start_tokens, max_new_tokens=10, temperature=0.8, top_k=50)

print(f"\nGenerated shape: {generated.shape}")
print(f"Generated tokens: {generated[0].tolist()}")

# Test training step
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

x = torch.randint(0, vocab_size, (4, 128))
y = torch.randint(0, vocab_size, (4, 128))

logits, loss = model(x, targets=y)
loss.backward()
optimizer.step()

print(f"\nTraining loss: {loss.item():.4f}")
```

**Common mistakes:**
1. ❌ Using post-norm instead of pre-norm (older architecture)
2. ❌ Forgetting dropout in residual connections
3. ❌ Wrong causal mask shape/application
4. ❌ Not tying embeddings and output weights

## Follow-up Questions
- Why pre-norm instead of post-norm?
- How many parameters in GPT-2 small? (Hint: ~117M)
- What's the memory bottleneck during training?
