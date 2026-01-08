# Master Einsum for Tensor Operations

**Category:** coding
**Difficulty:** 3
**Tags:** coding, tensor-operations, efficiency

## Question
Implement common tensor operations using `torch.einsum`.

Einsum (Einstein summation) provides a concise way to express tensor operations. Implement the following using einsum:
- Matrix multiplication
- Batch matrix multiplication
- Attention computation
- Tensor transpose
- Outer product

**Function signatures:**
```python
def matmul_einsum(A, B):
    """Matrix multiplication using einsum."""
    pass

def batch_matmul_einsum(A, B):
    """Batch matrix multiplication using einsum."""
    pass

def attention_einsum(Q, K, V):
    """Compute attention using einsum."""
    pass

def transpose_einsum(x):
    """Transpose last two dimensions using einsum."""
    pass

def outer_product_einsum(a, b):
    """Outer product using einsum."""
    pass
```

## Answer

**Key concepts:**
1. Einsum notation: repeated indices are summed over
2. Free indices appear in output
3. More readable and often faster than manual operations

**Reference implementations:**
```python
import torch

def matmul_einsum(A, B):
    """
    Matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]

    Args:
        A: (m, k)
        B: (k, n)
    Returns:
        C: (m, n)
    """
    return torch.einsum('ik,kj->ij', A, B)

def batch_matmul_einsum(A, B):
    """
    Batch matrix multiplication: C[b,i,j] = sum_k A[b,i,k] * B[b,k,j]

    Args:
        A: (batch, m, k)
        B: (batch, k, n)
    Returns:
        C: (batch, m, n)
    """
    return torch.einsum('bik,bkj->bij', A, B)

def attention_einsum(Q, K, V):
    """
    Compute attention: softmax(Q @ K^T / sqrt(d)) @ V

    Args:
        Q: (batch, num_heads, seq_len, d_k)
        K: (batch, num_heads, seq_len, d_k)
        V: (batch, num_heads, seq_len, d_v)
    Returns:
        output: (batch, num_heads, seq_len, d_v)
    """
    d_k = Q.size(-1)

    # Compute attention scores: Q @ K^T
    # 'bhqd,bhkd->bhqk'
    # b=batch, h=heads, q=query_seq, k=key_seq, d=dimension
    scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))

    # Softmax over key dimension
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply attention to values: attn @ V
    # 'bhqk,bhkd->bhqd'
    output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, V)

    return output

def transpose_einsum(x):
    """
    Transpose last two dimensions.

    Args:
        x: (..., m, n)
    Returns:
        x_T: (..., n, m)
    """
    # For 2D: 'ij->ji'
    # For 3D: 'bij->bji'
    # For 4D: 'bhij->bhji'
    # General: just swap the last two indices

    # Example for 4D
    if x.dim() == 4:
        return torch.einsum('bhij->bhji', x)
    elif x.dim() == 3:
        return torch.einsum('bij->bji', x)
    elif x.dim() == 2:
        return torch.einsum('ij->ji', x)
    else:
        raise ValueError("Unsupported dimension")

def outer_product_einsum(a, b):
    """
    Outer product: C[i,j] = a[i] * b[j]

    Args:
        a: (m,)
        b: (n,)
    Returns:
        C: (m, n)
    """
    return torch.einsum('i,j->ij', a, b)

# More complex examples

def batch_diagonal_einsum(x):
    """
    Extract diagonal from batched matrices.

    Args:
        x: (batch, n, n)
    Returns:
        diag: (batch, n)
    """
    return torch.einsum('bii->bi', x)

def trace_einsum(x):
    """
    Compute trace of matrix.

    Args:
        x: (n, n)
    Returns:
        trace: scalar
    """
    return torch.einsum('ii->', x)

def bilinear_einsum(x, W, y):
    """
    Bilinear product: x^T W y

    Args:
        x: (batch, m)
        W: (m, n)
        y: (batch, n)
    Returns:
        result: (batch,)
    """
    return torch.einsum('bi,ij,bj->b', x, W, y)

def multi_head_reshape(x, num_heads):
    """
    Reshape for multi-head attention.

    Args:
        x: (batch, seq_len, d_model)
    Returns:
        x: (batch, num_heads, seq_len, d_k)
    """
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // num_heads

    x = x.view(batch_size, seq_len, num_heads, d_k)
    # Transpose: 'bshd->bhsd'
    return torch.einsum('bshd->bhsd', x)

def tensor_contraction(A, B, C):
    """
    Complex tensor contraction.

    Args:
        A: (i, j, k)
        B: (k, l, m)
        C: (m, n)
    Returns:
        result: (i, j, l, n)
    """
    # Sum over k and m
    return torch.einsum('ijk,klm,mn->ijln', A, B, C)
```

**Testing:**
```python
# Test matmul
A = torch.randn(3, 4)
B = torch.randn(4, 5)

result_einsum = matmul_einsum(A, B)
result_torch = torch.matmul(A, B)

print(f"Matmul match: {torch.allclose(result_einsum, result_torch)}")

# Test batch matmul
A_batch = torch.randn(2, 3, 4)
B_batch = torch.randn(2, 4, 5)

result_batch = batch_matmul_einsum(A_batch, B_batch)
expected_batch = torch.bmm(A_batch, B_batch)

print(f"Batch matmul match: {torch.allclose(result_batch, expected_batch)}")

# Test attention
batch, num_heads, seq_len, d_k = 2, 4, 10, 64
Q = torch.randn(batch, num_heads, seq_len, d_k)
K = torch.randn(batch, num_heads, seq_len, d_k)
V = torch.randn(batch, num_heads, seq_len, d_k)

output = attention_einsum(Q, K, V)
print(f"Attention output shape: {output.shape}")

# Test transpose
x = torch.randn(2, 3, 4, 5)
x_T = transpose_einsum(x)
expected_T = x.transpose(-2, -1)

print(f"Transpose match: {torch.allclose(x_T, expected_T)}")

# Test outer product
a = torch.randn(5)
b = torch.randn(7)

outer = outer_product_einsum(a, b)
expected_outer = torch.outer(a, b)

print(f"Outer product match: {torch.allclose(outer, expected_outer)}")

# Benchmark: einsum vs manual
import time

A_large = torch.randn(1000, 1000).cuda()
B_large = torch.randn(1000, 1000).cuda()

# Einsum
start = time.time()
for _ in range(100):
    _ = torch.einsum('ik,kj->ij', A_large, B_large)
torch.cuda.synchronize()
time_einsum = time.time() - start

# Manual matmul
start = time.time()
for _ in range(100):
    _ = torch.matmul(A_large, B_large)
torch.cuda.synchronize()
time_matmul = time.time() - start

print(f"\nEinsum: {time_einsum:.4f}s")
print(f"Matmul: {time_matmul:.4f}s")
print(f"Ratio: {time_einsum/time_matmul:.2f}x")
```

**Common einsum patterns:**
```python
# Common patterns cheat sheet

# Element-wise multiplication
# 'ij,ij->ij'
torch.einsum('ij,ij->ij', A, B) == A * B

# Sum all elements
# 'ij->'
torch.einsum('ij->', A) == A.sum()

# Sum along rows
# 'ij->i'
torch.einsum('ij->i', A) == A.sum(dim=1)

# Sum along columns
# 'ij->j'
torch.einsum('ij->j', A) == A.sum(dim=0)

# Batch dot product
# 'bi,bi->b'
torch.einsum('bi,bi->b', x, y) == (x * y).sum(dim=1)

# Matrix diagonal
# 'ii->i'
torch.einsum('ii->i', A) == A.diagonal()

# Attention scores (simplified)
# 'bqd,bkd->bqk'
torch.einsum('bqd,bkd->bqk', Q, K) == Q @ K.transpose(-2, -1)
```

**Common mistakes:**
1. ❌ Wrong index labels (mixing up dimensions)
2. ❌ Forgetting to specify output indices
3. ❌ Not understanding which indices are summed over
4. ❌ Inefficient einsum for simple operations

## Follow-up Questions
- When is einsum faster than manual operations?
- How does einsum handle broadcasting?
- What operations can't be expressed with einsum?
