# Master Einsum for Tensor Operations
# ====================================================================
#
# Implement common tensor operations using `torch.einsum`.
# 
# Einsum (Einstein summation) provides a concise way to express tensor operations. Implement the following using einsum:
# - Matrix multiplication
# - Batch matrix multiplication
# - Attention computation
# - Tensor transpose
# - Outer product
# 
# **Function signatures:**
#
# ====================================================================

def matmul_einsum(A, B):
    """Matrix multiplication using einsum."""    
    return torch.einsum("ij, jk->ik", A, B)
    

def batch_matmul_einsum(A, B):
    """Batch matrix multiplication using einsum."""
    return torch.einsum("ijk, ikl -> ijl", A, B)

def attention_einsum(Q, K, V):
    """Compute attention using einsum."""
    pass

def transpose_einsum(x):
    """Transpose last two dimensions using einsum."""
    pass

def outer_product_einsum(a, b):
    """Outer product using einsum."""
    pass

