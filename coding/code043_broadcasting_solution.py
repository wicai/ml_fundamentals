# PyTorch Broadcasting
# ====================================================================
#
# Broadcasting lets PyTorch operate on tensors of different shapes
# without copying data. Implement the functions below using only
# tensor operations — no Python loops.
#
# ====================================================================
import torch


def causal_mask(T: int) -> torch.Tensor:
    """
    Return a boolean mask of shape (T, T) where mask[i, j] is True
    if position j is allowed to attend to position i (i.e. j <= i).
    """
    i = torch.arange(T)[:, None]  # (T, 1)
    j = torch.arange(T)[None, :]  # (1, T)
    return j <= i                  # (T, T)


def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Args:
        lengths: shape (B,)
        max_len: int
    Returns:
        mask: shape (B, max_len), True where t < lengths[b]
    """
    positions = torch.arange(max_len, device=lengths.device)[None, :]  # (1, max_len)
    return positions < lengths[:, None]                                  # (B, max_len)


def pairwise_l2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Args:
        A: shape (m, d)
        B: shape (n, d)
    Returns:
        dists: shape (m, n)
    """
    return ((A[:, None] - B[None, :]) ** 2).sum(-1).sqrt()


def apply_padding_mask(
    logits: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        logits:   shape (B, T, V)
        pad_mask: shape (B, T), True = real token
    Returns:
        masked_logits: shape (B, T, V)
    """
    return logits.masked_fill(~pad_mask[:, :, None], float('-inf'))


if __name__ == "__main__":
    torch.manual_seed(1)

    # --- Part 1: Shape prediction ---
    print("=" * 60)
    print("PART 1: Shape prediction")
    print("=" * 60)

    a = torch.ones(5, 1, 4, 1)
    b = torch.ones(3, 1, 6)
    print(f"(5,1,4,1) + (3,1,6) = {(a + b).shape}")   # (5,3,4,6)

    c = torch.ones(5, 2, 4, 1)
    d = torch.ones(3, 1, 6)
    try:
        _ = c + d
    except RuntimeError as e:
        print(f"(5,2,4,1) + (3,1,6) → RuntimeError: {e}")

    e = torch.ones(4,)
    f = torch.ones(4, 1)
    print(f"(4,) + (4,1) = {(e + f).shape}")          # (4,4) — the gotcha

    # --- Part 2: causal_mask ---
    print("\n" + "=" * 60)
    print("PART 2a: causal_mask")
    print("=" * 60)
    mask = causal_mask(5)
    print(f"Shape: {mask.shape}")
    print(mask.int())

    # --- Part 2: sequence_mask ---
    print("\n" + "=" * 60)
    print("PART 2b: sequence_mask")
    print("=" * 60)
    lengths = torch.tensor([3, 1, 5, 0])
    smask = sequence_mask(lengths, max_len=5)
    print(f"Shape: {smask.shape}")
    print(smask.int())
    # Row 0: [1,1,1,0,0], Row 1: [1,0,0,0,0], Row 2: [1,1,1,1,1], Row 3: [0,0,0,0,0]

    # --- Part 2: pairwise_l2 ---
    print("\n" + "=" * 60)
    print("PART 2c: pairwise_l2")
    print("=" * 60)
    A = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    dists = pairwise_l2(A, B)
    print(f"Shape: {dists.shape}")
    print(dists)
    # A[0]→B[0]=1, A[0]→B[1]=1, A[1]→B[0]=0, A[1]→B[1]=sqrt(2), etc.

    # --- Part 2: apply_padding_mask ---
    print("\n" + "=" * 60)
    print("PART 2d: apply_padding_mask")
    print("=" * 60)
    logits = torch.zeros(2, 3, 5)  # (B=2, T=3, V=5)
    pad_mask = torch.tensor([[True, True, False], [True, False, False]])
    masked = apply_padding_mask(logits, pad_mask)
    probs = masked.softmax(-1)
    print(f"Probs at padded position (should be 0): {probs[0, 2, 0].item():.4f}")
    print(f"Probs at real position (uniform over V): {probs[0, 0, 0].item():.4f}")

    # --- Part 3: gradient shape ---
    print("\n" + "=" * 60)
    print("PART 3: Gradient shape")
    print("=" * 60)
    a = torch.ones(3, 1, requires_grad=True)
    b = torch.ones(1, 4)
    c = (a + b).sum()
    c.backward()
    print(f"a.grad.shape: {a.grad.shape}")           # (3, 1)
    print(f"a.grad values:\n{a.grad}")               # all 4.0
