# Implement RMSNorm (Llama)
# ====================================================================
#
# Implement RMSNorm as used in Llama models.
# 
# RMSNorm is a simpler alternative to LayerNorm that:
# - Only normalizes by RMS (no mean centering)
# - Has fewer operations (faster)
# - Only uses scale parameter (no shift)
# 
# **Function signature:**
#
# ====================================================================

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        """
        Args:
            normalized_shape: int, dimension to normalize
            eps: float, epsilon for numerical stability
        """
        super().__init__()        
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        """
        Args:
            x: tensor of shape (..., normalized_shape)
        Returns:
            normalized tensor of same shape as x
        """
        # get mean         
        x_rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / x_rms * self.gamma        

