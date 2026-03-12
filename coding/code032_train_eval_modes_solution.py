# Implement Train vs Eval Mode Effects
# ====================================================================
#
# Demonstrate and verify the concrete effects of `model.train()` vs `model.eval()` on model behavior.
# 
# Two PyTorch modules behave differently in train vs eval mode:
# - **Dropout**: active during training, disabled during eval
# - **BatchNorm**: uses batch statistics during training, running statistics during eval
# 
# Write functions that demonstrate and verify these differences.
# 
# **Function signature:**
#
# ====================================================================
from torch import nn
import torch
def demonstrate_dropout_modes(drop_prob: float = 0.5) -> dict[str, bool]:
    """
    Show that dropout is active in train mode and inactive in eval mode.

    Args:
        drop_prob: dropout probability
    Returns:
        dict with:
            'train_outputs_vary': bool — True if multiple forward passes give
                different results in train mode
            'eval_outputs_same': bool — True if multiple forward passes give
                identical results in eval mode
            'eval_preserves_values': bool — True if eval mode output equals input
                (no dropout applied)
    """
    proof = {}
    d = nn.Dropout(drop_prob)
    d.train()
    x = torch.ones(1,100)
    out1 = d(x)
    out2 = d(x)
    proof['train_outputs_vary'] = not torch.equal(out1, out2)
    d.eval()
    out1 = d(x)
    out2 = d(x)
    proof['eval_outputs_same'] = torch.equal(out1, out2)
    proof['eval_preserves_values'] = torch.equal(x, out1)
    return proof    

def demonstrate_batchnorm_modes() -> dict[str, bool]:
    """
    Show that batchnorm uses batch stats in train mode and running stats in eval.

    Returns:
        dict with:
            'train_uses_batch_stats': bool — True if output depends on batch
                composition in train mode
            'eval_uses_running_stats': bool — True if output is the same regardless
                of batch composition in eval mode
            'running_stats_updated_during_train': bool — True if running_mean/var
                change after a forward pass in train mode
    """
    feature_dim = 3
    bn = nn.BatchNorm1d(feature_dim)
    bn.train()
    b1 = torch.tensor([[1,2,3],[2,3,4]], dtype=torch.float32)
    b2 = torch.tensor([[1,2,3],[10,20,30]], dtype=torch.float32)    
    b1_out = bn(b1)
    running_mean_pre = bn.running_mean.clone()
    b2_out = bn(b2)
    running_mean_post = bn.running_mean.clone()
    proof = {}
    proof['running_stats_updated_during_train'] = not torch.equal(running_mean_pre, running_mean_post)    
    proof['train_uses_batch_stats'] = not torch.equal(b1_out[0],b2_out[0]) # if they aren't equal, it is batch dependent

    bn.eval()
    b1_out = bn(b1)
    b2_out = bn(b2)
    proof['eval_uses_running_stats'] = torch.equal(b1_out[0], b2_out[0])
    
    

def freeze_for_inference(model: nn.Module) -> nn.Module:
    """
    Prepare a model for inference: eval mode + no gradients.
    Return the model and a context manager for no_grad.

    Args:
        model: nn.Module
    Returns:
        model in eval mode (modified in-place)
    """    
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

