import torch
import torch.nn as nn


def get_layer_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    captured = {}

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured['activations'] = hidden.mean(dim=1)

    hook = model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        hook.remove()

    return captured['activations']


def compute_steering_vector(
    model: nn.Module,
    positive_inputs: torch.Tensor,
    negative_inputs: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    pos_acts = get_layer_activations(model, positive_inputs, layer_idx)
    neg_acts = get_layer_activations(model, negative_inputs, layer_idx)
    return pos_acts.mean(dim=0) - neg_acts.mean(dim=0)


def hooked_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    layer_idx: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden, *rest = output
            steered = hidden + alpha * steering_vector.to(hidden.device)
            return (steered, *rest)
        return output + alpha * steering_vector.to(output.device)

    hook = model.layers[layer_idx].register_forward_hook(steering_hook)
    try:
        with torch.no_grad():
            logits = model(input_ids)
    finally:
        hook.remove()

    return logits
