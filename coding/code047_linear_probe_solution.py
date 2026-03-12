import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression


def extract_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
    position: int | str = 'last',
) -> torch.Tensor:
    captured = {}

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured['hidden'] = hidden.detach()

    hook = model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        hook.remove()

    hidden = captured['hidden']  # (batch, seq_len, hidden_dim)

    if position == 'last':
        return hidden[:, -1, :]
    elif position == 'mean':
        return hidden.mean(dim=1)
    else:
        return hidden[:, position, :]


def train_probe(
    hiddens_train: torch.Tensor,
    labels_train: torch.Tensor,
    max_iter: int = 1000,
    C: float = 1.0,
) -> LogisticRegression:
    X = hiddens_train.cpu().numpy()
    y = labels_train.cpu().numpy()
    probe = LogisticRegression(max_iter=max_iter, C=C, random_state=1)
    probe.fit(X, y)
    return probe


def probe_accuracy(
    probe: LogisticRegression,
    hiddens: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    X = hiddens.cpu().numpy()
    y = labels.cpu().numpy()
    preds = probe.predict(X)
    return float((preds == y).mean())


def layer_probe_sweep(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    val_inputs: torch.Tensor,
    val_labels: torch.Tensor,
    position: int | str = 'last',
) -> dict[str, list]:
    n_layers = len(model.layers)
    results: dict[str, list] = {'layer_idx': [], 'train_accuracy': [], 'val_accuracy': []}

    for layer_idx in range(n_layers):
        train_hiddens = extract_hidden_states(model, train_inputs, layer_idx, position)
        val_hiddens = extract_hidden_states(model, val_inputs, layer_idx, position)

        probe = train_probe(train_hiddens, train_labels)

        results['layer_idx'].append(layer_idx)
        results['train_accuracy'].append(probe_accuracy(probe, train_hiddens, train_labels))
        results['val_accuracy'].append(probe_accuracy(probe, val_hiddens, val_labels))

    return results
