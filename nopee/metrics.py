"""Truth-relative metrics: less motion by itself is not better prediction."""
import math

import torch


def score(prediction, truth):
    if prediction.shape != truth.shape or truth.ndim != 5 or truth.shape[1] < 2:
        raise ValueError("Metrics require matching B,T,C,H,W tensors with T >= 2.")
    mse = (prediction-truth).square().mean().item()
    mask = truth > .15
    fg = (prediction-truth).square()[mask].mean().item() if mask.any() else mse
    temporal = ((prediction[:, 1:]-prediction[:, :-1])-(truth[:, 1:]-truth[:, :-1])).square().mean().item()
    return dict(mse=mse, psnr_db=-10*math.log10(max(mse, 1e-12)), foreground_mse=fg,
                temporal_error=temporal)


def classical(prefix, horizon, method):
    previous, current = prefix[:, -2], prefix[:, -1]
    if method == "persistence":
        return current[:, None].expand(-1, horizon, -1, -1, -1).clone()
    if method != "linear":
        raise ValueError("Unknown classical baseline.")
    frames = []
    for _ in range(horizon):
        nxt = (2*current-previous).clamp(0, 1)
        frames.append(nxt)
        previous, current = current, nxt
    return torch.stack(frames, dim=1)
