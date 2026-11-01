"""Deterministic moving-shape worlds. No downloaded data or hidden future inputs."""
import numpy as np
import torch


def make_world(seed: int, batch: int = 1, frames: int = 20, size: int = 32) -> torch.Tensor:
    if batch < 1 or frames < 2 or size < 16:
        raise ValueError("Use batch >= 1, frames >= 2 and size >= 16.")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    out = np.zeros((batch, frames, 1, size, size), dtype=np.float32)
    for b in range(batch):
        # A faint fixed textured background makes preserving it part of the task.
        bg = .025 + .015 * np.sin(xx * .55 + rng.uniform(0, 6.28))
        out[b, :, 0] = bg
        for obj in range(2):
            radius = rng.uniform(2.1, 4.0) * size / 32
            start = rng.uniform(radius + 1, size - radius - 2, 2)
            vel = rng.uniform(.4, 1.4, 2) * rng.choice([-1, 1], 2) * size / 32
            brightness = rng.uniform(.5, .95)
            width = size - 1 - 2 * radius
            for t in range(frames):
                phase = np.mod(start - radius + vel * t, 2 * width)
                cx, cy = radius + np.where(phase <= width, phase, 2 * width - phase)
                distance = np.sqrt((xx - cx)**2 + (yy - cy)**2) if obj == 0 else np.maximum(abs(xx-cx), abs(yy-cy))
                mask = np.clip(radius + .5 - distance, 0, 1)
                out[b, t, 0] = np.maximum(out[b, t, 0], brightness * mask)
    return torch.from_numpy(out)


def observe(clean: torch.Tensor, seed: int, noise: float = .08) -> torch.Tensor:
    if not 0 <= noise <= .5:
        raise ValueError("Noise must be between 0 and 0.5.")
    rng = np.random.default_rng(seed)
    eps = torch.from_numpy(rng.standard_normal(tuple(clean.shape)).astype(np.float32)).to(clean.device)
    return (clean + noise * eps).clamp(0, 1)
