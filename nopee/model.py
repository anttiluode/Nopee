"""Tiny iterative predictor and a stable spatial / signed E-I memory branch.

This is not a diffusion model, an implementation of TBC, or a neuron simulator.
Lambda is in image pixels; tau is in video frames. State ticks once per frame.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("off", "fixed", "learned", "shuffled")
INTERVENTIONS = ("none", "reset_memory", "bypass_spatial", "shuffle_constants")


def transport(frame, flow):
    """Backward warp, flow in pixels; positive x samples to the right."""
    b, _, h, w = frame.shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=frame.device, dtype=frame.dtype),
                          torch.linspace(-1, 1, w, device=frame.device, dtype=frame.dtype), indexing='ij')
    grid = torch.stack((x, y), -1)[None].expand(b, -1, -1, -1)
    scale = frame.new_tensor([2/(w-1), 2/(h-1)])
    return F.grid_sample(frame, grid+flow.permute(0, 2, 3, 1)*scale,
                         mode='bilinear', padding_mode='border', align_corners=True)


def _logit(x):
    return torch.logit(x.clamp(1e-5, 1-1e-5))


class Adapter(nn.Module):
    def __init__(self, mode="fixed", features=16, channels=8):
        super().__init__()
        if mode not in VARIANTS[1:]:
            raise ValueError(f"Unknown adapter mode: {mode}")
        self.mode, self.channels = mode, channels
        widths = torch.linspace(.6, 1.6, channels)
        times = torch.linspace(2., 8., channels)
        if mode == "shuffled":
            g = torch.Generator().manual_seed(917)
            widths = widths[torch.randperm(channels, generator=g)]
            times = times[torch.randperm(channels, generator=g)]
        self.raw_lambda = nn.Parameter(_logit((widths-.4)/1.6), requires_grad=mode == "learned")
        self.raw_tau = nn.Parameter(_logit((times-1)/11), requires_grad=mode == "learned")
        self.write = nn.Conv2d(features, channels, 1)
        self.read = nn.Conv2d(2*channels, features, 1, bias=False)
        nn.init.normal_(self.read.weight, std=.005)
        y, x = torch.meshgrid(torch.arange(-3, 4), torch.arange(-3, 4), indexing="ij")
        self.register_buffer("distance2", (x*x+y*y).float())

    def constants(self, shuffle=False):
        width = .4 + 1.6 * self.raw_lambda.sigmoid()
        tau = 1 + 11 * self.raw_tau.sigmoid()
        if shuffle:
            width, tau = width.flip(0), tau.roll(3)
        return width, tau

    def kernels(self, shuffle=False):
        width, _ = self.constants(shuffle)
        def gaussian(sigma):
            k = torch.exp(-self.distance2[None] / (2*sigma[:, None, None]**2))
            return k / k.sum((-2, -1), keepdim=True)
        return (gaussian(width) - gaussian(2*width))[:, None]

    def spatial(self, x, shuffle=False):
        return F.conv2d(F.pad(x, (3, 3, 3, 3), mode="replicate"), self.kernels(shuffle), groups=self.channels)

    def advance(self, drive, state=None, shuffle=False):
        e, i = (torch.zeros_like(drive), torch.zeros_like(drive)) if state is None else state
        _, tau = self.constants(shuffle)
        r = torch.exp(-1/tau)[None, :, None, None]
        c, s = math.cos(.45), math.sin(.45)
        # A damped rotation has spectral radius r < 1 for every allowed tau.
        # Signed fields, not nonnegative firing rates. Units are frames, not ms.
        return r*(c*e-s*i)+(1-r)*drive, r*(s*e+c*i)

    def forward(self, features, state=None, intervention="none"):
        if intervention not in INTERVENTIONS:
            raise ValueError(f"Unknown intervention: {intervention}")
        drive = torch.tanh(self.write(features))
        shuffle = intervention == "shuffle_constants"
        if intervention != "bypass_spatial":
            drive = self.spatial(drive, shuffle)
        if intervention == "reset_memory":
            state = None
        state = self.advance(drive, state, shuffle)
        residual = .3 * torch.tanh(self.read(torch.cat(state, dim=1)))
        return residual, state


class Base(nn.Module):
    def __init__(self, features=16):
        super().__init__()
        self.encode = nn.Sequential(nn.Conv2d(3, features, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(features, features, 3, padding=1), nn.ReLU())
        self.refine = nn.Sequential(nn.Conv2d(features+5, features, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(features, features, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(features, 3, 3, padding=1))


class Predictor(nn.Module):
    def __init__(self, variant="off"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.base = Base()
        self.adapter = None if variant == "off" else Adapter(variant)

    def step(self, previous, current, state=None, steps=2, intervention="none"):
        if steps not in (1, 2, 4):
            raise ValueError("Refinement steps must be 1, 2 or 4.")
        if previous.shape != current.shape or current.ndim != 4 or current.shape[1] != 1:
            raise ValueError("Expected matching B,1,H,W frames.")
        features = self.base.encode(torch.cat((previous, current, current-previous), dim=1))
        if self.adapter is not None:
            residual, state = self.adapter(features, state, intervention)
            features = features + residual
        prediction = current
        flow = current.new_zeros(current.shape[0], 2, *current.shape[-2:])
        appearance = torch.zeros_like(current)
        for j in range(steps):
            fraction = torch.full_like(current, (j+1)/steps)
            budget = torch.full_like(current, 1/steps)
            delta = self.base.refine(torch.cat((features, prediction, current, current-previous, fraction, budget), dim=1))
            flow = (flow + 2*torch.tanh(delta[:, :2])/steps).clamp(-3, 3)
            appearance = (appearance + .1*torch.tanh(delta[:, 2:])/steps).clamp(-.2, .2)
            prediction = (transport(current, flow)+appearance).clamp(0, 1)
        return prediction, state


def make_variants(seed=0, base=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        common = copy.deepcopy(base) if base is not None else Base()
        models = {}
        for name in VARIANTS:
            torch.manual_seed(seed+100)
            model = Predictor(name)
            model.base.load_state_dict(common.state_dict())
            models[name] = model
    return models


@torch.no_grad()
def forecast(model, prefix, horizon=12, steps=2, intervention="none"):
    if prefix.ndim != 5 or prefix.shape[1] < 2 or prefix.shape[2] != 1:
        raise ValueError("Provide at least two observed grayscale frames (B,T,1,H,W).")
    if not 1 <= horizon <= 120:
        raise ValueError("Forecast horizon must be between 1 and 120.")
    if not torch.isfinite(prefix).all():
        raise ValueError("Input contains non-finite values.")
    state = None
    # Warm memory with observed pairs; discard their predictions.
    for t in range(1, prefix.shape[1]-1):
        _, state = model.step(prefix[:, t-1], prefix[:, t], state, steps, intervention)
    previous, current = prefix[:, -2], prefix[:, -1]
    frames = []
    for _ in range(horizon):
        predicted, state = model.step(previous, current, state, steps, intervention)
        frames.append(predicted)
        previous, current = current, predicted
    return torch.stack(frames, dim=1)
