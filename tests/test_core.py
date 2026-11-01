import numpy as np
import pytest
import torch

from nopee.data import make_world
from nopee.model import Adapter, Predictor, forecast, make_variants
from nopee.metrics import score

torch.set_num_threads(2)


def test_world_is_reproducible_and_moves():
    x = make_world(seed=44, batch=2, frames=12)
    assert x.shape == (2, 12, 1, 32, 32)
    assert torch.equal(x, make_world(seed=44, batch=2, frames=12))
    assert not torch.equal(x, make_world(seed=45, batch=2, frames=12))
    assert x.min() >= 0 and x.max() <= 1
    assert (x[:, 1:] - x[:, :-1]).abs().mean() > .001


def test_dog_rejects_constant_field_and_has_trainable_width():
    a = Adapter('learned')
    kernels = a.kernels()
    assert torch.allclose(kernels.sum((-2, -1)), torch.zeros_like(kernels.sum((-2, -1))), atol=1e-6)
    x = torch.ones(2, a.channels, 32, 32)
    assert a.spatial(x).abs().max() < 2e-6
    a.spatial(torch.randn_like(x)).square().mean().backward()
    assert a.raw_lambda.grad is not None and a.raw_lambda.grad.abs().sum() > 0


def test_ei_state_decays_without_drive():
    a = Adapter('fixed')
    e, i = torch.randn(1, a.channels, 8, 8), torch.randn(1, a.channels, 8, 8)
    before = (e.square() + i.square()).sum()
    for _ in range(80):
        e, i = a.advance(torch.zeros_like(e), (e, i))
    assert (e.square() + i.square()).sum() < before * 1e-5


def test_variants_share_base_and_adapter_initialization():
    variants = make_variants(seed=12)
    ref = variants['fixed']
    for name, m in variants.items():
        for k, v in ref.base.state_dict().items():
            assert torch.equal(v, m.base.state_dict()[k])
        if name != 'off':
            assert torch.equal(ref.adapter.read.weight, m.adapter.read.weight)
    assert not torch.equal(ref.adapter.constants()[0], variants['shuffled'].adapter.constants()[0])


def test_forecast_uses_only_prefix_and_resets_between_calls():
    m = make_variants(3)['fixed'].eval()
    x = make_world(7, 1, 12)
    pred = forecast(m, x[:, :4], horizon=8, steps=2)
    x[:, 4:] = 1 - x[:, 4:]
    again = forecast(m, x[:, :4], horizon=8, steps=2)
    assert torch.equal(pred, again)
    assert torch.isfinite(pred).all() and pred.min() >= 0 and pred.max() <= 1


def test_budget_counts_refinement_calls_but_memory_once_per_frame():
    m = make_variants(3)['fixed']
    calls = []
    hook = m.base.refine.register_forward_hook(lambda *args: calls.append(1))
    x = make_world(7, 1, 6)
    _, s1 = m.step(x[:, 0], x[:, 1], None, steps=1)
    assert len(calls) == 1
    calls.clear()
    _, s4 = m.step(x[:, 0], x[:, 1], None, steps=4)
    assert len(calls) == 4
    assert all(torch.equal(a, b) for a, b in zip(s1, s4))
    hook.remove()


def test_budget_and_prefix_validation():
    m = Predictor()
    with pytest.raises(ValueError):
        forecast(m, torch.zeros(1, 1, 1, 32, 32), 3, 1)
    with pytest.raises(ValueError):
        forecast(m, torch.zeros(1, 4, 1, 32, 32), 3, 0)


def test_motion_metric_does_not_reward_frozen_output():
    truth = make_world(7, 2, 8)
    exact = score(truth, truth)
    frozen = score(truth[:, :1].expand_as(truth), truth)
    assert exact['mse'] == 0 and exact['temporal_error'] == 0
    assert frozen['temporal_error'] > 0
    assert frozen['foreground_mse'] > 0


def test_transport_moves_a_pixel_without_smearing_at_integer_offsets():
    from nopee.model import transport
    frame = torch.zeros(1, 1, 16, 16)
    frame[0, 0, 8, 8] = 1
    flow = torch.zeros(1, 2, 16, 16)
    flow[:, 0] = -1
    moved = transport(frame, flow)
    assert moved[0, 0, 8, 9] > .999
    assert torch.allclose(moved.sum(), frame.sum(), atol=1e-5)
    assert torch.allclose(transport(frame, torch.zeros_like(flow)), frame, atol=1e-5)
