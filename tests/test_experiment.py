import json

import numpy as np
import pytest
import torch

from nopee.data import make_world
from nopee.model import forecast, make_variants
from nopee.experiment import benchmark, load_bundle, save_bundle, train_experiment


def test_checkpoint_roundtrip_and_missing_error(tmp_path):
    models = make_variants(4)
    save_bundle(tmp_path, models, {'seed': 4, 'label': 'test'})
    loaded, meta = load_bundle(tmp_path)
    x = make_world(5, 1, 4)
    assert torch.equal(forecast(models['learned'], x, 4), forecast(loaded['learned'], x, 4))
    assert meta['seed'] == 4
    with pytest.raises(FileNotFoundError, match='train'):
        load_bundle(tmp_path / 'absent')


def test_short_training_freezes_shared_base_and_exports_paired_metrics(tmp_path):
    models, meta = train_experiment(seed=9, base_steps=2, adapter_steps=2, batch=2, device='cpu')
    for name, m in models.items():
        for k, value in models['off'].base.state_dict().items():
            assert torch.equal(value, m.base.state_dict()[k])
    assert meta['biological_measurements'] is False
    assert meta['base_steps'] == 2 and meta['adapter_steps'] == 2
    report = benchmark(models, seeds=[81001, 81002], budgets=[1, 2], horizon=4, repeats=1)
    assert len(report['per_sequence']) == 2 * (4*2+2)
    assert all(np.isfinite(r['mse']) and r['latency_ms'] >= 0 for r in report['per_sequence'])
    assert {r['seed'] for r in report['per_sequence']} == {81001, 81002}
    json.dumps(report, allow_nan=False)


def test_checkpoint_rejects_nonfinite_weights(tmp_path):
    models = make_variants(1)
    save_bundle(tmp_path, models, {})
    path = tmp_path / 'weights.npz'
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    key = next(iter(arrays))
    arrays[key] = arrays[key].astype('float32')
    arrays[key].flat[0] = np.nan
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='finite'):
        load_bundle(tmp_path)


def test_training_worlds_cannot_be_called_held_out():
    with pytest.raises(ValueError, match='overlap'):
        benchmark(make_variants(0), seeds=[12], training_seeds=[12], horizon=4)


def test_incomplete_checkpoint_has_actionable_error(tmp_path):
    save_bundle(tmp_path, make_variants(1), {})
    np.savez_compressed(tmp_path/'weights.npz', unrelated=np.zeros(1))
    with pytest.raises(ValueError, match='Restore.*retrain'):
        load_bundle(tmp_path)
