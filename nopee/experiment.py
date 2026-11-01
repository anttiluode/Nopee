"""Matched training, safe NumPy checkpoint format, and paired held-out evaluation."""
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .data import make_world, observe
from .metrics import classical, score
from .model import Predictor, VARIANTS, forecast, make_variants


def training_stream(seed, count):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield int(rng.integers(0, 2**31)), int(rng.integers(0, 2**31)), int(rng.choice([1, 2, 4]))


def resolve_device(device="auto"):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be auto, cpu or cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA PyTorch build or choose cpu.")
    return device


def _fit(model, count, seed, batch, device, progress=None):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.002)
    model.train()
    losses = []
    for n, (world_seed, noise_seed, steps) in enumerate(training_stream(seed, count)):
        clean = make_world(world_seed, batch, 7).to(device)
        noisy = observe(clean, noise_seed, .08)
        state, loss = None, 0.
        previous, current = noisy[:, 0], noisy[:, 1]
        # Observe four frames, then train a short autonomous rollout.
        for t in range(1, 6):
            prediction, state = model.step(previous, current, state, steps)
            target = clean[:, t+1]
            weight = 1 + 4*(target > .15).float()
            loss = loss + ((prediction-target).square()*weight).mean()
            previous, current = current, noisy[:, t+1] if t < 3 else prediction
        loss = loss/5
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if progress is not None and (n == 0 or (n+1) % 25 == 0 or n+1 == count):
            progress(f"{model.variant}: {n+1}/{count}, loss={losses[-1]:.5f}")
    model.eval()
    return losses


def train_experiment(seed=0, base_steps=240, adapter_steps=120, batch=8, device="auto", progress=None):
    if min(base_steps, adapter_steps, batch) < 1:
        raise ValueError("Training steps and batch size must be positive.")
    device = resolve_device(device)
    torch.manual_seed(seed)
    base_model = Predictor().to(device)
    started = time.perf_counter()
    logs = {"off": _fit(base_model, base_steps, seed+100, batch, device, progress)}
    models = make_variants(seed, base_model.base)
    for model in models.values():
        model.to(device)
        for p in model.base.parameters():
            p.requires_grad_(False)
    for name in VARIANTS[1:]:
        # Reset the data RNG in _fit: identical samples, noise and budgets across variants.
        logs[name] = _fit(models[name], adapter_steps, seed+200, batch, device, progress)
    metadata = dict(format_version=1, architecture='transport-refiner-v1', seed=seed, base_steps=base_steps, adapter_steps=adapter_steps,
                    batch=batch, device=device, torch=str(torch.__version__), numpy=np.__version__,
                    python=platform.python_version(), platform=platform.platform(),
                    training_seconds=time.perf_counter()-started, biological_measurements=False,
                    image_size=32, noise=.08, training_frames=7, training_prefix=4,
                    lambda_units="pixels", tau_units="frames", losses=logs,
                    label="Small synthetic-world demonstration; not a biological or speedup result.")
    metadata['training_world_seeds'] = sorted({x[0] for x in training_stream(seed+100, base_steps)} |
                                             {x[0] for x in training_stream(seed+200, adapter_steps)})
    source_dir = Path(__file__).parent
    metadata['source_sha256'] = {name: hashlib.sha256((source_dir/name).read_bytes()).hexdigest()
                               for name in ('model.py', 'data.py', 'experiment.py')}
    metadata["parameters"] = {}
    metadata["constants"] = {}
    for name, m in models.items():
        metadata["parameters"][name] = dict(total=sum(p.numel() for p in m.parameters()),
                                           trainable=sum(p.numel() for p in m.parameters() if p.requires_grad))
        if m.adapter:
            width, tau = m.adapter.constants()
            metadata["constants"][name] = dict(lambda_pixels=width.detach().cpu().tolist(), tau_frames=tau.detach().cpu().tolist())
    return models, metadata


def save_bundle(directory, models, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, model in models.items():
        arrays.update({f"{name}/{k}": v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    np.savez_compressed(directory / "weights.npz", **arrays)
    meta = dict(metadata)
    meta["weights_sha256"] = hashlib.sha256((directory / "weights.npz").read_bytes()).hexdigest()
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")


def load_bundle(directory, device="cpu"):
    device = resolve_device(device)
    try:
        return _load_bundle(directory, device)
    except FileNotFoundError:
        raise
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, EOFError) as exc:
        raise ValueError(f'Invalid checkpoint in {directory}: {exc}. Restore weights.npz and metadata.json together or retrain.') from exc


def _load_bundle(directory, device):
    directory = Path(directory)
    if not (directory / "weights.npz").is_file() or not (directory / "metadata.json").is_file():
        raise FileNotFoundError(f"Missing checkpoint in {directory}. Run: python -m nopee train --output {directory}")
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError('Checkpoint metadata must be a JSON object.')
    if meta.get('architecture', 'transport-refiner-v1') != 'transport-refiner-v1':
        raise ValueError('Unsupported checkpoint architecture; retrain with this version of Nopee.')
    models = {}
    with np.load(directory / "weights.npz", allow_pickle=False) as arrays:
        for name in VARIANTS:
            model = Predictor(name)
            state = {}
            for key in model.state_dict():
                a = arrays[f"{name}/{key}"]
                if not np.isfinite(a).all():
                    raise ValueError("Checkpoint weights must be finite.")
                state[key] = torch.from_numpy(a.copy())
            model.load_state_dict(state, strict=True)
            models[name] = model.to(resolve_device(device)).eval()
    expected_hash = meta.get('weights_sha256')
    if expected_hash and hashlib.sha256((directory/'weights.npz').read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Checkpoint checksum mismatch. Restore both weights.npz and metadata.json together.')
    return models, meta


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def timed(call, device, repeats=3):
    call()  # warm kernels; each forecast constructs fresh recurrent state
    times = []
    result = None
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        result = call()
        _sync(device)
        times.append((time.perf_counter()-started)*1000)
    return result, float(np.median(times))


@torch.no_grad()
def benchmark(models, seeds=None, budgets=(1, 2, 4), horizon=12, prefix_length=4, noise=.08, repeats=3,
              intervention="none", training_seeds=()):
    seeds = list(range(90000, 90012)) if seeds is None else list(seeds)
    if not seeds or repeats < 1 or horizon < 2 or prefix_length < 2:
        raise ValueError("Use nonempty seeds, repeats >= 1, horizon >= 2 and prefix >= 2.")
    if not budgets or any(b not in (1, 2, 4) for b in budgets):
        raise ValueError("Budgets must contain only 1, 2 or 4.")
    if set(seeds) & set(training_seeds):
        raise ValueError('Evaluation world seeds overlap training; choose a different seed-start.')
    device = next(models['off'].parameters()).device
    rows = []
    for seed in seeds:
        truth = make_world(seed, 1, prefix_length+horizon).to(device)
        prefix = observe(truth[:, :prefix_length], seed+500000, noise)
        target = truth[:, prefix_length:]
        # Rotate condition order by sequence to reduce systematic timing-order bias.
        conditions = [(name, b) for name in models for b in budgets]
        conditions += [("persistence", 0), ("linear", 0)]
        order = np.random.default_rng(seed).permutation(len(conditions))
        for index in order:
            name, budget = conditions[index]
            if name in models:
                call = lambda: forecast(models[name], prefix, horizon, budget, intervention)
            else:
                call = lambda: classical(prefix, horizon, name)
            pred, elapsed = timed(call, device, repeats)
            rows.append(dict(seed=seed, variant=name, budget=budget, intervention=intervention,
                             **score(pred, target), latency_ms=elapsed,
                             predicted_frames_per_second=1000*horizon/max(elapsed, 1e-9)))
    summary = []
    for name, budget in sorted({(r['variant'], r['budget']) for r in rows}):
        group = [r for r in rows if r['variant'] == name and r['budget'] == budget]
        summary.append(dict(variant=name, budget=budget,
                            **{k: float(np.mean([r[k] for r in group])) for k in
                               ('mse', 'psnr_db', 'foreground_mse', 'temporal_error', 'latency_ms', 'predicted_frames_per_second')}))
    return dict(per_sequence=rows, summary=summary,
                protocol=dict(seeds=seeds, budgets=list(budgets), horizon=horizon, prefix_length=prefix_length,
                              noise=noise, repeats=repeats, timing="warm, synchronized, includes prefix processing; excludes video encoding",
                              device=str(device), cpu=platform.processor(), threads=torch.get_num_threads(),
                              gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                              torch=str(torch.__version__), intervention=intervention))


def write_report(directory, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for key in ('summary', 'per_sequence'):
        with (directory / f'{key}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(report[key][0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(report[key])
