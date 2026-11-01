import argparse
from pathlib import Path

import torch

from .experiment import benchmark, load_bundle, resolve_device, save_bundle, train_experiment, write_report


def main():
    parser = argparse.ArgumentParser(description='Nopee: tiny frame prediction with matched adapter controls')
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train', help='Train a base and three adapter controls on synthetic worlds')
    train.add_argument('--output', type=Path, default=Path('runs/seed0'))
    train.add_argument('--seed', type=int, default=0)
    train.add_argument('--base-steps', type=int, default=240)
    train.add_argument('--adapter-steps', type=int, default=120)
    train.add_argument('--batch', type=int, default=8)
    bench = sub.add_parser('benchmark', help='Evaluate identical held-out sequences at 1/2/4 refinement calls')
    bench.add_argument('--checkpoint', type=Path, default=Path('checkpoints/demo'))
    bench.add_argument('--output', type=Path, default=Path('outputs/benchmark'))
    bench.add_argument('--sequences', type=int, default=12)
    bench.add_argument('--seed-start', type=int, default=90000)
    bench.add_argument('--horizon', type=int, default=12)
    bench.add_argument('--noise', type=float, default=.08)
    bench.add_argument('--repeats', type=int, default=3)
    bench.add_argument('--intervention', choices=['none', 'reset_memory', 'bypass_spatial', 'shuffle_constants'], default='none')
    for child in (train, bench):
        child.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        child.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if args.command == 'train':
        models, meta = train_experiment(args.seed, args.base_steps, args.adapter_steps, args.batch, device,
                                        progress=lambda s: print(s, flush=True))
        save_bundle(args.output, models, meta)
        print(f'Saved trained weights and provenance to {args.output}')
    else:
        models, meta = load_bundle(args.checkpoint, device)
        report = benchmark(models, range(args.seed_start, args.seed_start+args.sequences),
                           horizon=args.horizon, noise=args.noise, repeats=args.repeats, intervention=args.intervention,
                           training_seeds=meta.get('training_world_seeds', []))
        report['checkpoint'] = dict(path=str(args.checkpoint), seed=meta.get('seed'), weights_sha256=meta.get('weights_sha256'))
        write_report(args.output, report)
        for r in report['summary']:
            print(f"{r['variant']:12s} budget={r['budget']} MSE={r['mse']:.5f} motion-error={r['temporal_error']:.5f} ms={r['latency_ms']:.1f}")
        print(f'Saved paired measurements to {args.output}')


if __name__ == '__main__':
    main()
