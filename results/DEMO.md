# Nopee demonstration ledger — 2026-11-01

**The full DoG + E/I branch did not improve overall prediction in this run.** It generally worsened whole-frame error and added computation. Foreground error improved, but that is not an overall quality win. Removing the spatial filter at inference improved whole-frame error, which argues against claiming that the centre–surround component is beneficial here.

This is an exploratory engineering demonstration, one training seed, not a preregistered result or a validation of TBC. Do not select a favourable seed or metric and call it a breakthrough.

## Checkpoint and protocol

- Architecture: `transport-refiner-v1`; training seed 0.
- Base: 600 optimizer steps; each adapter: 200; batch 8.
- Parameters: base 8,563; with adapter 8,971.
- Training time on this host: 145.4 seconds. PyTorch 2.15.0+cpu, CPU, two threads.
- Train sequences: 7 frames; 4 observed, 3 autonomous training predictions (plus intermediate supervised predictions). Test: 4 observed, 12 autonomous future frames.
- Evaluation worlds: 91000–91011, checked against recorded training world seeds. Observation noise standard deviation 0.08, clipped to [0,1].
- All neural variants evaluated at 1, 2 and 4 refinement calls. Base weights remain identical. All adapter variants receive matched initial projection weights and training streams.
- Timing: per-sequence median of three warm synchronized runs, then mean across sequences. Includes prefix processing; excludes MP4 encoding. Shared-host timings fluctuate; no GPU or energy measurement.
- Weight SHA-256: `f4af6462a76b9377a610175162be4258865613eada38c33e3e2e4f8850f99cb5`.

## Two-call budget

| System | Whole-frame MSE ↓ | Foreground MSE ↓ | Motion error ↓ | Forecast ms ↓ |
|---|---:|---:|---:|---:|
| fixed | 0.03765 | 0.10266 | 0.00928 | 43.566 |
| learned | 0.03887 | 0.09793 | 0.00936 | 47.591 |
| linear | 0.15206 | 0.24454 | 0.01044 | 0.097 |
| off | 0.03390 | 0.12101 | 0.00904 | 41.047 |
| persistence | 0.04047 | 0.26024 | 0.00886 | 0.010 |
| shuffled | 0.03758 | 0.10395 | 0.00930 | 53.288 |

Persistence has lower motion error despite a worse foreground reconstruction. This illustrates why no single metric should be read as perceptual quality. More refinement calls also do not monotonically improve quality.

## Interventions on the trained adapters

These change the same checkpoints at inference, without retraining. All use two refinement calls and the same 12 evaluation worlds. Intervention timings use one measured warm run; do not use them for speed conclusions.

| Intervention | Fixed MSE | Learned MSE | Shuffled-fixed MSE |
|---|---:|---:|---:|
| none | 0.03765 | 0.03887 | 0.03758 |
| reset_memory | 0.03564 | 0.03630 | 0.03560 |
| bypass_spatial | 0.02538 | 0.02522 | 0.02358 |
| shuffle_constants | 0.03776 | 0.03885 | 0.03742 |

Clearing memory improves over the full adapter here, and bypassing the spatial filter improves further. These interventions show the branch affects predictions; they do **not** establish that either component is necessary or helpful. The next scientifically useful comparison would train memory-only and ordinary convolutional adapters under matched budgets, with multiple training seeds and longer prediction training.

## Limits and provenance

An initial direct-pixel refinement architecture blurred the toy shapes too quickly. Visual inspection prompted the transport-based base used here; all conditions were then retrained. Prototype development used world 2026 and an earlier evaluation range 90000–90011. The final ledger uses 91000–91011. The design and development choices were not preregistered.

The bundled model remains very small and degrades over longer forecasts, collisions and occlusion. Real footage is out of distribution. Fixed constants are synthetic pixels/frames. No measured biological constants were used. Freely trained projections may compensate for shuffled channel assignments.

Raw records: [all budgets](demo/benchmark.json), [summary CSV](demo/summary.csv), [paired sequences](demo/per_sequence.csv), and the three directories under [interventions](interventions/). Training metadata and losses are in [checkpoint metadata](../checkpoints/demo/metadata.json).
