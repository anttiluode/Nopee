# Nopee: a small experiment in selective spatiotemporal computation

Build requested by Antti, 2026-11-01. Implementation choices below are engineering
defaults, not claims about TBC's proprietary product or cortical physiology.

## Scope

A runnable Python/PyTorch research toy with a Gradio app deployable unchanged to
Hugging Face Spaces. It predicts grayscale 32x32 moving-shape frames after a short
observed prefix. A video upload/webcam recording is an explicitly out-of-distribution
test, not a realistic video-generation promise. No accounts, external model downloads,
cloud GPU, or training on visitors' uploads. Ship an actual small trained checkpoint.

## Model and controls

A shared convolutional predictor refines a pixel-displacement field and small
appearance correction 1, 2 or 4 times, then backward-warps the last frame.
These are actual network evaluations, not diffusion/denoising steps. A parallel
adapter uses normalized centre-minus-surround Gaussian kernels and a stable signed
E/I-like damped rotation with an input drive. State advances once per video frame,
not once per refinement. All variant rollouts use only the same observed prefix,
then their own predictions; future truth is used only for metrics.

Train a common base, freeze it, initialize identical adapters, and train learned,
synthetic-fixed and independently shuffled-fixed spatial/temporal constants under
equal data and optimizer budgets. Also report adapter-off, persistence and linear
extrapolation. Fixed constants are illustrative pixel/frame units, NOT measured
biology. A shuffled-and-retrained adapter can compensate through its projections;
this is not proof of biological specificity. Inference-time memory reset, spatial
bypass and constant reassignment are separate interventions, not retrained controls.

Compare held-out sequences and refinement budgets with MSE, PSNR, foreground MSE,
temporal-difference error against truth, and measured synchronized latency. Raw
flicker alone rewards frozen/blurry outputs. Log data seeds, train budget, parameters,
constants, hardware, versions and per-sequence measurements. Small bundled run is
a demonstration, not a positive-result guarantee or a preregistered claim.

## Alternatives and choice

Wan fine-tuning would be closer to commercial video but impractical as a small CPU
Space and too costly for routine 12 GB experimentation. A pure image smoother would
be cheap but would not test autonomous prediction. Choose a tiny learned recurrent
predictor: modest task, real training, explicit controls, reproducible locally.

## Delivery

GitHub source, tests/CI, trained demo weights with provenance and held-out results,
CLI training/benchmark/export, Gradio toy and upload tabs, Windows instructions,
Space YAML/dependencies. No Hugging Face publication without a destination.

## Implementation decision

Visual inspection of the first direct-pixel prototype showed premature blur. The
shipped base uses learned transport instead. This choice was made during development,
not preregistered; the bundled evaluation is exploratory. All adapter conditions
share the revised base and are retrained from scratch before final reporting.
