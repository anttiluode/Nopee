"""Nopee Gradio demo. Run `python app.py`, or copy this repo into a Gradio Space."""
import json
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr
import pandas as pd
import torch

from nopee.data import make_world, observe
from nopee.experiment import load_bundle, resolve_device, timed
from nopee.media import artifact_directory, decode_video, export_comparison
from nopee.metrics import classical, score
from nopee.model import INTERVENTIONS, forecast

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(os.environ.get('NOPEE_CHECKPOINT', ROOT/'checkpoints/demo'))
torch.set_num_threads(max(1, int(os.environ.get('NOPEE_THREADS', '2'))))
DEVICE = resolve_device(os.environ.get('NOPEE_DEVICE', 'cpu'))
LABELS = {'fixed': 'Synthetic fixed', 'learned': 'Learned constants', 'shuffled': 'Shuffled fixed'}


@lru_cache(maxsize=2)
def _models(path):
    return load_bundle(path, DEVICE)


def _compare(truth, observed, variant, steps, intervention, checkpoint, source):
    models, meta = _models(str(checkpoint))
    if variant not in LABELS or intervention not in INTERVENTIONS:
        raise ValueError('Choose a listed adapter and intervention.')
    truth, observed = truth.to(DEVICE), observed.to(DEVICE)
    prefix_length = observed.shape[1]
    target = truth[:, prefix_length:]
    horizon = target.shape[1]
    predictions, rows, curves = {}, [], []
    for name in ('off', variant, 'persistence', 'linear'):
        if name in models:
            fn = lambda: forecast(models[name], observed, horizon, int(steps), intervention)
        else:
            fn = lambda: classical(observed, horizon, name)
        pred, latency = timed(fn, DEVICE, repeats=2)
        predictions[name] = pred
        rows.append({'System': name, 'MSE ↓': score(pred, target)['mse'],
                     'Foreground MSE ↓': score(pred, target)['foreground_mse'],
                     'Motion error ↓': score(pred, target)['temporal_error'],
                     'PSNR dB ↑': score(pred, target)['psnr_db'], 'Forecast ms ↓': latency})
        if name != 'linear':
            for t in range(horizon):
                curves.append({'Future frame': t+1, 'MSE': float((pred[:, t]-target[:, t]).square().mean()), 'System': name})
    directory = artifact_directory()
    video = export_comparison([truth[0], torch.cat((observed, predictions['off']), 1)[0],
                               torch.cat((observed, predictions[variant]), 1)[0]],
                              ['Reference', 'Adapter off', LABELS[variant]], prefix_length, directory)
    width, tau = models[variant].adapter.constants(shuffle=intervention == 'shuffle_constants')
    constants = {'lambda_pixels': width.detach().cpu().tolist(), 'tau_frames': tau.detach().cpu().tolist(),
                 'assignment': 'effective channel assignments after the selected intervention'}
    report = {'source': source, 'variant': variant, 'refinement_calls_per_frame': int(steps),
              'intervention': intervention, 'prefix_frames': prefix_length, 'forecast_frames': horizon,
              'device': DEVICE, 'checkpoint': meta, 'constants': constants, 'measurements': rows,
              'timing': 'Median of 2 warm runs, prefix processing included; excludes rendering and video encoding.',
              'warning': 'Single example. Lower error here does not establish general superiority or biological specificity.'}
    path = directory/'measurement.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    description = (f"**{source} · {horizon} future frames · {steps} refinement calls/frame · {DEVICE.upper()}**\n\n"
                   f"Checkpoint: {meta.get('label', 'Local checkpoint')}. "
                   f"Intervention: `{intervention}`. Each forecast starts with empty memory. "
                   "Timing includes prefix processing but excludes video encoding. "
                   "The reference's future frames are used only for scoring. "
                   "Fixed constants are synthetic; no living-neuron measurements are used.")
    return video, pd.DataFrame(rows).round(6), pd.DataFrame(curves), description, str(path)


def run_world(seed, variant, steps, noise, horizon, intervention, checkpoint=CHECKPOINT):
    seed, horizon = int(seed), int(horizon)
    if not 0 <= seed <= 2**31-500001 or not 2 <= horizon <= 32:
        raise gr.Error('Use a nonnegative seed below two billion and 2–32 forecast frames.')
    truth = make_world(seed, 1, 4+horizon)
    observed = observe(truth[:, :4], seed+500000, float(noise))
    try:
        return _compare(truth, observed, variant, int(steps), intervention, checkpoint, f'Toy world, seed {seed}')
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc


def run_upload(video, variant, steps, intervention):
    try:
        truth = decode_video(video, max_frames=28)
        return _compare(truth, truth[:, :4].clone(), variant, int(steps), intervention,
                        CHECKPOINT, 'Uploaded clip: out-of-distribution test, resized to 32×32 at 8 fps')
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc


CSS = """
.gradio-container {max-width: 1160px !important;}
#title {border-left: 5px solid #24ad8a; padding: 8px 20px; margin: 18px 0;}
#title h1 {font-size: 2.8rem; letter-spacing: -.06em; margin: 0;}
#title p {max-width: 830px;}
footer {display: none !important;}
"""


def _controls():
    with gr.Row():
        variant = gr.Dropdown([(v, k) for k, v in LABELS.items()], value='fixed', label='Adapter')
        steps = gr.Radio([1, 2, 4], value=2, label='Refinement calls per frame')
    intervention = gr.Dropdown(list(INTERVENTIONS), value='none', label='Intervention on the selected adapter',
                              info='Reset memory, bypass spatial filtering, or reassign constants without retraining.')
    return variant, steps, intervention


def _outputs():
    video = gr.Video(label='Reference / adapter off / selected adapter', interactive=False, autoplay=False)
    info = gr.Markdown()
    table = gr.Dataframe(label='Errors against future reference frames; smaller is better except PSNR', interactive=False)
    curve = gr.LinePlot(x='Future frame', y='MSE', color='System', title='How error grows into the future')
    report = gr.File(label='Download this measurement and checkpoint provenance')
    return video, table, curve, info, report


def build_app():
    with gr.Blocks(title='Nopee · Frame Lab', theme=gr.themes.Soft(primary_hue='emerald'), css=CSS,
                   delete_cache=(3600, 3600)) as demo:
        gr.Markdown('# Nopee\n### A little space. A little memory. An honest comparison.\n'
                    'Watch a tiny trained model predict what comes next. Compare a centre–surround filter '
                    'and E/I-like memory against the same predictor with its adapter off.', elem_id='title')
        gr.Markdown('**Research toy, not text-to-video.** 32×32 grayscale, trained on moving shapes. '
                    'Runs on CPU. No biological measurements, no TBC weights, no claimed speedup.')
        with gr.Tab('Moving world'):
            with gr.Row():
                seed = gr.Number(value=2026, precision=0, label='World seed', minimum=0, maximum=2000000000)
                noise = gr.Slider(0, .25, value=.08, step=.01, label='Noise in the 4 observed frames')
                horizon = gr.Slider(4, 32, value=12, step=1, label='Frames to predict')
            variant, steps, intervention = _controls()
            run = gr.Button('Predict and compare', variant='primary')
            outputs = _outputs()
            run.click(run_world, [seed, variant, steps, noise, horizon, intervention], outputs, api_name='world',
                      concurrency_limit=1, concurrency_id='inference')
        with gr.Tab('Your video / webcam'):
            gr.Markdown('Upload a short clip or record your webcam. The first four frames are observed; '
                        'the model predicts the rest. **It was trained only on toy shapes:** real footage '
                        'is a stress test and may blur or fail badly. Uploads are processed for inference only. '
                        'No audio is used. Clips are resized to 32×32, sampled at 8 fps, and capped at 28 frames / 50 MB. '
                        'On a hosted Space, your recording is sent to that server; run locally for local processing.')
            upload = gr.Video(sources=['upload', 'webcam'], include_audio=False, label='Short video')
            uv, us, ui = _controls()
            go = gr.Button('Test this clip', variant='primary')
            uo = _outputs()
            go.click(run_upload, [upload, uv, us, ui], uo, api_name='upload',
                     concurrency_limit=1, concurrency_id='inference')
        with gr.Tab('What is being tested?'):
            gr.Markdown('''### Structure versus constants
All variants share the same trained, frozen base. Fixed, learned and shuffled adapters
start from identical projection weights and train on the same sequence stream.

- **Synthetic fixed:** hand-chosen spatial widths and decay times.
- **Learned constants:** starts from those values; gradients can adjust them.
- **Shuffled fixed:** independently permutes spatial widths and times before training.
- **Adapter off:** the common base, with no added branch.

The branch uses a zero-sum difference-of-Gaussians and a stable, damped rotation of
two signed fields. The fields are E/I-inspired state variables, not simulated firing rates.
State advances once per video frame, regardless of refinement budget.

The controls do not identify a uniquely biological benefit. There are no measured
biological constants, and a retrained shuffled adapter can compensate in its projections.
An intervention may also put a model outside its training distribution.

### Read the errors, not just the smoothness
MSE and PSNR compare the predicted frames with the held-out future. Foreground error
keeps the small moving objects visible in the score. Motion error compares temporal
differences against the true temporal differences; a frozen output is not automatically good.

One, two and four steps are **actual calls to a learned residual refinement network**.
They are not diffusion sampling steps. Lower budgets need not give a favourable quality/speed tradeoff.
The persistence and linear baselines are deliberately included.

Use `python -m nopee benchmark` for paired held-out sequences, all budgets and CSV/JSON
measurements. Repeat training with several seeds before making a general claim.

[Source and instructions](https://github.com/anttiluode/Nopee) ·
[TBC research inspiration](https://www.tbc.co/post/a-sense-of-space-and-time-optimizing-interactive-video-generation)
''')
    return demo


if __name__ == '__main__':
    build_app().queue(max_size=8).launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', '7860')),
                                       max_file_size='50mb', show_error=True)
