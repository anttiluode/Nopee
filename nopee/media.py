"""Bounded video I/O; raw uploads are never used for training."""
import tempfile
import shutil
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw


def artifact_directory(root=None):
    """Remove only our generated export directories, never arbitrary upload paths."""
    root = Path(root) if root is not None else Path(tempfile.gettempdir())/'nopee-exports'
    root.mkdir(parents=True, exist_ok=True)
    entries = sorted((p for p in root.glob('nopee-*') if p.is_dir() and not p.is_symlink()),
                     key=lambda p: p.stat().st_mtime)
    now = time.time()
    for index, path in enumerate(entries):
        if now-path.stat().st_mtime > 3600 or index < len(entries)-63:
            shutil.rmtree(path)
    return Path(tempfile.mkdtemp(prefix='nopee-', dir=root))


def decode_video(path, max_frames=48, min_frames=6, fps=8):
    if path is None or not Path(path).is_file():
        raise ValueError('Upload or record a video first.')
    if Path(path).stat().st_size > 50*1024*1024:
        raise ValueError('Use a video smaller than 50 MB.')
    frames = []
    reader = None
    try:
        # Resize inside ffmpeg, before frames reach Python; cap duration and frames.
        reader = imageio.get_reader(str(path), format='FFMPEG', size=(32, 32), fps=fps,
                                   output_params=['-t', str(max_frames/fps)])
        for frame in reader:
            rgb = frame[..., :3].astype(np.float32)/255
            gray = rgb @ np.array([.299, .587, .114], dtype=np.float32)
            frames.append(gray)
            if len(frames) >= max_frames:
                break
    except Exception as exc:
        raise ValueError('Could not decode this video. Try a short MP4 or WebM clip.') from exc
    finally:
        if reader is not None:
            reader.close()
    if len(frames) < min_frames:
        raise ValueError(f'The clip needs at least {min_frames} frames at {fps} fps; record a little longer.')
    return torch.from_numpy(np.stack(frames))[None, :, None]


def export_comparison(sequences, labels, prefix_length=4, directory=None, fps=8):
    if len(sequences) != len(labels) or not sequences:
        raise ValueError('Each video needs a label.')
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError('Comparison videos must have identical lengths.')
    directory = Path(directory) if directory else artifact_directory()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'comparison.mp4'
    width, panel, height = 192*len(sequences), 192, 240
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264', macro_block_size=1,
                               ffmpeg_log_level='error', pixelformat='yuv420p')
    try:
        for t in range(next(iter(lengths))):
            canvas = Image.new('RGB', (width, height), '#111c22')
            draw = ImageDraw.Draw(canvas)
            for j, (seq, label) in enumerate(zip(sequences, labels)):
                frame = seq[t].detach().cpu().squeeze().clamp(0, 1).numpy()
                tile = Image.fromarray((frame*255).round().astype('uint8')).convert('RGB')
                canvas.paste(tile.resize((panel, panel), Image.Resampling.NEAREST), (j*panel, 24))
                draw.text((j*panel+8, 6), label, fill='#8ce4c8')
            status = 'OBSERVED PREFIX' if t < prefix_length else f'FORECAST +{t-prefix_length+1}'
            draw.text((8, 222), status, fill='#e6eef1')
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
    return str(path)
