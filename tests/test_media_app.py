from pathlib import Path

import pytest
import torch

from nopee.data import make_world
from nopee.media import decode_video, export_comparison


def test_video_export_decode_roundtrip_and_limit(tmp_path):
    x = make_world(2, 1, 16)[0]
    path = export_comparison([x, x, x], ['Truth', 'Off', 'Fixed'], prefix_length=4, directory=tmp_path)
    assert Path(path).is_file() and Path(path).stat().st_size > 100
    decoded = decode_video(path, max_frames=8)
    assert decoded.shape == (1, 8, 1, 32, 32)
    assert torch.isfinite(decoded).all()


def test_bad_and_short_video_errors(tmp_path):
    bad = tmp_path / 'bad.mp4'
    bad.write_text('not video')
    with pytest.raises(ValueError, match='decode'):
        decode_video(bad)
    x = make_world(2, 1, 2)[0]
    short = export_comparison([x], ['Short'], 1, tmp_path)
    with pytest.raises(ValueError, match='frames'):
        decode_video(short, min_frames=6)


def test_demo_callback_uses_checkpoint_and_produces_artifacts(tmp_path):
    from nopee.experiment import save_bundle
    from nopee.model import make_variants
    from app import run_world, build_app
    save_bundle(tmp_path, make_variants(2), {'label': 'untrained test only'})
    video, table, curve, info, report = run_world(12, 'fixed', 2, .08, 6, 'none', checkpoint=tmp_path)
    assert Path(video).is_file() and Path(report).is_file()
    assert len(table) == 4 and len(curve) == 18
    assert 'untrained test only' in info
    assert build_app().config['mode'] == 'blocks'


def test_generated_exports_are_bounded_and_other_folders_untouched(tmp_path):
    import os
    import time
    from nopee.media import artifact_directory
    old = tmp_path / 'nopee-old'
    old.mkdir()
    os.utime(old, (time.time()-7200, time.time()-7200))
    unrelated = tmp_path / 'personal'
    unrelated.mkdir()
    current = artifact_directory(tmp_path)
    assert current.is_dir() and not old.exists() and unrelated.is_dir()


def test_report_records_effective_shuffled_constants(tmp_path):
    import json
    from app import run_world
    from nopee.experiment import save_bundle
    from nopee.model import make_variants
    models = make_variants(3)
    original = models['fixed'].adapter.constants()[0].detach().tolist()
    save_bundle(tmp_path, models, {'constants': {'fixed': {'lambda_pixels': original}}})
    result = run_world(31, 'fixed', 1, .08, 4, 'shuffle_constants', checkpoint=tmp_path)
    report = json.loads(Path(result[-1]).read_text())
    assert report['constants']['lambda_pixels'] == list(reversed(original))
