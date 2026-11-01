"""Live local API test, including Gradio's video preprocessing and output serialization."""
import socket
from pathlib import Path

from gradio_client import Client, handle_file

from app import build_app
from nopee.data import make_world
from nopee.media import export_comparison


def test_world_and_upload_over_http(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    app = build_app()
    app.queue(max_size=2).launch(server_name='127.0.0.1', server_port=port,
                               prevent_thread_lock=True, quiet=True)
    try:
        client = Client(f'http://127.0.0.1:{port}', verbose=False)
        world = client.predict(2026, 'fixed', 1, .08, 4, 'none', api_name='/world')
        assert len(world) == 5 and Path(world[0]['video']).is_file() and Path(world[-1]).is_file()
        source = export_comparison([make_world(21, 1, 8)[0]], ['Test clip'], 4, tmp_path)
        video_input = {'video': handle_file(source), 'subtitles': None}
        upload = client.predict(video_input, 'learned', 2, 'reset_memory', api_name='/upload')
        assert len(upload) == 5 and Path(upload[0]['video']).is_file() and Path(upload[-1]).is_file()
        assert 'out-of-distribution' in upload[3]
    finally:
        app.close()
