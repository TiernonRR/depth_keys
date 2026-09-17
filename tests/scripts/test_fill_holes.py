'''
    NOTE this script tests obsolete code. Consider removing.
'''
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_script():
    util = types.ModuleType("markovids.vid.util")
    util.fill_holes = lambda frame, **kwargs: frame
    io = types.ModuleType("markovids.vid.io")
    vid = types.ModuleType("markovids.vid")
    vid.util = util
    vid.io = io
    markovids = types.ModuleType("markovids")
    markovids.vid = vid
    path = Path(__file__).parents[2] / "scripts" / "fill_holes.py"
    spec = importlib.util.spec_from_file_location("test_fill_holes_script", path)
    module = importlib.util.module_from_spec(spec)
    names = ("markovids", "markovids.vid", "markovids.vid.util", "markovids.vid.io")
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    try:
        sys.modules.update({"markovids": markovids, "markovids.vid": vid, "markovids.vid.util": util, "markovids.vid.io": io})
        spec.loader.exec_module(module)
    finally:
        for name in names:
            old = previous[name]
            if old is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return module


fill_holes = _load_script()


def test_avi_writer_rejects_non_avi(tmp_path):
    with pytest.raises(RuntimeError, match="avi"):
        fill_holes.AviWriter(str(tmp_path / "out.mp4"))


def test_avi_writer_opens_lazily_writes_configured_dtype_and_closes(tmp_path, monkeypatch):
    class Pipe:
        def __init__(self):
            self.stdin = self
            self.data = bytearray()
            self.closed = False
            self.waited = False
        def write(self, data): self.data.extend(data)
        def close(self): self.closed = True
        def wait(self): self.waited = True
    pipe = Pipe()
    commands = []
    monkeypatch.setattr(fill_holes.subprocess, "Popen", lambda *a, **k: commands.append((a, k)) or pipe)
    writer = fill_holes.AviWriter(str(tmp_path / "out.avi"), frame_size=(2, 1), dtype=np.dtype("<u2"), prepend_args="source env")
    writer.write_frames(np.array([[[1, 2]], [[3, 4]]], dtype=np.uint8), progress_bar=False)
    assert pipe.data == np.array([[[1, 2]], [[3, 4]]], dtype="<u2").tobytes()
    assert commands and commands[0][1]["shell"] is True
    assert "source env ;" in commands[0][0][0]
    writer.close()
    assert pipe.closed
    assert pipe.waited


def test_preprocess_batch_copies_squeezes_and_passes_fill_options(monkeypatch):
    calls = []
    monkeypatch.setattr(fill_holes.util, "fill_holes", lambda frame, **kwargs: calls.append((frame.copy(), kwargs)) or frame + 1)
    frames = np.zeros((2, 2, 3, 1), dtype=np.uint16)
    result = fill_holes.preprocess_batch(frames, "kernel")
    assert result.shape == (2, 2, 3)
    assert np.all(result == 1)
    assert np.all(frames == 0)
    assert len(calls) == 2
    assert all(call[1] == {"fill_kernel": "kernel", "iterations": fill_holes.FILL_ITERATIONS} for call in calls)


class _Reader:
    def __init__(self, nframes):
        self.nframes = nframes
        self.frame_size = (2, 1)
        self.fps = 50
        self.pixel_format = "gray16le"
        self.dtype = np.dtype("<u2")
        self.indices = []
    def get_file_info(self): pass
    def get_frames(self, indices):
        self.indices.append(np.asarray(indices))
        return np.zeros((len(indices), 1, 2), dtype=np.uint16)


def test_process_video_batches_reader_output_and_writer_parameters(tmp_path, monkeypatch):
    reader = _Reader(5)
    monkeypatch.setattr(fill_holes.io, "AviReader", lambda *a, **k: reader, raising=False)
    calls = []
    class Writer:
        def __init__(self, **kwargs): calls.append(("init", kwargs)); self.batches = []; self.closed = False
        def write_frames(self, frames, **kwargs): self.batches.append(frames.copy())
        def close(self): self.closed = True
    writer = None
    def make_writer(**kwargs):
        nonlocal writer
        writer = Writer(**kwargs)
        return writer
    monkeypatch.setattr(fill_holes, "AviWriter", make_writer)
    monkeypatch.setattr(fill_holes, "preprocess_batch", lambda frames, kernel: frames.astype(np.float32))
    monkeypatch.setattr(fill_holes, "BATCH_SIZE", 2)
    assert fill_holes.process_video(str(tmp_path / "in.avi"), str(tmp_path / "out" / "out.avi")) is True
    assert [x.tolist() for x in reader.indices] == [[0, 1], [2, 3], [4]]
    assert calls[0][1]["frame_size"] == (2, 1)
    assert calls[0][1]["dtype"] == reader.dtype
    assert len(writer.batches) == 3 and all(batch.dtype == reader.dtype for batch in writer.batches)
    assert writer.closed


def test_process_video_zero_frames_returns_false(monkeypatch, tmp_path):
    reader = _Reader(0)
    monkeypatch.setattr(fill_holes.io, "AviReader", lambda *a, **k: reader, raising=False)
    assert fill_holes.process_video(str(tmp_path / "in.avi"), str(tmp_path / "out.avi")) is False


def test_process_video_closes_writer_after_batch_failure(monkeypatch, tmp_path):
    reader = _Reader(1)
    monkeypatch.setattr(fill_holes.io, "AviReader", lambda *a, **k: reader, raising=False)
    class Writer:
        closed = False
        def __init__(self, **kwargs): pass
        def write_frames(self, *args, **kwargs): raise ValueError("bad frame")
        def close(self): self.closed = True
    writer = Writer()
    monkeypatch.setattr(fill_holes, "AviWriter", lambda **kwargs: writer)
    assert fill_holes.process_video(str(tmp_path / "in.avi"), str(tmp_path / "out.avi")) is False
    assert writer.closed
