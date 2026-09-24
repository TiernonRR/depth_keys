import io
import os
import subprocess

import h5py
import numpy as np
import pytest

from depth_keys.visualization import overlay_processor as op


class FakePipe:
    """Record pipe writes and expose configurable FFmpeg error output."""

    def __init__(self, stderr=b""):
        """Initialize the captured writes and error bytes."""
        self.writes = []
        self.closed = False
        self._stderr = stderr

    def write(self, data):
        """Record bytes written to the pipe."""
        self.writes.append(data)

    def close(self):
        """Mark the pipe as closed."""
        self.closed = True

    def read(self):
        """Return the configured error output."""
        return self._stderr


class FakeProcess:
    """Provide fake standard streams and track FFmpeg wait calls."""

    def __init__(self, stderr=b""):
        """Create input and error pipes for a fake subprocess."""
        self.stdin = FakePipe()
        self.stderr = FakePipe(stderr=stderr)
        self.wait_calls = 0

    def wait(self):
        """Record a wait and report a successful exit."""
        self.wait_calls += 1
        return 0


class FakeReader:
    """Serve predictable camera frames and record read requests."""

    def __init__(self, nframes=5, frame_size=(4, 3), frames=None):
        """Set frame count, dimensions, and optional frame content."""
        self.nframes = nframes
        self.frame_size = frame_size
        self.frames = np.zeros((nframes, frame_size[1], frame_size[0], 3), dtype=np.uint8) if frames is None else frames
        self.calls = []
        self.closed = False

    def get_frames(self, indices):
        """Return selected frames and record their indices."""
        self.calls.append(list(indices))
        return self.frames[indices]

    def get_file_info(self):
        """Leave the preconfigured video metadata unchanged."""
        return None

    def close(self):
        """Mark the fake reader as closed."""
        self.closed = True


def make_processor_files(tmp_path, cameras=("cam0", "cam1")):
    """Create session metadata and intrinsics files for processor tests."""
    session = tmp_path / "session"
    output = session / "_proc" / "_kpoints_v1_3d"
    output.mkdir(parents=True, exist_ok=True)
    (output / "merged_keypoints.toml").write_text(
        "cameras = [" + ", ".join(repr(c) for c in cameras) + "]\n"
    )
    intrinsics = tmp_path / "intrinsics.toml"
    intrinsics.write_text("dummy = true\n")
    return session, intrinsics, output


@pytest.fixture(autouse=True)
def fake_format_intrinsics(monkeypatch):
    """Replace intrinsics parsing with fixed matrices for both test cameras."""
    matrices = {
        "cam0": np.array([[10.0, 0, 2.0], [0, 20.0, 3.0], [0, 0, 1.0]]),
        "cam1": np.array([[12.0, 0, 4.0], [0, 24.0, 5.0], [0, 0, 1.0]]),
    }
    monkeypatch.setattr(op, "format_intrinsics", lambda _: (matrices, None))


def make_processor(tmp_path, **kwargs):
    """Create a processor with fixture files and optional constructor overrides."""
    session, intrinsics, _ = make_processor_files(tmp_path)
    args = dict(
        session_dir=str(session), version_num="1", reference_camera="cam0", intrinsics_file=str(intrinsics)
    )
    args.update(kwargs)
    return op.KeypointVideoProcessor(**args)


def test_inverse_project_world_coordinates_projection_scale_floor_and_invalid_depth():
    xyz = np.array([[1.0, 2.0, 2.0], [1.0, -1.0, 0.0], [np.nan, 1.0, 1.0]])

    result = op.inverse_project_world_coordinates(xyz, z_scale=2, floor_distance=None, cx=10, cy=20, fx=4, fy=5)

    np.testing.assert_allclose(result[0], [14, 30, 4])
    np.testing.assert_allclose(result[1], [4e10, -5e10, 0]) # is this desired behavior; if z = 0, then nan?
    assert np.isnan(result[2, 0])
    np.testing.assert_array_equal(xyz, np.array([[1.0, 2.0, 2.0], [1.0, -1.0, 0.0], [np.nan, 1.0, 1.0]]))

    floored = op.inverse_project_world_coordinates(
        np.array([[1.0, 2.0, 3.0]]), z_scale=2, floor_distance=5, cx=10, cy=20, fx=4, fy=5
    )
    np.testing.assert_allclose(floored, [[14, 30, 6]])


def test_mp4_writer_validates_extension():
    with pytest.raises(ValueError, match=".mp4"):
        op.MP4Writer("movie.avi", (4, 3), 30)


def test_mp4_writer_open_builds_command_and_honors_safe_prefix(monkeypatch, tmp_path):
    calls = []
    process = FakeProcess()
    monkeypatch.setattr(op.subprocess, "Popen", lambda *args, **kwargs: (calls.append((args, kwargs)) or process))
    writer = op.MP4Writer(str(tmp_path / "movie.mp4"), (4, 3), 30, prepend_args="source env.sh")

    writer.open()

    command, kwargs = calls[0]
    assert command[0].startswith("source env.sh ; ffmpeg -y")
    assert "-s 4x3" in command[0]
    assert kwargs == {"shell": True, "stdin": subprocess.PIPE, "stderr": subprocess.PIPE, "executable": "/bin/bash"}

    with pytest.raises(ValueError, match="Dangerous"):
        op.MP4Writer(str(tmp_path / "danger.mp4"), (1, 1), 1, prepend_args="sudo echo ok").open()


def test_mp4_writer_lazy_writes_uint8_bytes(monkeypatch, tmp_path):
    process = FakeProcess()
    monkeypatch.setattr(op.subprocess, "Popen", lambda *args, **kwargs: process)
    writer = op.MP4Writer(str(tmp_path / "movie.mp4"), (2, 1), 30)
    frames = np.array([[[[1.9, 2.1, 3.9], [4.0, 5.0, 6.0]]]], dtype=float)

    writer.write_frames(frames, progress_bar=False)

    assert process.stdin.writes == [bytes([1, 2, 3, 4, 5, 6])]


def test_mp4_writer_reports_and_reraises_broken_pipe(monkeypatch, capsys, tmp_path):
    process = FakeProcess(stderr=b"codec failed")
    process.stdin.write = lambda _: (_ for _ in ()).throw(BrokenPipeError("closed"))
    monkeypatch.setattr(op.subprocess, "Popen", lambda *args, **kwargs: process)
    writer = op.MP4Writer(str(tmp_path / "movie.mp4"), (1, 1), 30)

    with pytest.raises(BrokenPipeError):
        writer.write_frames(np.zeros((1, 1, 1, 3), dtype=np.uint8), progress_bar=False)
    assert "codec failed" in capsys.readouterr().out


def test_mp4_writer_close_closes_waits_reports_and_clears(monkeypatch, capsys, tmp_path):
    process = FakeProcess(stderr=b"warning")
    monkeypatch.setattr(op.subprocess, "Popen", lambda *args, **kwargs: process)
    writer = op.MP4Writer(str(tmp_path / "movie.mp4"), (1, 1), 30)
    writer.open()

    writer.close()

    assert process.stdin.closed
    assert process.wait_calls == 1
    assert writer.pipe is None
    assert "warning" in capsys.readouterr().out


def test_processor_init_sets_paths_metadata_and_creates_output(tmp_path):
    processor = make_processor(tmp_path, output_path=str(tmp_path / "custom"), n_frames=7)

    assert processor.output_dir == str(tmp_path / "custom")
    assert processor.video_dir.endswith(os.path.join("session", "_proc"))
    assert processor.cameras == ["cam0", "cam1"]
    assert processor.n_frames == 7
    assert (tmp_path / "custom").is_dir()


@pytest.mark.xfail(strict=True, reason="KeypointVideoProcessor currently discards prepend_args")
def test_processor_init_preserves_prepend_args_regression(tmp_path):
    processor = make_processor(tmp_path, prepend_args="source env.sh")
    assert processor.prepend_args == "source env.sh"


def test_processor_init_rejects_missing_keypoint_override(tmp_path):
    with pytest.raises(FileNotFoundError, match="Keypoint override"):
        make_processor(tmp_path, keypoint_file=str(tmp_path / "missing.h5"))


def write_keypoints(path, raw, smooth, conf=None):
    """Write raw, smoothed, and optional confidence datasets to HDF5."""
    with h5py.File(path, "w") as f:
        f.create_dataset("merged_keypoints_raw", data=raw)
        f.create_dataset("merged_keypoints_smooth", data=smooth)
        if conf is not None:
            f.create_dataset("proj_point_conf", data=conf)


def test_load_keypoints_selects_raw_or_smooth_and_projects_with_intrinsics(tmp_path):
    processor = make_processor(tmp_path)
    h5_path = tmp_path / "keys.h5"
    raw = np.array([[[1.0, 2.0, 2.0]]], dtype=np.float32)
    smooth = np.array([[[2.0, 4.0, 4.0]]], dtype=np.float32)
    write_keypoints(h5_path, raw, smooth)
    processor.keypoint_file = str(h5_path)

    np.testing.assert_allclose(processor.load_keypoints(), [[[7, 23, 4]]])
    processor.raw = True
    np.testing.assert_allclose(processor.load_keypoints(), [[[7, 23, 2]]])


def test_load_keypoints_loads_confidence_when_requested(tmp_path):
    processor = make_processor(tmp_path, cam_by_conf=True)
    h5_path = tmp_path / "keys.h5"
    conf = np.array([[[0.1]], [[0.9]]])
    write_keypoints(h5_path, np.ones((1, 1, 3)), np.ones((1, 1, 3)), conf)
    processor.keypoint_file = str(h5_path)

    processor.load_keypoints()

    np.testing.assert_array_equal(processor.conf, conf)


def test_load_keypoints_reports_missing_file_and_dataset(tmp_path):
    processor = make_processor(tmp_path)
    with pytest.raises(FileNotFoundError):
        processor.load_keypoints()

    path = tmp_path / "bad.h5"
    with h5py.File(path, "w"):
        pass
    processor.keypoint_file = str(path)
    with pytest.raises(KeyError):
        processor.load_keypoints()


def test_load_keypoints_missing_reference_intrinsics(tmp_path, monkeypatch):
    processor = make_processor(tmp_path, reference_camera="missing")
    path = tmp_path / "keys.h5"
    write_keypoints(path, np.ones((1, 1, 3)), np.ones((1, 1, 3)))
    processor.keypoint_file = str(path)

    with pytest.raises(KeyError):
        processor.load_keypoints()


def test_get_video_path_returns_existing_or_none(tmp_path):
    processor = make_processor(tmp_path)
    assert processor.get_video_path() is None
    path = tmp_path / "session" / "_proc" / "cam0.avi"
    path.touch()
    assert processor.get_video_path() == str(path)


def test_determine_video_length_uses_shortest_and_user_limit(tmp_path):
    processor = make_processor(tmp_path)
    reader = FakeReader(nframes=10)
    keypoints = np.zeros((7, 2, 3))
    assert processor.determine_video_length(reader, keypoints) == 7
    assert processor.n_frames == 7
    processor.n_frames = 4
    assert processor.determine_video_length(reader, keypoints) == 4
    assert processor.determine_video_length(None, keypoints) == 0


def test_load_video_batch_clamps_end_and_returns_opencv_frame_size(tmp_path):
    processor = make_processor(tmp_path)
    reader = FakeReader(nframes=3, frame_size=(4, 3))

    frames, frame_size = processor.load_video_batch(reader, 1, 10)

    assert reader.calls == [[1, 2]]
    assert frames.shape == (2, 3, 4, 3)
    assert frame_size == (3, 4)


def test_load_video_batch_returns_none_for_empty_batch(tmp_path):
    processor = make_processor(tmp_path)
    reader = FakeReader(nframes=2)
    reader.get_frames = lambda indices: np.empty((0, 3, 4, 3), dtype=np.uint8)
    assert processor.load_video_batch(reader, 2, 2) == (None, None)


def test_load_video_batch_returns_none_for_reader_none_result(tmp_path):
    processor = make_processor(tmp_path)
    reader = FakeReader(nframes=2)
    reader.get_frames = lambda indices: None

    assert processor.load_video_batch(reader, 0, 1) == (None, None)


def test_calculate_z_range_ignores_nan_and_limits_frames(tmp_path):
    processor = make_processor(tmp_path, n_frames=2)
    keys = np.array(
        [
            [[1, 2, np.nan], [1, 2, 4]],
            [[1, 2, -2], [1, 2, np.nan]],
            [[1, 2, 99], [1, 2, 99]],
        ],
        dtype=float,
    )
    assert processor.calculate_z_range(keys) == (-2, 4)
    processor.n_frames = 1
    assert processor.calculate_z_range(keys) == (4, 4)
    assert make_processor(tmp_path, n_frames=1).calculate_z_range(np.full((1, 1, 3), np.nan)) == (0, 1)


def test_draw_keypoints_handles_gray_one_channel_bgr_invalid_and_nonmutation(tmp_path):
    processor = make_processor(tmp_path)
    processor.keypoint_radius = 1
    frame = np.zeros((20, 20), dtype=np.uint8)
    original = frame.copy()
    keys = np.array([[5, 6, 0.5], [np.nan, 3, 1], [100, 3, 1], [2, 2, 1]])
    result = processor.draw_keypoints_on_frame(frame, keys, op.Normalize(0, 1))
    assert result.shape == (20, 20, 3)
    assert np.count_nonzero(result) > 0
    np.testing.assert_array_equal(frame, original)

    one_channel = np.zeros((20, 20, 1), dtype=np.uint8)
    assert processor.draw_keypoints_on_frame(one_channel, keys, op.Normalize(0, 1)).shape == (20, 20, 3)
    bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    assert processor.draw_keypoints_on_frame(bgr, keys, op.Normalize(0, 1)).shape == bgr.shape


def test_draw_keypoints_skips_short_malformed_point_and_draws_valid_point(tmp_path, monkeypatch):
    processor = make_processor(tmp_path)
    processor.keypoint_radius = 1
    monkeypatch.setattr(op.cv2, "putText", lambda *args, **kwargs: None)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    result = processor.draw_keypoints_on_frame(
        frame,
        [[3, 4], [8, 9, 0.5]],
        op.Normalize(0, 1),
    )

    # OpenCV indexes as [y, x], while the keypoint is supplied as [x, y].
    assert np.count_nonzero(result[9, 8]) > 0
    assert np.count_nonzero(result[3, 4]) == 0


def test_draw_keypoints_uses_reference_or_highest_confidence_camera_label(tmp_path, monkeypatch):
    processor = make_processor(tmp_path, cam_by_conf=True)
    labels = []
    monkeypatch.setattr(op.cv2, "putText", lambda image, text, *args: labels.append(text))
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    keys = np.array([[2, 2, 1]])
    processor.draw_keypoints_on_frame(frame, keys, op.Normalize(0, 1), np.array([[0.1, 0.2], [0.9, 0.8]]))
    assert labels[-1] == "cam1"
    labels.clear()
    with pytest.warns(RuntimeWarning, match="Mean of empty slice") as caught:
        processor.draw_keypoints_on_frame(
            frame, keys, op.Normalize(0, 1), np.full((2, 2), np.nan)
        )
    assert len(caught) == 1
    assert labels[-1] == "cam0"


def test_add_colorbar_to_frame_normal_case_mutates_frame_and_has_labels_and_orientation(tmp_path, monkeypatch):
    processor = make_processor(tmp_path)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    labels = []
    monkeypatch.setattr(op.cv2, "putText", lambda image, text, *args, **kwargs: labels.append(text))

    result = processor.add_colorbar_to_frame(frame, 1, 5)

    assert result is frame
    assert labels == ["5.00", "1.00"]
    colorbar_x = frame.shape[1] - 20 - 30
    colorbar_y = (frame.shape[0] - frame.shape[0] // 2) // 2
    expected_top = tuple(int(c * 255) for c in processor.colormap(1.0)[:3][::-1])
    expected_bottom = tuple(int(c * 255) for c in processor.colormap(0.0)[:3][::-1])
    np.testing.assert_array_equal(frame[colorbar_y, colorbar_x], expected_top)
    np.testing.assert_array_equal(frame[colorbar_y + frame.shape[0] // 2 - 1, colorbar_x], expected_bottom)


@pytest.mark.xfail(strict=True, reason="Current colorbar implementation cannot place a bar in tiny frames")
def test_add_colorbar_tiny_frame_has_defined_behavior(tmp_path):
    processor = make_processor(tmp_path)
    processor.add_colorbar_to_frame(np.zeros((1, 1, 3), dtype=np.uint8), 0, 1)


def test_process_frame_batch_empty_overflow_confidence_and_single_write(tmp_path, monkeypatch):
    processor = make_processor(tmp_path, cam_by_conf=True)
    processor.conf = np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2)
    frames = np.zeros((2, 5, 6, 3), dtype=np.uint8)
    keys = np.ones((1, 1, 3), dtype=float)
    writer = FakeProcess()
    writer.write_frames = lambda batch, progress_bar: setattr(writer, "batch", batch)
    seen_conf = []
    monkeypatch.setattr(processor, "draw_keypoints_on_frame", lambda frame, k, normalizer, conf: (seen_conf.append(conf) or frame.copy()))
    monkeypatch.setattr(processor, "add_colorbar_to_frame", lambda frame, *_: frame)

    processor.process_frame_batch(frames, keys, 0, 2, 0, 1, writer)
    assert len(writer.batch) == 1
    assert len(seen_conf) == 1
    np.testing.assert_array_equal(seen_conf[0], processor.conf[:, 0])

    writer2 = FakeProcess()
    processor.process_frame_batch(None, keys, 0, 1, 0, 1, writer2)
    assert not hasattr(writer2, "batch")


def test_create_video_smooth_raw_and_custom_output_names_and_batching(tmp_path, monkeypatch):
    processor = make_processor(tmp_path, n_frames=3, batch_size=2)
    calls = []

    class Writer:
        def __init__(self, path, frame_size, fps, prepend_args):
            calls.append(("init", path, frame_size, fps, prepend_args))
        def open(self): calls.append(("open",))
        def close(self): calls.append(("close",))

    monkeypatch.setattr(op, "MP4Writer", Writer)
    monkeypatch.setattr(processor, "load_video_batch", lambda reader, start, end: (np.zeros((end-start, 3, 4, 3), dtype=np.uint8), (3, 4)))
    monkeypatch.setattr(processor, "process_frame_batch", lambda *args: calls.append(("batch", args[2], args[3])))
    processor.create_video(np.zeros((3, 1, 3)), 0, 1, (3, 4), object())
    assert calls[0][1].endswith("keypoints_overlay_v1.mp4")
    assert calls[0][2] == (4, 3)
    assert calls[-1] == ("close",)
    assert [x[1:] for x in calls if x[0] == "batch"] == [(0, 2), (2, 3)]

    calls.clear()
    processor.raw = True
    processor.save_name = None
    processor.frame_start = processor.frame_end = None
    processor.create_video(np.zeros((3, 1, 3)), 0, 1, (3, 4), object())
    assert calls[0][1].endswith("keypoints_overlay_v1-raw.mp4")
    assert [x[1:] for x in calls if x[0] == "batch"] == [(0, 2), (2, 3)]

    calls.clear()
    processor.save_name = "custom"
    processor.frame_start, processor.frame_end = 1, 3
    processor.create_video(np.zeros((3, 1, 3)), 0, 1, (3, 4), object())
    assert calls[0][1].endswith("custom.mp4")
    assert [x[1:] for x in calls if x[0] == "batch"] == [(1, 3)]


@pytest.mark.xfail(strict=True, reason="create_video does not close writer when batch processing raises")
def test_create_video_closes_writer_on_processing_error(tmp_path, monkeypatch):
    processor = make_processor(tmp_path, n_frames=1)
    closed = []

    class Writer:
        def __init__(self, *args, **kwargs): pass
        def open(self): pass
        def close(self): closed.append(True)

    monkeypatch.setattr(op, "MP4Writer", Writer)
    monkeypatch.setattr(processor, "load_video_batch", lambda *args: (np.zeros((1, 3, 4, 3), dtype=np.uint8), (3, 4)))
    monkeypatch.setattr(processor, "process_frame_batch", lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        processor.create_video(np.zeros((1, 1, 3)), 0, 1, (3, 4), object())
    assert closed


def prepare_process_mocks(tmp_path, monkeypatch):
    """Create a processor with stubbed I/O for process-flow tests."""
    processor = make_processor(tmp_path, n_frames=3)
    reader = FakeReader(nframes=3)
    monkeypatch.setattr(processor, "load_keypoints", lambda: np.zeros((3, 1, 3)))
    monkeypatch.setattr(processor, "get_video_path", lambda: str(tmp_path / "session" / "_proc" / "cam0.avi"))
    monkeypatch.setattr(op, "AviReader", lambda *args, **kwargs: reader)
    monkeypatch.setattr(processor, "calculate_z_range", lambda keys: (0, 1))
    monkeypatch.setattr(processor, "load_video_batch", lambda *args: (np.zeros((1, 3, 4, 3), dtype=np.uint8), (3, 4)))
    monkeypatch.setattr(processor, "create_video", lambda *args: None)
    return processor, reader


def test_process_missing_video_returns_without_reader(tmp_path, monkeypatch):
    processor = make_processor(tmp_path)
    monkeypatch.setattr(processor, "load_keypoints", lambda: np.zeros((1, 1, 3)))
    monkeypatch.setattr(processor, "get_video_path", lambda: None)
    called = []
    monkeypatch.setattr(op, "AviReader", lambda *args: called.append(True))
    assert processor.process() is None
    assert not called


@pytest.mark.parametrize("kwargs", [{"frame_start": 1}, {"frame_end": 2}, {"frame_start": 3, "frame_end": 2}])
def test_process_validates_and_clamps_frame_bounds(tmp_path, monkeypatch, kwargs):
    processor, reader = prepare_process_mocks(tmp_path, monkeypatch)
    processor.frame_start = kwargs.get("frame_start")
    processor.frame_end = kwargs.get("frame_end")
    with pytest.raises(ValueError):
        processor.process()
    assert reader.closed


def test_process_clamps_negative_and_out_of_range_bounds_and_closes_reader(tmp_path, monkeypatch):
    processor, reader = prepare_process_mocks(tmp_path, monkeypatch,)
    processor.frame_start, processor.frame_end = -2, 100
    processor.process()
    assert (processor.frame_start, processor.frame_end) == (0, 3)
    assert reader.closed


def test_process_closes_reader_when_create_fails(tmp_path, monkeypatch):
    processor, reader = prepare_process_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(processor, "create_video", lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        processor.process()
    assert reader.closed
