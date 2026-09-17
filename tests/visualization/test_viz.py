import numpy as np
import pytest
from types import SimpleNamespace

from depth_keys.visualization import viz


def test_create_overlay_video_constructs_processor_and_processes(monkeypatch, tmp_path):
    calls = []

    class Processor:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
        def process(self):
            calls.append(("process",))

    monkeypatch.setattr(viz, "KeypointVideoProcessor", Processor)
    viz.create_overlay_video(
        "session", "2", "cam0", "intrinsics.toml", "env", str(tmp_path / "out"),
        keypoint_file="keys.h5", n_frames=4, raw=True,
    )

    assert calls[0] == ("init", {
        "session_dir": "session", "version_num": "2", "reference_camera": "cam0",
        "output_path": str(tmp_path / "out"), "intrinsics_file": "intrinsics.toml",
        "conda_env_name": "env", "keypoint_file": "keys.h5", "n_frames": 4, "raw": True,
    })
    assert calls[1] == ("process",)


def test_render_3d_matplotlib_inverts_z_does_not_mutate_and_forwards_limits(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        viz.pcl,
        "viz",
        SimpleNamespace(visualize_xyz_trajectories_to_mp4=lambda *args, **kwargs: calls.append((args, kwargs))),
        raising=False,
    )
    merged = np.array([
        [[-1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[0.0, 0.0, 1.0], [2.0, 3.0, 4.0]],
        [[5.0, -2.0, 2.0], [1.0, 1.0, 8.0]],
    ])
    original = merged.copy()

    viz.render_3d_matplotlib(merged, [(0, 1)], str(tmp_path), "render", fps=24, burn_in=1, max_frames=2)

    args, kwargs = calls[0]
    np.testing.assert_array_equal(args[0][..., 2], -merged[..., 2])
    np.testing.assert_array_equal(merged, original)
    assert args[1] == str(tmp_path / "render.mp4")
    assert kwargs["fps"] == 24
    assert list(kwargs["frame_ids"]) == [1]
    assert kwargs["skeleton_edges"] == [(0, 1)]
    assert kwargs["trail_length"] == 5
    assert kwargs["xlim"] == (-6, 10)
    assert kwargs["ylim"] == (-7, 10)
    assert kwargs["zlim"] == (-13, 4)
    assert (tmp_path).is_dir()


def test_render_3d_defaults_to_all_frames_after_burn_in(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        viz.pcl,
        "viz",
        SimpleNamespace(visualize_xyz_trajectories_to_mp4=lambda *a, **kw: seen.update(kw)),
        raising=False,
    )
    data = np.ones((4, 1, 3))
    viz.render_3d_matplotlib(data, [], str(tmp_path), "movie", burn_in=2)
    assert list(seen["frame_ids"]) == [2, 3]


@pytest.mark.xfail(strict=True, reason="render_3d_matplotlib currently calls nanmin/nanmax on empty data")
def test_render_3d_empty_data_has_defined_behavior(monkeypatch, tmp_path):
    monkeypatch.setattr(viz.pcl, "viz", SimpleNamespace(visualize_xyz_trajectories_to_mp4=lambda *a, **kw: None), raising=False)
    viz.render_3d_matplotlib(np.empty((0, 1, 3)), [], str(tmp_path), "empty")


def test_render_3d_all_nan_data_has_defined_behavior(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        viz.pcl,
        "viz",
        SimpleNamespace(visualize_xyz_trajectories_to_mp4=lambda *a, **kw: calls.append((a, kw))),
        raising=False,
    )

    with pytest.warns(RuntimeWarning, match="All-NaN slice encountered") as caught:
        viz.render_3d_matplotlib(
            np.full((2, 1, 3), np.nan), [], str(tmp_path), "nan", burn_in=0
        )
    assert len(caught) == 6

    args, kwargs = calls[0]
    assert args[1] == str(tmp_path / "nan.mp4")
    assert list(kwargs["frame_ids"]) == [0, 1]
    for limit in (kwargs["xlim"], kwargs["ylim"], kwargs["zlim"]):
        assert all(np.isnan(value) for value in limit)
