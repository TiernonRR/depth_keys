from pathlib import Path
import sys
import types

import pytest

import depth_keys.proc as proc

try:
    import depth_keys.experiment.trial
except ModuleNotFoundError as exc:
    if exc.name != "sleap_io":
        raise
    sys.modules["sleap_io"] = types.ModuleType("sleap_io")
    import depth_keys.experiment.trial


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_check_directory_requires_each_2d_3d_and_render_artifact(tmp_path):
    _touch(tmp_path / "a.avi")
    _touch(tmp_path / "b.avi")
    d2 = tmp_path / "_kpoints_v1_2d"
    d3 = tmp_path / "_kpoints_v1_3d"
    renders = tmp_path / "renders"
    for stem in ("a", "b"):
        _touch(d2 / f"{stem}.slp")
        _touch(d3 / f"{stem}.pkl.gz")
    _touch(d3 / "merged_keypoints.h5")
    _touch(renders / "keypoints_overlay_v1.mp4")
    _touch(renders / "matplotlib_render.mp4")
    assert proc.check_directory(str(tmp_path)) == {"2d": True, "3d": True, "render": True}
    (d2 / "b.slp").unlink()
    assert proc.check_directory(str(tmp_path))["2d"] is False


def test_check_directory_accepts_cached_2d_pickle_and_missing_camera_3d_is_incomplete(tmp_path):
    _touch(tmp_path / "a.avi")
    _touch(tmp_path / "_kpoints_v1_2d" / "a.pkl.gz")
    _touch(tmp_path / "_kpoints_v1_3d" / "merged_keypoints.h5") # missing camera-specific 3d files
    result = proc.check_directory(str(tmp_path))
    assert result["2d"] is True
    assert result["3d"] is False
    assert result["render"] is False


@pytest.mark.xfail(strict=True, reason="check_directory uses all([]), so a source with no AVIs is incorrectly complete")
def test_check_directory_with_no_avis_is_not_complete(tmp_path):
    assert proc.check_directory(str(tmp_path)) == {"2d": False, "3d": False, "render": False}


@pytest.mark.xfail(strict=True, reason="check_directory currently ignores the configured renders output directory")
def test_check_directory_honors_custom_output_dirs(tmp_path):
    _touch(tmp_path / "a.avi")
    custom = {"kpoints_2d": "custom2d", "kpoints_3d": "custom3d", "renders": "customrenders"}
    _touch(tmp_path / "custom2d" / "a.slp")
    _touch(tmp_path / "custom3d" / "a.pkl.gz")
    _touch(tmp_path / "custom3d" / "merged_keypoints.h5")
    _touch(tmp_path / "customrenders" / "keypoints_overlay_v1.mp4")
    _touch(tmp_path / "customrenders" / "matplotlib_render.mp4")
    assert proc.check_directory(str(tmp_path), output_dirs=custom)["render"] is True


def test_process_directory_discovers_sorted_videos_and_runs_selected_stages(tmp_path, monkeypatch):
    source = tmp_path / "session"
    for stem in ("z", "a"):
        _touch(source / "_proc" / f"{stem}.avi")
    calls = []

    class FakeTrial:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
        def predict_keypoints(self, **kwargs):
            calls.append(("2d", kwargs))
        def compute_3d_keypoints(self, **kwargs):
            calls.append(("3d", kwargs))
        def visualize(self, **kwargs):
            calls.append(("render", kwargs))

    monkeypatch.setattr("depth_keys.experiment.trial.Trial", FakeTrial)
    proc.process_directory(
        str(source), "reg.toml", "ci", "centroid", "intr.toml", "trans.toml", "skeleton.json", ["nose"],
        version_num=3, reference_camera="cam", cable=True, compute_2d=True, compute_3d=True, render=True,
        force=False, output_dirs={"kpoints_2d": "two_{version}", "kpoints_3d": "three_{version}"},
    )
    init = calls[0][1]
    assert init["video_paths"] == [str(source / "_proc" / "a.avi"), str(source / "_proc" / "z.avi")]
    assert init["keypoints2d_output_path"].endswith("_proc/two_3")
    assert init["keypoints3d_output_path"].endswith("_proc/three_3")
    assert calls[1:] == [
        ("2d", {"ci_model_path": "ci", "centroid_model_path": "centroid"}),
        ("3d", {"registration_config_path": "reg.toml"}),
        ("render", {"matplot_viz": True, "overlay_viz": True, "output_dir": str(source / "_proc" / "renders"), "skeleton_json_path": "skeleton.json", "alt_key_path": str(source / "_proc" / "three_3" / "merged_keypoints.h5")}),
    ]


@pytest.mark.xfail(strict=True, reason="process_directory ignores output_dirs['renders'] and always uses _proc/renders")
def test_process_directory_honors_custom_render_output_dir(tmp_path, monkeypatch):
    source = tmp_path / "session"
    _touch(source / "_proc" / "cam.avi")
    calls = []

    class FakeTrial:
        def __init__(self, **kwargs): pass
        def visualize(self, **kwargs): calls.append(kwargs)

    monkeypatch.setattr("depth_keys.experiment.trial.Trial", FakeTrial)
    proc.process_directory(
        str(source), "r", "c", "u", "i", "t", "s", [], compute_2d=False,
        compute_3d=False, render=True, output_dirs={"renders": "custom-renders"},
    )
    assert calls[0]["output_dir"] == str(source / "_proc" / "custom-renders")


@pytest.mark.parametrize(
    "flags, expected",
    [
        ((True, False, False), ["2d"]),
        ((False, True, False), ["3d"]),
        ((False, False, True), ["render"]),
    ],
)
def test_process_directory_stage_flags(tmp_path, monkeypatch, flags, expected):
    source = tmp_path / "session"
    _touch(source / "_proc" / "cam.avi")
    calls = []

    class FakeTrial:
        def __init__(self, **kwargs): pass
        def predict_keypoints(self, **kwargs): calls.append("2d")
        def compute_3d_keypoints(self, **kwargs): calls.append("3d")
        def visualize(self, **kwargs): calls.append("render")
    monkeypatch.setattr("depth_keys.experiment.trial.Trial", FakeTrial)
    proc.process_directory(str(source), "r", "c", "u", "i", "t", "s", [], compute_2d=flags[0], compute_3d=flags[1], render=flags[2], force=False)
    assert calls == expected


@pytest.mark.xfail(strict=True, reason="process_directory passes exist_ok=False when force is false, so existing output directories raise")
def test_process_directory_force_false_allows_existing_output_dirs(tmp_path, monkeypatch):
    source = tmp_path / "session"
    _touch(source / "_proc" / "cam.avi")
    (source / "_proc" / "_kpoints_v1_2d").mkdir(parents=True)
    monkeypatch.setattr("depth_keys.experiment.trial.Trial", lambda **kwargs: type("T", (), {"predict_keypoints": lambda self, **k: None})())
    proc.process_directory(str(source), "r", "c", "u", "i", "t", "s", [], compute_2d=True, compute_3d=False, render=False, force=False)
