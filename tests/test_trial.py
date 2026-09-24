import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

try:
    import depth_keys.experiment.trial as trial_module
except ModuleNotFoundError as exc:
    if exc.name != "sleap_io":
        raise
    sys.modules["sleap_io"] = types.ModuleType("sleap_io")
    import depth_keys.experiment.trial as trial_module

Trial = trial_module.Trial


def test_trial_defaults_are_instance_local_and_logger_is_retained():
    first = Trial("a", [])
    second = Trial("b", [])
    first.metadata["x"] = 1
    assert second.metadata == {}
    assert first.node_names == second.node_names == []
    logger = object()
    configured = Trial("c", [], metadata={"m": 2}, node_names=["nose"], logger=logger)
    assert configured.metadata == {"m": 2}
    assert configured.node_names == ["nose"]
    assert configured.logger is logger


def test_predict_keypoints_uses_default_output_and_one_call_per_video(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(trial_module, "run_inference_on_video", lambda **kwargs: calls.append(kwargs) or True)
    monkeypatch.chdir(tmp_path)
    videos = [str(tmp_path / "cam two.avi"), str(tmp_path / "cam1.avi")]
    trial = Trial("session", videos, version_num=4)
    trial.predict_keypoints("ci", "centroid")
    assert trial.keypoints2d_output_path == "./_keypoints_v4_2d"
    assert Path(trial.keypoints2d_output_path).is_dir()
    assert [c["video_path"] for c in calls] == videos
    assert calls[0]["output_path"].endswith("cam two.slp")
    assert all(c["ci_model_path"] == "ci" and c["centroid_model_path"] == "centroid" for c in calls)


def test_predict_keypoints_honors_custom_output_and_empty_input(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(trial_module, "run_inference_on_video", lambda **kwargs: calls.append(kwargs) or True)
    output = tmp_path / "predictions"
    trial = Trial("session", [], keypoints2d_output_path=str(output))
    trial.predict_keypoints()
    assert output.is_dir()
    assert calls == []


def _install_markovids_fakes(monkeypatch):
    """Install fake Markovids modules used by registration tests."""
    format_intrinsics = lambda data: ("K", "D")
    registration = lambda *args, **kwargs: None
    io_module = types.ModuleType("markovids.vid.io")
    io_module.format_intrinsics = format_intrinsics
    pipeline_module = types.ModuleType("markovids.pcl.pipeline")
    pipeline_module.registration_pipeline = registration
    vid_module = types.ModuleType("markovids.vid")
    vid_module.io = io_module
    pcl_module = types.ModuleType("markovids.pcl")
    pcl_module.pipeline = pipeline_module
    markovids = types.ModuleType("markovids")
    markovids.vid = vid_module
    markovids.pcl = pcl_module
    for name, module in {
        "markovids": markovids,
        "markovids.vid": vid_module,
        "markovids.vid.io": io_module,
        "markovids.pcl": pcl_module,
        "markovids.pcl.pipeline": pipeline_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return io_module, pipeline_module


def test_compute_3d_keypoints_forwards_conversion_and_registration(tmp_path, monkeypatch):
    io_module, pipeline_module = _install_markovids_fakes(monkeypatch)
    conversion = lambda *args, **kwargs: None
    monkeypatch.setattr(trial_module, "convert_2d_to_3d", conversion)
    convert_call = {}
    register_call = {}
    monkeypatch.setattr(trial_module, "convert_2d_to_3d", lambda *a, **k: convert_call.update(args=a, kwargs=k))
    monkeypatch.setattr(io_module, "format_intrinsics", lambda data: ("K", "D"))
    monkeypatch.setattr(pipeline_module, "registration_pipeline", lambda *a, **k: register_call.update(args=a, kwargs=k))
    intrinsics = tmp_path / "intrinsics.toml"
    intrinsics.write_text("fx = 1\n")
    trial = Trial(
        "session", ["cam.avi"], base_dir=str(tmp_path), version_num=3,
        keypoints2d_output_path="2d", keypoints3d_output_path="3d",
        node_names=["nose"], intrinsics_file=str(intrinsics), cable=True,
        conda_env_name="env", transforms_path="transforms", bundle_adjust=True,
        verbose=False,
    )
    trial.compute_3d_keypoints("registration.toml")
    assert convert_call["args"] == ("2d", ["cam.avi"], 3, True, ["nose"], "registration.toml")
    assert convert_call["kwargs"] == {"conda_env_name": "env"}
    assert register_call["args"] == ("registration.toml", str(tmp_path / "session"))
    assert register_call["kwargs"] == {
        "kpoints_save_dir": "3d", "intrinsics_matrix": "K",
        "distortion_coefficients": "D", "alt_save_dir": "3d",
        "bundle_adjust": True, "transforms_path": "transforms",
    }


def test_compute_3d_keypoints_does_not_register_when_conversion_fails(tmp_path, monkeypatch):
    _install_markovids_fakes(monkeypatch)
    monkeypatch.setattr(trial_module, "convert_2d_to_3d", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("conversion")))
    trial = Trial("session", [], base_dir=str(tmp_path), keypoints2d_output_path="2d", keypoints3d_output_path="3d", intrinsics_file=str(tmp_path / "missing.toml"))
    with pytest.raises(RuntimeError, match="conversion"):
        trial.compute_3d_keypoints("registration.toml")


def test_compute_3d_keypoints_default_output_path_regression(tmp_path, monkeypatch):
    _install_markovids_fakes(monkeypatch)
    monkeypatch.setattr(trial_module, "convert_2d_to_3d", lambda *a, **k: None)
    intrinsics = tmp_path / "intrinsics.toml"
    intrinsics.write_text("fx = 1\n") # need a file with text as the proceeding code will open
    trial = Trial("session", [], base_dir=str(tmp_path), keypoints2d_output_path="2d", intrinsics_file=str(intrinsics))
    trial.compute_3d_keypoints("registration.toml")
    assert trial.keypoints3d_output_path.endswith("_kpoints_v1_3d")


def _visualization_fixture(tmp_path):
    """Create merged keypoints, metadata, and a skeleton for render tests."""
    out = tmp_path / "3d"
    out.mkdir()
    with h5py.File(out / "merged_keypoints.h5", "w") as h5:
        h5.create_dataset("merged_keypoints_smooth", data=np.zeros((2, 2, 4), dtype=np.float32))
    metadata = {"kpoints": {"node_names": ["nose", "tail"]}, "reference_camera": "cam"}
    (out / "merged_keypoints.toml").write_text('reference_camera = "cam"\n[kpoints]\nnode_names = ["snout", "tail_tip"]\n')
    skeleton = tmp_path / "skeleton.json"
    skeleton.write_text(json.dumps([["snout", "tail_tip"]]))
    return out, skeleton


@pytest.mark.parametrize("matplot, overlay", [(True, False), (False, True), (True, True)])
def test_visualize_render_branches_and_forwarded_kwargs(tmp_path, monkeypatch, matplot, overlay):
    out, skeleton = _visualization_fixture(tmp_path)
    calls = {"matplot": [], "overlay": []}
    monkeypatch.setattr(trial_module.viz, "render_3d_matplotlib", lambda *a, **k: calls["matplot"].append((a, k)))
    monkeypatch.setattr(trial_module.viz, "create_overlay_video", lambda *a, **k: calls["overlay"].append((a, k)))
    trial = Trial("session", [], base_dir=str(tmp_path), keypoints3d_output_path=str(out), version_num=2, intrinsics_file="intrinsics.toml", conda_env_name="env")
    trial.visualize(matplot_viz=matplot, overlay_viz=overlay, skeleton_json_path=str(skeleton), max_frames_matplot=7, alt_key_path="alt.h5", n_frames=4)
    assert bool(calls["matplot"]) is matplot
    assert bool(calls["overlay"]) is overlay
    if matplot:
        args, kwargs = calls["matplot"][0]
        assert args[1] == [(0, 1)]
        assert kwargs["max_frames"] == 7
    if overlay:
        args, kwargs = calls["overlay"][0]
        assert kwargs["session_dir"] == str(tmp_path / "session")
        assert kwargs["keypoint_file"] == "alt.h5"
        assert kwargs["n_frames"] == 4


def test_visualize_reports_missing_inputs(tmp_path):
    trial = Trial("session", [], keypoints3d_output_path=None)
    with pytest.raises(ValueError, match="must be set"):
        trial.visualize()
    trial.keypoints3d_output_path = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        trial.visualize()


def test_visualize_reports_missing_dataset(tmp_path):
    out = tmp_path / "3d"
    out.mkdir()
    with h5py.File(out / "merged_keypoints.h5", "w"):
        pass
    trial = Trial("session", [], base_dir=str(tmp_path), keypoints3d_output_path=str(out))
    with pytest.raises(KeyError, match="merged_keypoints_smooth"):
        trial.visualize()
