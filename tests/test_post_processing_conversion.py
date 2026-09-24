"""Unit tests for depth-keypoint conversion and post-processing helpers.

The conversion module imports the optional SLEAP package at module import time.
The small import shim below keeps these tests runnable in environments where
SLEAP is not installed; individual tests replace ``sio.load_file`` with the
fake loader they need.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import joblib
import numpy as np
import pytest
import toml
from numpy.testing import assert_allclose, assert_array_equal


SRC = Path(__file__).parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import sleap_io  # noqa: F401
except ModuleNotFoundError:
    # ``conversion_computations`` only needs the module object until
    # convert_2d_to_3d calls ``load_file`` (which tests monkeypatch).
    sys.modules["sleap_io"] = types.SimpleNamespace(load_file=None)

from depth_keys.post_processing import conversion_computations as cc


def write_config(path: Path, *, legacy: bool = False, include_cable: bool = True,
                 include_bilateral: bool = True, depth_params=None) -> Path:
    """Write a tiny, real TOML config matching the production contract."""
    section = "post_processing" if legacy else "resolve_z"
    cfg = {section: {}}
    if depth_params is not None:
        cfg[section]["depth_patch_parameters"] = depth_params
    if include_bilateral:
        cfg[section]["depth_processing"] = {
            "bilateral_kwargs": {"d": 1, "sigmaColor": 2, "sigmaSpace": 3},
            "replace_height_spikes_kwargs": False,
        }
        if include_cable:
            cfg[section]["depth_processing_cable"] = {
                "bilateral_kwargs": {"d": 5, "sigmaColor": 6, "sigmaSpace": 7},
                "replace_height_spikes_kwargs": {"threshold": 9, "ksize": 7},
            }
    with path.open("w") as stream:
        toml.dump(cfg, stream)
    return path


def dump_toml(value, path: Path) -> Path:
    with path.open("w") as stream:
        toml.dump(value, stream)
    return path


def write_2d_artifacts(root: Path, cam: str, kpoints: np.ndarray,
                       node_names: list[str]) -> tuple[Path, Path]:
    directory = root / "_kpoints_v0_2d"
    directory.mkdir(parents=True, exist_ok=True)
    kpoint_file = directory / f"{cam}.pkl.gz"
    metadata_file = directory / f"{cam}.toml"
    joblib.dump(kpoints, kpoint_file)
    dump_toml({"node_names": node_names, "camera": cam}, metadata_file)
    return kpoint_file, metadata_file


class FakeReader:
    def __init__(self, frames: np.ndarray, *, frame_size=None):
        self.frames = frames
        self.nframes = len(frames)
        self.frame_size = frame_size or (frames.shape[2], frames.shape[1])
        self.ranges = []
        self.closed = False

    def get_frames(self, frame_range):
        indexes = list(frame_range)
        self.ranges.append(indexes)
        return self.frames[indexes]

    def close(self):
        self.closed = True


def test_to_plain_types_recurses_without_mutating_input():
    value = {"a": [1, (2, {"b": 3})]}
    result = cc._to_plain_types(value)
    assert result == value
    assert result is not value
    assert result["a"] is not value["a"]


def test_get_resolve_z_config_prefers_current_section():
    current = {"depth_patch_parameters": {"a": 1}}
    assert cc._get_resolve_z_config(
        {"resolve_z": current, "post_processing": {"old": True}}, "config.toml"
    ) is current


def test_get_resolve_z_config_falls_back_to_legacy_section():
    legacy = {"depth_patch_parameters": {"a": 1}}
    assert cc._get_resolve_z_config({"post_processing": legacy}, "config.toml") is legacy


def test_get_resolve_z_config_requires_a_section():
    with pytest.raises(KeyError, match="Missing \[resolve_z\]"):
        cc._get_resolve_z_config({}, "config.toml")


def test_load_depth_params_current_and_legacy(tmp_path):
    expected = {"nose": {"patch_radius": 2, "agg_func": 50}}
    current_path = write_config(tmp_path / "current.toml", depth_params=expected)
    legacy_path = write_config(tmp_path / "legacy.toml", legacy=True, depth_params=expected)
    assert cc._load_depth_params(current_path) == expected
    assert cc._load_depth_params(legacy_path) == expected


def test_load_depth_params_missing_table(tmp_path):
    path = write_config(tmp_path / "missing.toml")
    with pytest.raises(KeyError, match="depth_patch_parameters"):
        cc._load_depth_params(path)


def test_load_depth_params_rejects_non_table(tmp_path):
    path = tmp_path / "wrong.toml"
    dump_toml({"resolve_z": {"depth_patch_parameters": "not-a-table"}}, path)
    with pytest.raises(TypeError, match="must be a table"):
        cc._load_depth_params(path)


def test_load_depth_processing_overrides_selects_cable_and_non_cable(tmp_path):
    path = write_config(tmp_path / "config.toml", depth_params={})
    bilateral, spikes = cc._load_depth_processing_overrides(path, cable=False)
    assert bilateral == {"d": 1, "sigmaColor": 2, "sigmaSpace": 3}
    assert spikes is None
    bilateral, spikes = cc._load_depth_processing_overrides(path, cable=True)
    assert bilateral == {"d": 5, "sigmaColor": 6, "sigmaSpace": 7}
    assert spikes == {"threshold": 9, "ksize": 7}


@pytest.mark.parametrize("value", [False, {}, None])
def test_load_depth_processing_overrides_false_spike_setting_is_none(tmp_path, value):
    path = tmp_path / "config.toml"
    dump_toml({"resolve_z": {"depth_processing": {
        "bilateral_kwargs": {"d": 1}, "replace_height_spikes_kwargs": value,
    }}}, path)
    assert cc._load_depth_processing_overrides(path, cable=False) == ({"d": 1}, None)


def test_load_depth_processing_overrides_requires_variant_and_bilateral(tmp_path):
    no_variant = tmp_path / "no_variant.toml"
    dump_toml({"resolve_z": {}}, no_variant)
    with pytest.raises(KeyError, match="depth_processing_cable"):
        cc._load_depth_processing_overrides(no_variant, cable=True)

    no_bilateral = tmp_path / "no_bilateral.toml"
    dump_toml({"resolve_z": {"depth_processing": {}}}, no_bilateral)
    with pytest.raises(KeyError, match="bilateral_kwargs"):
        cc._load_depth_processing_overrides(no_bilateral, cable=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("snout", "snout"), (b"snout", "snout"), (b"a\xffb", "ab"), (3, "3")],
)
def test_normalize_node_name(value, expected):
    assert cc._normalize_node_name(value) == expected


def test_map_instance_points_uses_names_and_ignores_unknown_points():
    names = ["snout", "tail_tip"]
    arr = np.full((2, 3), np.nan)
    points = [
        {"name": b"tail_tip", "xy": [8, 9], "score": 0.8},
        {"name": "unknown", "xy": [1, 2], "score": 0.1},
        {"name": "snout", "xy": [3, 4], "score": 0.9},
    ]
    cc._map_instance_points_to_array(
        points, arr, {"snout": 0, "tail_tip": 1}, names
    )
    assert_array_equal(arr, [[3, 4, 0.9], [8, 9, 0.8]])


def test_map_instance_points_falls_back_positionally_only_for_missing_or_matching_names():
    names = ["snout", "tail_tip"]
    arr = np.full((2, 3), np.nan)
    points = [
        {"xy": [1, 2], "score": 0.2},
        {"name": "tail_tip", "xy": [3, 4], "score": 0.4},
    ]
    cc._map_instance_points_to_array(points, arr, {"snout": 0, "tail_tip": 1}, names)
    assert_array_equal(arr, [[1, 2, 0.2], [3, 4, 0.4]])


def test_replace_height_spikes_replaces_isolated_spike_without_mutation():
    source = np.full((5, 5), 100, dtype=np.float32)
    source[2, 2] = 500
    original = source.copy()
    result = cc.replace_height_spikes(source, threshold=30, ksize=5)
    assert result.dtype == source.dtype
    assert result[2, 2] == 100
    assert_array_equal(source, original)


def test_replace_height_spikes_leaves_difference_equal_to_threshold():
    source = np.full((5, 5), 100, dtype=np.float32)
    source[2, 2] = 130
    result = cc.replace_height_spikes(source, threshold=30, ksize=5)
    assert result[2, 2] == 130


def test_replace_height_spikes_exercises_scaled_large_kernel_branch():
    source = np.full((9, 9), 100, dtype=np.float32)
    source[4, 4] = 500
    result = cc.replace_height_spikes(source, threshold=30, ksize=7, z_scale=4)
    assert result[4, 4] == 100
    assert result.dtype == source.dtype


def test_get_3d_kpoints_happy_path_batches_edges_and_invalid_values(tmp_path, monkeypatch):
    avi = tmp_path / "cam0.avi"
    avi.touch()
    config = write_config(tmp_path / "config.toml", depth_params={
        "snout": {"patch_radius": 0, "agg_func": 50},
        "tail_tip": {"patch_radius": 0, "agg_func": 50},
    })
    frames = np.full((2, 4, 4), 100, dtype=np.float32)
    frames[0, 0, 0] = 20       # valid edge patch for snout
    frames[0, 3, 3] = 40       # valid edge patch for tail_tip
    frames[1, 0, 0] = 255      # outside z_valid_range
    frames[1, 3, 3] = 60
    reader = FakeReader(frames)

    # overwriting methods from pkgs we didn't write
    monkeypatch.setattr(cc.vid.io, "AutoReader", lambda *a, **kw: reader)
    monkeypatch.setattr(cc.vid.util, "fill_holes", lambda frame: frame, raising=False)
    monkeypatch.setattr(cc.cv2, "bilateralFilter", lambda frame, **kw: frame)
    monkeypatch.setattr(cc, "tqdm", lambda values: values)

    kpoints = np.array([
        [[0, 0, .1], [3, 3, .2]],
        [[0, 0, .3], [np.nan, np.nan, .4]],
    ], dtype=np.float32)
    write_2d_artifacts(tmp_path, "cam0", kpoints, ["snout", "tail_tip"])

    # runs with monkeypatched methods and writes output to disk
    cc.get_3d_kpoints(
        str(avi), str(config), batch_size=1, z_valid_range=(1, 200),
        bilateral_kwargs={"d": 1, "sigmaColor": 1, "sigmaSpace": 1},
    )

    output = joblib.load(tmp_path / "_kpoints_v0_3d" / "cam0.pkl.gz")
    assert output.shape == (2, 2, 4)
    assert output.dtype == np.float32
    assert_allclose(output[0], [[0, 0, 20, .1], [3, 3, 40, .2]])
    assert_allclose(output[1, 1], [np.nan, np.nan, np.nan, .4], equal_nan=True)
    assert np.isnan(output[1, 0, 2])
    assert reader.ranges == [[0], [1]]
    assert reader.closed
    metadata = toml.load(tmp_path / "_kpoints_v0_3d" / "cam0.toml")
    assert metadata["z_valid_range"] == [1, 200]
    assert metadata["depth_patch_parameters"]["snout"]["patch_radius"] == 0


def test_get_3d_kpoints_calls_optional_spike_filter(tmp_path, monkeypatch):
    avi = tmp_path / "cam0.avi"
    avi.touch()
    config = write_config(tmp_path / "config.toml", depth_params={
        "snout": {"patch_radius": 0, "agg_func": 50},
    })
    reader = FakeReader(np.full((2, 2, 2), 10, dtype=np.float32))
    monkeypatch.setattr(cc.vid.io, "AutoReader", lambda *a, **kw: reader)
    monkeypatch.setattr(cc.vid.util, "fill_holes", lambda frame: frame, raising=False)
    monkeypatch.setattr(cc.cv2, "bilateralFilter", lambda frame, **kw: frame)
    monkeypatch.setattr(cc, "tqdm", lambda values: values)
    spike_filter = Mock(side_effect=lambda frame, **kwargs: frame)
    monkeypatch.setattr(cc, "replace_height_spikes", spike_filter)
    write_2d_artifacts(tmp_path, "cam0", np.array([[[0, 0, 1]], [[0, 0, 1]]]), ["snout"])
    cc.get_3d_kpoints(
        str(avi), str(config), replace_height_spikes_kwargs={"threshold": 5},
        bilateral_kwargs={"d": 1, "sigmaColor": 1, "sigmaSpace": 1},
    )
    assert spike_filter.call_count == 2
    assert all(call.kwargs == {"threshold": 5} for call in spike_filter.call_args_list)
    assert all(np.array_equal(call.args[0], np.full((2, 2), 10, dtype=np.float32))
               for call in spike_filter.call_args_list)


def test_get_3d_kpoints_skips_existing_output(tmp_path, monkeypatch):
    avi = tmp_path / "cam0.avi"
    avi.touch()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "cam0.pkl.gz").touch()
    config = write_config(tmp_path / "config.toml", depth_params={})
    monkeypatch.setattr(cc.vid.io, "AutoReader", Mock(side_effect=AssertionError("reader should not be opened"))) # TODO need to think of better handling, what if someone wants to regen (e.g. force=True)
    with pytest.warns(UserWarning, match="already computed"):
        assert cc.get_3d_kpoints(str(avi), str(config), save_dir="out") is None


def test_get_3d_kpoints_reports_missing_node_configuration(tmp_path, monkeypatch):
    avi = tmp_path / "cam0.avi"
    avi.touch()
    config = write_config(tmp_path / "config.toml", depth_params={"snout": {"patch_radius": 0, "agg_func": 50}})
    write_2d_artifacts(tmp_path, "cam0", np.zeros((1, 2, 3), dtype=np.float32), ["snout", "tail_tip"])
    monkeypatch.setattr(cc.vid.io, "AutoReader", lambda *a, **kw: FakeReader(np.ones((1, 2, 2))))
    with pytest.raises(KeyError, match="tail_tip"):
        cc.get_3d_kpoints(str(avi), str(config))


@pytest.mark.xfail(
    strict=True,
    reason="get_3d_kpoints does not close AutoReader when node validation raises after opening",
)
def test_get_3d_kpoints_closes_reader_when_validation_raises(tmp_path, monkeypatch):
    avi = tmp_path / "cam0.avi"
    avi.touch()
    config = write_config(
        tmp_path / "config.toml",
        depth_params={"snout": {"patch_radius": 0, "agg_func": 50}},
    )
    write_2d_artifacts(
        tmp_path,
        "cam0",
        np.zeros((1, 2, 3), dtype=np.float32),
        ["snout", "tail_tip"],
    )
    reader = FakeReader(np.ones((1, 2, 2), dtype=np.float32))
    monkeypatch.setattr(cc.vid.io, "AutoReader", lambda *a, **kw: reader)

    with pytest.raises(KeyError, match="tail_tip"):
        cc.get_3d_kpoints(str(avi), str(config))
    assert reader.closed


@pytest.mark.xfail(strict=True, reason="get_3d_kpoints returns on empty 2D data without closing AutoReader")
def test_get_3d_kpoints_closes_reader_for_empty_2d(tmp_path, monkeypatch):
    avi = tmp_path / "cam0.avi"
    avi.touch()
    config = write_config(tmp_path / "config.toml", depth_params={})
    write_2d_artifacts(tmp_path, "cam0", np.empty((0, 0, 3)), [])
    reader = FakeReader(np.ones((1, 2, 2)))
    monkeypatch.setattr(cc.vid.io, "AutoReader", lambda *a, **kw: reader)
    monkeypatch.setattr(cc, "tqdm", lambda values: values)
    cc.get_3d_kpoints(str(avi), str(config))
    assert reader.closed


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mismatched video/keypoint frame counts should process the shared minimum "
        "and save output shaped (min(n_video, n_keypoints), nodes, 4)"
    ),
)
def test_get_3d_kpoints_uses_shared_minimum_for_mismatched_frame_counts(tmp_path, monkeypatch):
    avi = tmp_path / "cam0.avi"
    avi.touch()
    config = write_config(
        tmp_path / "config.toml",
        depth_params={"snout": {"patch_radius": 0, "agg_func": 50}},
    )
    # The video has one frame more than the 2D predictions. The intended
    # contract is to process the shared minimum rather than index past kpoints.
    write_2d_artifacts(
        tmp_path,
        "cam0",
        np.array([[[0, 0, 1]]], dtype=np.float32),
        ["snout"],
    )
    reader = FakeReader(np.full((2, 2, 2), 10, dtype=np.float32))
    monkeypatch.setattr(cc.vid.io, "AutoReader", lambda *a, **kw: reader)
    monkeypatch.setattr(cc.vid.util, "fill_holes", lambda frame: frame, raising=False)
    monkeypatch.setattr(cc.cv2, "bilateralFilter", lambda frame, **kw: frame)
    monkeypatch.setattr(cc, "tqdm", lambda values: values)

    cc.get_3d_kpoints(
        str(avi),
        str(config),
        bilateral_kwargs={"d": 1, "sigmaColor": 1, "sigmaSpace": 1},
    )
    output = joblib.load(tmp_path / "_kpoints_v0_3d" / "cam0.pkl.gz")
    assert output.shape == (1, 1, 4)


def test_convert_2d_to_3d_creates_reordered_and_empty_sleap_arrays_and_uses_cache(tmp_path, monkeypatch):
    node_names = ["snout", "tail_tip"]
    config = write_config(tmp_path / "config.toml", depth_params={
        "snout": {"patch_radius": 0, "agg_func": 50}, "tail_tip": {"patch_radius": 1, "agg_func": 75},
    })
    root = tmp_path / "sleap"
    root.mkdir()
    avi_a = tmp_path / "cam_a.avi"
    avi_b = tmp_path / "cam_b.avi"
    avi_a.touch(); avi_b.touch()

    def point(name, x, y, score):
        return {"name": name, "xy": [x, y], "score": score}

    sleap_data = {
        "cam_a": SimpleNamespace(labeled_frames=[
            SimpleNamespace(instances=[SimpleNamespace(points=[point("tail_tip", 8, 9, .8), point("snout", 1, 2, .2)])]),
            SimpleNamespace(instances=[]),
        ])
    }
    loads = Mock(side_effect=lambda filename: sleap_data[Path(filename).stem])
    monkeypatch.setattr(cc.sio, "load_file", loads)
    cached = np.full((1, 2, 3), 7, dtype=np.float64)
    cached_dir = tmp_path / "_kpoints_v2_2d"
    cached_dir.mkdir()
    joblib.dump(cached, cached_dir / "cam_b.pkl.gz")

    delayed_calls = []
    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            delayed_calls.append((func, args, kwargs))
            return (args, kwargs)
        return wrapper

    parallel_calls = []
    class FakeParallel:
        def __init__(self, **kwargs):
            parallel_calls.append(("init", kwargs))
        def __call__(self, jobs):
            parallel_calls.append(("call", jobs))
            return jobs

    monkeypatch.setattr(cc.joblib, "delayed", fake_delayed)
    monkeypatch.setattr(cc.joblib, "Parallel", FakeParallel)
    monkeypatch.setattr(cc, "tqdm", lambda values: values)

    result = cc.convert_2d_to_3d(
        str(root), [str(avi_a), str(avi_b)], version_num=2, cable=True,
        node_names=node_names, config_path=str(config), conda_env_name="depth-env",
    )

    assert loads.call_count == 1
    array_a = joblib.load(tmp_path / "_kpoints_v2_2d" / "cam_a.pkl.gz")
    assert_allclose(array_a[0], [[1, 2, .2], [8, 9, .8]])
    assert np.isnan(array_a[1]).all()
    metadata = toml.load(tmp_path / "_kpoints_v2_2d" / "cam_a.toml")
    assert metadata["node_names"] == node_names
    assert metadata["node_mapping"] == {"snout": 0, "tail_tip": 1}
    assert len(delayed_calls) == 2
    assert len(result) == 2
    assert all(
        call[2]["bilateral_kwargs"] == {"d": 5, "sigmaColor": 6, "sigmaSpace": 7}
        for call in delayed_calls
    )
    assert delayed_calls[0][2]["replace_height_spikes_kwargs"] == {"threshold": 9, "ksize": 7}
    assert delayed_calls[0][2]["reader_kwargs"]["prepend_args"] == "source ~/conda_activate ; conda activate depth-env"
    assert parallel_calls[0][1]["n_jobs"] == -1
    assert parallel_calls[1][1] == result
