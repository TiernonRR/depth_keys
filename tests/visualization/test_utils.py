import importlib
import sys

import numpy as np
import pytest

from depth_keys.visualization import utils


def test_load_external_module_success_and_registers_name(tmp_path):
    path = tmp_path / "external.py"
    path.write_text("VALUE = 42\n")

    module = utils.load_external_module(path, module_name="test_external_module")

    assert module.VALUE == 42
    assert sys.modules["test_external_module"] is module


def test_load_external_module_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="External module not found"):
        utils.load_external_module(tmp_path / "missing.py")


def test_load_external_module_propagates_execution_error(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("raise RuntimeError('module failed')\n")

    with pytest.raises(RuntimeError, match="module failed"):
        utils.load_external_module(path, module_name="broken_external_module")


def test_get_skeleton_edges_maps_names_to_indices(capsys):
    edges = utils.get_skeleton_edges(
        [("snout", "back_top"), ("back_top", "tail_tip")], ["back_top", "snout", "tail_tip"]
    )

    assert edges == [(1, 0), (0, 2)]
    assert capsys.readouterr().out == ""


def test_get_skeleton_edges_skips_invalid_bones_and_warns(capsys):
    edges = utils.get_skeleton_edges(
        [("snout", "missing"), ("back_top", "tail_tip")], ["back_top", "snout", "tail_tip"]
    )

    assert edges == [(0, 2)]
    assert "Bone ('snout', 'missing') not found" in capsys.readouterr().out


@pytest.mark.parametrize(
    "points, expected_shape",
    [
        (np.array([[2.0, 4.0, 2.0]]), (1, 2)),
        (np.ones((2, 3, 3), dtype=float), (2, 3, 2)),
    ],
)
def test_inverse_project_3d_to_2d_known_projection_and_shapes(points, expected_shape):
    projected = utils.inverse_project_3d_to_2d(points, cx=10, cy=20, fx=4, fy=5)

    expected = np.array([[14.0, 30.0]]) if points.ndim == 2 else np.full((2, 3, 2), [14.0, 25.0])
    np.testing.assert_allclose(projected, expected)
    assert projected.shape == expected_shape


def test_inverse_project_3d_to_2d_zero_depth_uses_small_positive_depth():
    points = np.array([[2.0, -3.0, 0.0]])

    result = utils.inverse_project_3d_to_2d(points, cx=1, cy=2, fx=4, fy=5)

    np.testing.assert_allclose(result, [[8_000_001.0, -14_999_998.0]])


def test_inverse_project_3d_to_2d_preserves_nan_and_does_not_mutate():
    points = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 2.0]])
    original = points.copy()

    result = utils.inverse_project_3d_to_2d(points, cx=0, cy=0, fx=1, fy=1)

    assert np.isnan(result[0]).all()
    np.testing.assert_allclose(result[1], [1.5, 2.0])
    np.testing.assert_array_equal(points, original)
