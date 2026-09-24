"""Helpers for loading modules and projecting keypoints for visualization."""
import sys
import importlib.util
import re
import numpy as np
import toml
from pathlib import Path

def load_external_module(file_path, module_name="keypoint_processor"):
    """Load a Python module from a file and register it in ``sys.modules``.

    Args:
        file_path: Path to the Python source file.
        module_name: Name assigned to the loaded module.

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"External module not found at {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(path_obj))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def get_skeleton_edges(skeleton_def, node_names):
    """Map skeleton node-name pairs to index pairs.

    Edges referencing names absent from ``node_names`` are skipped with a
    warning printed to standard output.

    Args:
        skeleton_def: Iterable of pairs of node names.
        node_names: Ordered node names corresponding to keypoint indices.

    Returns:
        List of ``(start_index, end_index)`` edges.
    """
    edges = []
    for bone in skeleton_def:
        try:
            p1 = node_names.index(bone[0])
            p2 = node_names.index(bone[1])
            edges.append((p1, p2))
        except ValueError as e:
            print(f"Warning: Bone {bone} not found in node names. Skipping.")
    return edges

def inverse_project_3d_to_2d(points_3d, cx, cy, fx, fy):
    """Project 3D camera coordinates into 2D pixel coordinates.

    Zero depth values are replaced with ``1e-6`` for division. The output
    preserves every input dimension except the final coordinate dimension.

    Args:
        points_3d: Array ending in ``(x, y, z)`` coordinates.
        cx: Horizontal principal point in pixels.
        cy: Vertical principal point in pixels.
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.

    Returns:
        Array ending in ``(u, v)`` pixel coordinates.
    """
    # Assuming points are (X, Y, Z) and we project to (u, v)
    # u = fx * (x / z) + cx
    # v = fy * (y / z) + cy
    
    # Handle shaping
    original_shape = points_3d.shape
    flat_points = points_3d.reshape(-1, 3)
    
    x = flat_points[:, 0]
    y = flat_points[:, 1]
    z = flat_points[:, 2]
    
    # Avoid division by zero
    z_safe = np.where(z == 0, 1e-6, z)
    
    u = (fx * (x / z_safe)) + cx
    v = (fy * (y / z_safe)) + cy
    
    projected = np.stack([u, v], axis=-1)
    
    # Reshape back to (Frames, K, 2)
    return projected.reshape(original_shape[:-1] + (2,))
