"""
utils.py
Helper functions for loading modules, processing intrinsics, and handling paths.
"""
import sys
import importlib.util
import re
import numpy as np
import toml
from pathlib import Path

def load_external_module(file_path, module_name="keypoint_processor"):
    """
    Dynamically loads a python module from a specific file path.
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
    """
    Converts string-based skeleton definitions to index-based edges.
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
    """
    Projects 3D points back to 2D pixel coordinates.
    
    Args:
        points_3d: (N, 3) or (Frames, K, 3) array of X, Y, Z coordinates.
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