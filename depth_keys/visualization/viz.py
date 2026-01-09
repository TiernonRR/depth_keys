"""
viz.py
Visualization functions for 3D trajectories and 2D overlays.
"""
import os
import numpy as np
import h5py
from markovids import pcl

from .overlay_processor import KeypointVideoProcessor

def create_overlay_video(session_dir, version_num, 
                         reference_camera,
                         intrinsics_file,
                         **overlay_kwargs):
    """
    Initializes a KeypointVideoProcessor and generates the 2D overlay video.
    
    Parameters:
    -----------
    session_dir : str
        Path to the session directory.
    version_num : str
        The version number for the keypoints.
    keypoints_2d : np.ndarray
        The projected 2D keypoints (usually with depth as the 3rd channel) to overlay.
    reference_camera : str, optional
        The name of the camera to process.
    intrinsics_file : str, optional
        Path to the intrinsics TOML file.
    **kwargs : 
        Additional arguments passed to KeypointVideoProcessor 
        (e.g., n_frames, batch_size, raw, save_name, frame_start, frame_end, cam_by_conf, output_path).
    """
    
    # 1. Initialize the processor
    # We pass the explicit args and expand any remaining kwargs

    video_processor = KeypointVideoProcessor(
        session_dir=session_dir,
        version_num=version_num,
        reference_camera=reference_camera,
        intrinsics_file=intrinsics_file,
        **overlay_kwargs
    )

    video_processor.process()


def render_3d_matplotlib(merged_keys, skeleton_edges, output_path, save_name, fps=100, 
                         burn_in=10, max_frames=None):
    """
    Renders the 3D keypoints to an MP4 using matplotlib.
    """
    # Calculate limits with padding
    pad = 5
    x_min = np.nanmin(merged_keys[:,:,0]) - pad
    x_max = np.nanmax(merged_keys[:,:,0]) + pad
    y_min = np.nanmin(merged_keys[:,:,1]) - pad
    y_max = np.nanmax(merged_keys[:,:,1]) + pad
    
    # Invert Z for visualization logic (Camera Z usually points forward)
    z_vals = -1 * merged_keys[:,:,2]
    z_min = np.nanmin(z_vals) - pad
    z_max = np.nanmax(z_vals) + pad

    renderer_kwargs = {
        "trail_length": 5,
        "xlim": (x_min, x_max),
        "ylim": (y_min, y_max),
        "zlim": (z_min, z_max),
    }

    if max_frames is None:
        max_frames = len(merged_keys)

    frame_ids = range(burn_in, min(max_frames, len(merged_keys)))
    
    os.makedirs(output_path, exist_ok=True)
    movie_file = f"{save_name}.mp4"
    full_output_path = os.path.join(output_path, movie_file)

    # Prepare data for plotting
    plot_merged = merged_keys.copy()
    plot_merged[..., 2] = -1 * plot_merged[..., 2] 

    print(f"Rendering 3D visualization to {full_output_path}...")
    
    pcl.viz.visualize_xyz_trajectories_to_mp4(
            plot_merged,
            full_output_path,
            fps=fps,
            figsize=(16, 12),
            frame_ids=frame_ids,
            skeleton_edges=skeleton_edges,
            **renderer_kwargs,
    )