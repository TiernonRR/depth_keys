"""Render 3D trajectories and camera-video keypoint overlays."""
import os
import numpy as np
import h5py
from markovids import pcl

from .overlay_processor import KeypointVideoProcessor

def create_overlay_video(session_dir, version_num, 
                         reference_camera,
                         intrinsics_file,
                         conda_env_name,
                         output_path,
                         keypoint_file=None,
                         **overlay_kwargs):
    """Create an overlay MP4 for one camera's depth video.

    Args:
        session_dir: Session directory containing ``_proc`` videos and keypoints.
        version_num: Version used to locate merged 3D keypoints by default.
        reference_camera: Camera whose video receives the overlay.
        intrinsics_file: Path to the camera intrinsics TOML file.
        conda_env_name: Environment name forwarded to the video processor.
        output_path: Destination directory for the MP4.
        keypoint_file: Optional merged keypoint HDF5 file overriding the
            version-based default.
        **overlay_kwargs: Additional processor options, such as ``n_frames``,
            ``batch_size``, ``raw``, ``render_save_name``, ``frame_start``,
            ``frame_end``, and ``cam_by_conf``.
    """


    video_processor = KeypointVideoProcessor(
        session_dir=session_dir,
        version_num=version_num,
        reference_camera=reference_camera,
        output_path=output_path,
        intrinsics_file=intrinsics_file,
        conda_env_name=conda_env_name,
        keypoint_file=keypoint_file,
        **overlay_kwargs
    )

    video_processor.process()


def render_3d_matplotlib(merged_keys, skeleton_edges, output_path, save_name, fps=100, 
                         burn_in=10, max_frames=None):
    """Render merged 3D keypoint trajectories to an MP4.

    Negates the Z coordinate for plotting and derives padded axis limits from
    the input. Renders frames from ``burn_in`` up to, but excluding,
    ``max_frames`` or the array length.

    Args:
        merged_keys: Keypoint array shaped ``(frames, nodes, 3)``.
        skeleton_edges: Pairs of node indices to connect in the render.
        output_path: Directory created for the output video if needed.
        save_name: Output filename stem; ``.mp4`` is appended.
        fps: Frame rate of the output video.
        burn_in: First frame index to render.
        max_frames: Exclusive upper frame bound; defaults to all frames.
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
