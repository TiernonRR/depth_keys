"""
Single-Camera 3D Keypoint Video Processor

This script processes a single camera video with 3D keypoint data and creates
a video with overlaid keypoints colored by depth (z-value).
Processes frames in batches to manage memory usage.
"""

import argparse
import os
import subprocess
from typing import Tuple

import cv2
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize
from tqdm import tqdm

import h5py
import toml
import tifffile

import sys
sys.path.append("/storage/home/hcoda1/3/triesenmy3/r-jmarkowitz30-0/markovids/src")
from markovids import depth
from markovids.vid.io import AviReader, format_intrinsics

# Global variables moved to class properties below

import numpy as np

def inverse_project_world_coordinates(
    xyz, z_scale=1.0, floor_distance=None, cx=319.0, cy=231.0, fx=525.0, fy=525.0
):
    """
    Inverse of project_world_coordinates function.
    Converts from world coordinates (x, y, z) back to image coordinates with depth (u, v, z).
    
    Parameters:
    - xyz: numpy array of shape (N, 3) containing world coordinates (x, y, z)
    - z_scale: scaling factor for z coordinate
    - floor_distance: distance to floor (if used in original function)
    - cx, cy: principal point of the camera
    - fx, fy: focal lengths in x and y directions
    
    Returns:
    - uvz: numpy array of shape (N, 3) containing image coordinates with depth (u, v, z)
    """
    # Extract world coordinates
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    
    # Calculate scaled_z for output depth
    scaled_z = z * z_scale
    
    # Calculate project_z for coordinate transformation
    if floor_distance is not None:
        project_z = (floor_distance - z) / z_scale
    else:
        project_z = z / z_scale
    
    # Prevent division by zero
    project_z = np.maximum(project_z, 1e-10)
    
    # Calculate image coordinates
    u = ((x * fx) / project_z) + cx
    v = ((y * fy) / project_z) + cy
    
    # Stack the results - output depth is scaled_z as you noted
    uvz = np.hstack([u[:, None], v[:, None], scaled_z[:, None]])
    
    return uvz

# --- Custom MP4 Writer Class ---
class MP4Writer:
    """
    A custom video writer that uses ffmpeg to write frames directly to an MP4 file.
    """
    def __init__(self, filepath: str, frame_size: Tuple[int, int], fps: int, prepend_args: str = None):
        if os.path.splitext(filepath)[1] != ".mp4":
            raise ValueError("Filepath must have an .mp4 extension")
        
        self.filepath = filepath
        self.frame_size = frame_size  # (width, height)
        self.fps = fps
        self.prepend_args = prepend_args
        self.pipe = None

    def open(self):
        """Opens the ffmpeg subprocess pipe."""
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-loglevel', 'error',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.frame_size[0]}x{self.frame_size[1]}',  # WxH
            '-pix_fmt', 'bgr24',  # Input pixel format from OpenCV
            '-r', str(self.fps),
            '-i', '-',  # Input from stdin
            '-an',  # No audio
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',  # Standard pixel format for compatibility
            '-preset', 'medium',
            '-crf', '22',
            f'"{self.filepath}"'  # Quoted output file path
        ]
        
        full_cmd = " ".join(command)
        if self.prepend_args:
            full_cmd = f"{self.prepend_args} ; {full_cmd}"

        self.pipe = subprocess.Popen(full_cmd, shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash')

    def write_frames(self, frames: np.ndarray, progress_bar: bool = True):
        """Writes a batch of frames to the video."""
        if self.pipe is None:
            self.open()
        
        iterable = tqdm(frames) if progress_bar else frames
        for frame in iterable:
            try:
                self.pipe.stdin.write(frame.astype(np.uint8).tobytes())
            except BrokenPipeError:
                print("Error: Broken pipe. ffmpeg may have closed unexpectedly.")
                stderr = self.pipe.stderr.read().decode()
                if stderr:
                    print(f"ffmpeg error output:\n{stderr}")
                # Re-raise the exception to stop the process
                raise

    def close(self):
        """Closes the ffmpeg pipe."""
        if self.pipe and self.pipe.stdin:
            self.pipe.stdin.close()
            self.pipe.wait()
            stderr = self.pipe.stderr.read().decode()
            if stderr:
                print(f"ffmpeg error output on close:\n{stderr}")
        self.pipe = None

class KeypointVideoProcessor:
    """Processes a single camera video with 3D keypoint overlays."""
    
    def __init__(self, session_dir: str, version_num: str, 
                 reference_camera: str = "Lucid Vision Labs-HTP003S-001-224500508",
                 intrinsics_file: str = "/storage/project/r-jmarkowitz30-0/shared/active_lab_members/markowitz_jeffrey/active_projects/mouse_open_field_lucid_rig_da_photometry/intrinsics_lucid_rig.toml",
                 n_frames: int = None, batch_size: int = 500, raw: bool = False, overlay_save_name: str = None,
                 frame_start: int = None, frame_end: int = None, cam_by_conf: bool = False, output_path: str = None):
        
        self.session_dir = session_dir
        self.video_dir = os.path.join(session_dir, "_proc")
        self.version_num = version_num
        self.output_dir = os.path.join(session_dir, "_proc", "renders") if output_path is None else output_path
        self.batch_size = batch_size
        self.raw = raw
        self.save_name = overlay_save_name
        self.frame_start = frame_start
        self.frame_end = frame_end
        
        # Set camera and intrinsics as instance properties
        self.reference_camera = reference_camera
        self.intrinsics_file = intrinsics_file

        # Load in metadata
        metadata = toml.load(os.path.join(session_dir, "_proc", f"_kpoints_v{version_num}_3d", "merged_keypoints.toml"))

        # Video processing parameters
        self.keypoint_radius = 3
        self.colormap = cm.jet
        self.fps = 100
        self.prepend_args = "source ~/conda_activate ; conda activate ffmpeg"
        self.cam_by_conf = cam_by_conf
        self.cameras = metadata['cameras']
        self.conf = None
        
        # Will be set dynamically based on video length
        self.n_frames = n_frames
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_keypoints(self):
        """Load 3D keypoint data for the camera."""
        keypoint_file = os.path.join(self.session_dir, "_proc", f"_kpoints_v{self.version_num}_3d", "merged_keypoints.h5")
        
        print(f"Loading keypoints from: {keypoint_file}")
        
        if self.raw:
            with h5py.File(keypoint_file, "r") as f:
                _arr = f["merged_keypoints_raw"][()]
                if self.cam_by_conf:
                    self.conf = f["proj_point_conf"][()]
        
        else:
            with h5py.File(keypoint_file, "r") as f:
                _arr = f["merged_keypoints_smooth"][()]
                if self.cam_by_conf:
                    self.conf = f["proj_point_conf"][()]

        
        # Load camera intrinsics using self.intrinsics_file
        intrinsics_matrix, _ = format_intrinsics(toml.load(self.intrinsics_file))
        
        # Access intrinsics using self.reference_camera
        cx = intrinsics_matrix[self.reference_camera][0, 2]
        cy = intrinsics_matrix[self.reference_camera][1, 2]
        fx = intrinsics_matrix[self.reference_camera][0, 0]
        fy = intrinsics_matrix[self.reference_camera][1, 1]

        # Get floor distance from background depth image using self.reference_camera
        bground_file = os.path.join(self.session_dir, "_bground", f"{self.reference_camera}.tiff")
        bground = tifffile.imread(bground_file)

        bground_roi = depth.plane.get_floor(bground.astype("float"), dilations=0)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (60, 60)
        )  # erode walls, etc.
        # use_bground_roi = cv2.erode(
        #     bground_roi, kernel
        # ) 

        # floor_distance = np.median(bground[use_bground_roi]) / 4.0

        # Convert world coordinates to camera image coordinates
        keypoints = inverse_project_world_coordinates(
            _arr.reshape(-1,3), 
            z_scale=1.0, 
            floor_distance=None,
            cx=cx, 
            cy=cy, 
            fx=fx, 
            fy=fy
        ).reshape(_arr.shape)
            
        print(f"Loaded {keypoints.shape[0]} frames of keypoint data")
        return keypoints
    
    def get_video_path(self):
        """Get path to the video file."""
        # Use self.reference_camera
        video_path = os.path.join(self.session_dir, "_proc", f"{self.reference_camera}.avi")
        if os.path.exists(video_path):
            print(f"Found video file: {video_path}")
            return video_path
        else:
            print(f"Warning: Video file not found: {video_path}")
            return None
    
    def determine_video_length(self, video_path, keypoints):
        """Determine the minimum video length between video and keypoints."""
        if video_path is None:
            return 0
            
        # Check video file length
        avi_reader = AviReader(video_path, prepend_args=self.prepend_args)
        avi_reader.get_file_info()
        video_length = avi_reader.nframes
        print(f"Video has {video_length} frames")
        
        # Check keypoint data length
        keypoint_frames = keypoints.shape[0]
        print(f"Keypoint data has {keypoint_frames} frames")
        
        # Set to minimum of both or user-specified value
        min_frames = min(video_length, keypoint_frames)
        self.n_frames = int(min_frames) if self.n_frames is None else min(int(min_frames), self.n_frames)
        print(f"Using {self.n_frames} frames")
        return self.n_frames
    
    def load_video_batch(self, video_path, start_frame, end_frame):
        """Load a batch of video frames."""
        print(f"Loading frames {start_frame} to {end_frame}...")
        
        # Load specific frame range
        avi_reader = AviReader(video_path, prepend_args=self.prepend_args)
        avi_reader.get_file_info()

        frames = avi_reader.get_frames(list(range(start_frame, min(end_frame, avi_reader.nframes))))
        
        # Get frame size
        if frames is not None and frames.size > 0:
            # AviReader returns (width, height), but we want (height, width) for OpenCV
            frame_size = (avi_reader.frame_size[1], avi_reader.frame_size[0])
            print(f"Loaded {len(frames)} frames of size {frame_size}")
            return frames, frame_size
        else:
            print(f"Warning: No frames loaded")
            return None, None
    
    def calculate_z_range(self, keypoints):
        """Calculate min and max z-values for consistent coloring."""
        # Ensure we don't go out of bounds
        max_frames = min(self.n_frames, keypoints.shape[0])
        z_values = keypoints[:max_frames, :, 2]
        valid_z_values = z_values[~np.isnan(z_values)]
        
        if len(valid_z_values) > 0:
            min_z = np.min(valid_z_values)
            max_z = np.max(valid_z_values)
            print(f"Z-range: {min_z:.2f} to {max_z:.2f}")
            return min_z, max_z
        else:
            print("Warning: No valid Z values found")
            return 0, 1  # Default range
    
    def draw_keypoints_on_frame(self, frame, frame_keypoints, normalizer, conf=None):
        """Draw colored keypoints on a single frame."""
        # Convert grayscale to BGR if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Make a copy to avoid modifying the original
        frame_with_keypoints = frame.copy()
        height, width = frame_with_keypoints.shape[:2]
        
        # Draw each keypoint
        for keypoint in frame_keypoints:
            # Make sure keypoint has at least 3 values
            if len(keypoint) < 3:
                continue
            
            x, y, z = keypoint[:3]
            
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue
                
            x_int, y_int = int(round(x)), int(round(y))
            
            if x_int < 0 or x_int >= width or y_int < 0 or y_int >= height:
                continue
            
            color_rgba = self.colormap(normalizer(z))
            color_bgr = tuple(int(c * 255) for c in color_rgba[:3][::-1])
            
            cv2.circle(frame_with_keypoints, (x_int, y_int), self.keypoint_radius, color_bgr, -1)
        
        # Add camera name label
        if self.cam_by_conf is None or conf is None:
            # Use self.reference_camera for the label
            cv2.putText(frame_with_keypoints, self.reference_camera, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # calculate majority highest conf
            _camIndex = np.nanargmax(np.nanmean(conf, axis=1))
            _cam = self.cameras[_camIndex]
            cv2.putText(frame_with_keypoints, _cam, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        
        return frame_with_keypoints
    
    def add_colorbar_to_frame(self, frame, min_z, max_z):
        """Add a vertical colorbar to the right of the frame."""
        height, width = frame.shape[:2]
        
        # Colorbar dimensions and position
        colorbar_width = 20
        colorbar_height = height // 2  # Half the frame height
        colorbar_x = width - colorbar_width - 30  # Position near right edge
        colorbar_y = (height - colorbar_height) // 2  # Center vertically
        
        # Create the vertical gradient (high values at the top)
        colorbar_img = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
        for i in range(colorbar_height):
            normalized_val = (colorbar_height - 1 - i) / (colorbar_height - 1)
            color_rgba = self.colormap(normalized_val)
            color_bgr = tuple(int(c * 255) for c in color_rgba[:3][::-1])
            colorbar_img[i, :] = color_bgr
        
        # Place the colorbar onto the frame
        frame[colorbar_y:colorbar_y + colorbar_height, colorbar_x:colorbar_x + colorbar_width] = colorbar_img
        
        # Add min/max labels
        cv2.putText(frame, f"{max_z:.2f}", (colorbar_x + colorbar_width + 5, colorbar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"{min_z:.2f}", (colorbar_x + colorbar_width + 5, colorbar_y + colorbar_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def process_frame_batch(self, frames, keypoints, start_idx, end_idx, min_z, max_z, video_writer):
        """Process a batch of frames and write to video."""
        if frames is None or len(frames) == 0:
            print(f"Warning: No frames to process in batch {start_idx}-{end_idx}")
            return
        
        normalizer = Normalize(vmin=min_z, vmax=max_z)
        batch_frames = []
        
        for rel_frame_idx, frame in enumerate(frames):
            frame_idx = start_idx + rel_frame_idx
            
            # Skip if frame index is out of bounds of keypoint data
            if frame_idx >= keypoints.shape[0]:
                print(f"Warning: Frame index {frame_idx} exceeds keypoint data length {keypoints.shape[0]}")
                continue
                
            # Get keypoints for this frame
            frame_keypoints = keypoints[frame_idx]
            
            # Draw keypoints on the frame
            _conf = None if self.conf is None else self.conf[:,frame_idx]
            frame_with_keypoints = self.draw_keypoints_on_frame(frame, frame_keypoints, normalizer, _conf)
            
            # Add colorbar and frame counter
            frame_with_keypoints = self.add_colorbar_to_frame(frame_with_keypoints, min_z, max_z)
            cv2.putText(frame_with_keypoints, f"Frame: {frame_idx}", (10, frame_with_keypoints.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            batch_frames.append(frame_with_keypoints)
        
        # Write the batch to video
        if batch_frames:
            video_writer.write_frames(np.array(batch_frames), progress_bar=False)
            print(f"Wrote {len(batch_frames)} frames (batch {start_idx}-{start_idx+len(batch_frames)-1})")
    
    def create_video(self, keypoints, min_z, max_z, frame_size):
        """Create the video with keypoint overlays."""
        print("Creating video with keypoint overlays...")
        
        video_path = self.get_video_path()
        if video_path is None:
            print("Error: Could not find video file. Aborting.")
            return
        
        if self.save_name:
            output_path = os.path.join(self.output_dir, f"{self.save_name}.mp4")
        else:
            if self.raw:
                output_path = os.path.join(self.output_dir, f"keypoints_overlay_v{self.version_num}-raw.mp4")
            else:
                output_path = os.path.join(self.output_dir, f"keypoints_overlay_v{self.version_num}.mp4")
        print(f"Output video will be saved to: {output_path}")
        
        # Use the actual frame size from the video
        writer = MP4Writer(
            output_path,
            frame_size=(frame_size[1], frame_size[0]),  # (width, height) for ffmpeg
            fps=self.fps,
            prepend_args=self.prepend_args
        )
        
        # try:
        writer.open()

        if self.frame_start is None:
            for start_idx in range(0, self.n_frames, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.n_frames)
                print(f"Processing batch {start_idx}-{end_idx} of {self.n_frames} frames...")
                
                # Load batch of frames
                frames, _ = self.load_video_batch(video_path, start_idx, end_idx)
                
                # Process batch
                self.process_frame_batch(
                    frames, keypoints, start_idx, end_idx, min_z, max_z, writer
                )
        else:
            for start_idx in range(self.frame_start, self.frame_end, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.n_frames)
                print(f"Processing batch {start_idx}-{end_idx} of {self.frame_end - self.frame_start} frames...")
                
                # Load batch of frames
                frames, _ = self.load_video_batch(video_path, start_idx, end_idx)
                
                # Process batch
                self.process_frame_batch(
                    frames, keypoints, start_idx, end_idx, min_z, max_z, writer
                )
            
        print(f"\nMP4 video saved successfully to: {output_path}")
            
        # except Exception as e:
        #     print(f"\nAn error occurred while creating the video: {e}")
        #     import traceback
        #     traceback.print_exc()
        # finally:
        #     print("Closing video writer...")
        #     writer.close()
        writer.close()
    
    def process(self):
        """Main processing pipeline."""
        print(f"Processing session: {os.path.basename(self.session_dir)}")
        
        # Load keypoints
        keypoints = self.load_keypoints()
        
        # Get video path
        video_path = self.get_video_path()
        if video_path is None:
            print("Error: Could not find video file. Aborting.")
            return
        
        # Determine video length
        self.determine_video_length(video_path, keypoints)
        
        # Calculate z-range for colormap
        min_z, max_z = self.calculate_z_range(keypoints)
        
        # Load a single frame to get dimensions
        first_frames, frame_size = self.load_video_batch(video_path, 0, 1)
        if frame_size is None:
            print("Error: Could not determine frame size. Aborting.")
            return
        
        # Create video with keypoint overlays
        self.create_video(keypoints, min_z, max_z, frame_size)
        
        print("Processing complete!")