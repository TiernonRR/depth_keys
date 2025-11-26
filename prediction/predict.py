"""
SLEAP-based pose estimation inference script.
Processes video files frame-by-frame with optional hole-filling preprocessing.
"""
from experiment import experiment
from trial import trial

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import sleap_io as sio
import torch

# Add custom module path
sys.path.append("/storage/home/hcoda1/3/triesenmy3/r-jmarkowitz30-0/markovids/src")
from markovids.vid import util, io

from sleap_nn.predict import run_inference

# Configuration constants
CENTERED_INSTANCE_MODEL_PATH = "../models/centered_instance"
CENTROID_MODEL_PATH = "../models/centroid_unet"
TEMP_DIR = "../temporary_space"

FILL_KERNEL_SIZE = (10, 10)
FILL_ITERATIONS = 4
MAX_INSTANCES = 1
PEAK_THRESHOLD = 0.0
BATCH_SIZE = 2

torch.set_default_dtype(torch.float32)


def validate_frame_range(frame_start, frame_end, total_frames):
    """
    Validate and adjust frame range based on video length.
    
    Returns:
        tuple: (adjusted_start, adjusted_end, is_valid)
    """
    if frame_start >= total_frames:
        return frame_start, frame_end, False
    
    # Handle -1 as "end of video"
    actual_end = total_frames - 1 if frame_end == -1 else frame_end
    
    # Clamp to valid range
    actual_end = min(actual_end, total_frames - 1)
    
    return frame_start, actual_end, True


def preprocess_frames(frames, fill_gaps=True):
    """
    Apply hole-filling preprocessing to frames.
    
    Args:
        frames: numpy array of shape (n_frames, height, width, channels)
        fill_gaps: whether to apply hole filling
        
    Returns:
        Preprocessed frames array
    """
    if not fill_gaps:
        return frames
    
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, FILL_KERNEL_SIZE)
    processed = frames.copy()
    
    for idx in range(len(processed)):
        processed[idx] = util.fill_holes(
            processed[idx], 
            fill_kernel=fill_kernel, 
            iterations=FILL_ITERATIONS
        )
    
    return processed


def create_temp_video(frames, output_path):
    """
    Save frames to temporary video file.
    
    Args:
        frames: numpy array of frames
        output_path: path to save video
    """
    # Ensure frames have channel dimension
    if frames.ndim == 3:
        frames = np.expand_dims(frames, axis=-1)
    
    sio.save_video(frames, output_path)


def run_inference_on_video(video_path, frame_range, output_path, fill_gaps=True):
    """
    Run SLEAP inference on a video file for specified frame range.
    
    Args:
        video_path: path to input video
        model_path: path to trained model
        frame_range: numpy array of frame indices
        output_path: path to save predictions
        fill_gaps: whether to apply hole-filling preprocessing
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Processing video: {video_path}")
    
    # Load video and validate frame range
    avi_reader = io.AviReader(
        video_path, 
        prepend_args="source ~/conda_activate ; conda activate ffmpeg"
    )
    avi_reader.get_file_info()
    total_frames = avi_reader.nframes
    
    frame_start, frame_end, is_valid = validate_frame_range(
        frame_range[0], frame_range[-1], total_frames
    )
    
    if not is_valid:
        print(f"Error: frame_range_start ({frame_start}) exceeds video length ({total_frames})")
        return False
    
    # Update frame range if adjusted
    if frame_end != frame_range[-1]:
        print(f"Warning: Adjusted frame_range_end from {frame_range[-1]} to {frame_end}")
        frame_range = np.arange(frame_start, frame_end + 1)
    
    print(f"Processing {len(frame_range)} frames: [{frame_start}, {frame_end}]")
    
    # Load and preprocess frames
    frames = avi_reader.get_frames(frame_range)
    frames = preprocess_frames(frames, fill_gaps=fill_gaps)
    
    # Create temporary video
    base_name = Path(video_path).stem
    temp_path = Path(TEMP_DIR) / f"{base_name}_{frame_start}_{frame_end}.avi"
    create_temp_video(frames, str(temp_path))
    
    try:
        # Load videos
        sleap_video_temp = sio.load_video(str(temp_path))
        sleap_video_orig = sio.load_video(video_path)
        
        # Run inference
        print("Running inference...")
        predictions = run_inference(
            input_video=sleap_video_temp,
            model_paths=[CENTROID_MODEL_PATH, CENTERED_INSTANCE_MODEL_PATH],
            output_path=output_path,
            make_labels=True,
            max_instances=MAX_INSTANCES,
            peak_threshold=PEAK_THRESHOLD,
            batch_size=BATCH_SIZE
        )
        
        # Fix video references in predictions
        for i, frame_idx in enumerate(frame_range):
            predictions[i].video = sleap_video_orig
            predictions[i].frame_idx = frame_idx
        
        predictions.videos.remove(sleap_video_temp)
        predictions.videos.append(sleap_video_orig)
        
        # Save predictions
        predictions.save(output_path)
        print(f"✓ Predictions saved to: {output_path}")
        
        return True
        
    finally:
        # Cleanup temporary video
        if temp_path.exists():
            temp_path.unlink()


# Example usages
# def main():
#     """Main execution function."""
#     args = parse_arguments()
    
#     # Setup paths
#     base_input_dir = Path(ACTIVE_PROJECTS_DIR) / args.project / args.session / "_proc"
#     output_dir = base_input_dir / f"_keypoints_v{args.version}"
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    
#     # Find all video files
#     videos = glob(str(base_input_dir / "*.avi"))
    
#     if not videos:
#         print(f"No videos found in {base_input_dir}")
#         return
    
#     print(f"Found {len(videos)} video(s) to process")
    
#     # Create frame range
#     frame_range = np.arange(args.frame_range_start, args.frame_range_end + 1)
#     fill_gaps = not args.no_fill_gaps
    
#     # Process each video
#     for video_path in videos:
#         camera = Path(video_path).stem
#         camera_output_dir = output_dir / camera
#         camera_output_dir.mkdir(parents=True, exist_ok=True)
        
#         output_path = camera_output_dir / f"{camera}_{args.frame_range_start}-{args.frame_range_end}.slp"
        
#         success = run_inference_on_video(
#             video_path=video_path,
#             model_path=args.model_dir,
#             frame_range=frame_range,
#             output_path=str(output_path),
#             fill_gaps=fill_gaps
#         )
        
#         print("-" * 50)
    
#     print("Processing complete!")


# if __name__ == "__main__":
#     main()