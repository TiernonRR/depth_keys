"""
SLEAP-based pose estimation inference script.
Processes video files frame-by-frame with optional hole-filling preprocessing.
"""

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
BATCH_SIZE = 16 # TODO make these parameters

torch.set_default_dtype(torch.float32)

def combine_labels(predictions):
    """
    Combine a list of labels into single labels
    
    Returns
        Labels
    """

    labels = predictions[0]

    for i in range(1, len(predictions)):
        labels.merge(predictions[i])

    return labels


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


def run_inference_in_splits(
    video_path: str,
    n_splits: int,
    output_path: str,
    session: str,
    fill_gaps: bool = True,
    ci_model_path: str = None,
    centroid_model_path: str = None
) -> bool:
    """
    Run SLEAP inference on a video file by splitting the video into n_splits 
    to conserve memory.
    
    Args:
        video_path: path to input video
        n_splits: The number of chunks to split the video into for inference.
        output_path: path to save final merged predictions (SLEAP .slp file)
        fill_gaps: whether to apply hole-filling preprocessing
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Processing video: {video_path}")

    if ci_model_path is None:
        ci_model_path = CENTERED_INSTANCE_MODEL_PATH
    
    if centroid_model_path is None:
        centroid_model_path = CENTROID_MODEL_PATH
    
    # 1. Setup Video Reader and get total frames
    avi_reader = io.AviReader(
        video_path, 
        prepend_args="source ~/conda_activate ; conda activate ffmpeg"
    )
    avi_reader.get_file_info()
    total_frames = avi_reader.nframes

    if total_frames == 0:
        print("Error: Video contains 0 frames.")
        return False
        
    print(f"Total frames: {total_frames}. Splitting into {n_splits} chunks.")

    session_temp_dir = Path(TEMP_DIR) / session
    session_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate frames per split (using ceil to ensure all frames are covered)
    frames_per_split = int(np.ceil(total_frames / n_splits))
    all_predictions = []
    
    # Load the original video object once for reference correction
    try:
        sleap_video_orig = sio.load_video(video_path)
    except Exception as e:
        print(f"Error loading original video for reference: {e}")
        return False
        
    # 2. Iterate through splits and run inference
    for split_idx in range(n_splits):
        
        # Calculate the frame range for the current split
        frame_start = split_idx * frames_per_split
        frame_end = min((split_idx + 1) * frames_per_split - 1, total_frames - 1)
        
        # Break if the start frame is beyond the end of the video
        if frame_start > frame_end:
            break
            
        frame_indices = np.arange(frame_start, frame_end + 1)
        
        print(f"\n--- Running Split {split_idx + 1}/{n_splits} ---")
        print(f"Frames: [{frame_start}, {frame_end}] ({len(frame_indices)} frames)")
        
        temp_path = None

        try:
            # Load and preprocess frames for the current split
            frames = avi_reader.get_frames(frame_indices)
            frames = preprocess_frames(frames, fill_gaps=fill_gaps)
            
            # Create and save temporary video for the current split
            base_name = Path(video_path).stem
            temp_path = session_temp_dir / f"{base_name}_{frame_start}_{frame_end}.avi"
            temp_output_path = session_temp_dir / f"{base_name}_{frame_start}_{frame_end}.slp"
            create_temp_video(frames, str(temp_path))
            
            # Load temporary video object for SLEAP inference
            sleap_video_temp = sio.load_video(str(temp_path))

            # Run inference on the temporary split video
            print("Running inference on split...")
            split_predictions = run_inference(
                input_video=sleap_video_temp,
                model_paths=[centroid_model_path, ci_model_path],
                output_path=temp_output_path,  # Do not save inside the loop
                make_labels=True,
                max_instances=MAX_INSTANCES,
                peak_threshold=PEAK_THRESHOLD,
                batch_size=BATCH_SIZE
            )
            
            # 3. Fix video references and frame indices
            for i, frame_idx in enumerate(frame_indices):
                # Ensure the original video object and correct absolute frame index are used
                split_predictions[i].video = sleap_video_orig
                split_predictions[i].frame_idx = frame_idx
            
            split_predictions.videos.remove(sleap_video_temp)
            split_predictions.videos.append(sleap_video_orig)

            # Append predictions to the master list
            all_predictions.append(split_predictions)

        except Exception as e:
            print(f"Error during inference on split {split_idx + 1}: {e}")
            return False # Fail the entire run if a split fails
            
        finally:
            # Cleanup temporary files for the current split
            if temp_path and temp_path.exists():
                print(f"Cleaning up temporary file: {temp_path.name}")
                temp_path.unlink()

            # if temp_output_path and temp_output_path.exists():
            #     print(f"Cleaning up temporary file: {temp_output_path.name}")
            #     temp_output_path.unlink()

    # 4. Final step: Merge and save all predictions
    if not all_predictions:
        print("Error: No predictions were generated.")
        return False
        
    # Create a final Predictions object

    final_predictions = combine_labels(all_predictions) 
    
    final_predictions.save(output_path)
    print(f"\nAll {len(final_predictions)} predictions merged and saved to: {output_path}")
    
    return True