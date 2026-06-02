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

FILL_KERNEL_SIZE = (10, 10)
FILL_ITERATIONS = 4
MAX_INSTANCES = 1
PEAK_THRESHOLD = 0.0
BATCH_SIZE = 64 # TODO make these parameters

torch.set_default_dtype(torch.float32)

def run_inference_on_video(
    video_path: str,
    output_path: str,
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

    _predictions = run_inference(
        data_path=video_path,
        model_paths=[centroid_model_path, ci_model_path],
        output_path=output_path,
        make_labels=True,
        max_instances=MAX_INSTANCES,
        peak_threshold=PEAK_THRESHOLD,
        batch_size=BATCH_SIZE
    )

    
    return True