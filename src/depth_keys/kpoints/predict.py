"""
SLEAP-based pose estimation inference script.
Processes video files frame-by-frame with optional hole-filling preprocessing.
"""

from pathlib import Path

import torch
import warnings
from sleap_nn.predict import run_inference

# Configuration constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CENTERED_INSTANCE_MODEL_PATH = PROJECT_ROOT / "models" / "centered_instance"
CENTROID_MODEL_PATH = PROJECT_ROOT / "models" / "centroid_unet"

# MAX_INSTANCES = 1
# PEAK_THRESHOLD = 0.0
# BATCH_SIZE = 64

torch.set_default_dtype(torch.float32)

def run_inference_on_video(
    video_path: str,
    output_path: str,
    ci_model_path: str = None,
    centroid_model_path: str = None,
    batch_size: int = 16,
    max_instances: int = 1,
    peak_threshold: float = 0.0,
) -> bool:
    """
    Run SLEAP inference on a video file.
    
    Args:
        video_path: Path to input video.
        output_path: Path to save predictions (SLEAP .slp file).
        ci_model_path: Optional path to centered-instance model directory.
        centroid_model_path: Optional path to centroid model directory.
        
    Returns:
        True when inference runs successfully.
    """
    print(f"Processing video: {video_path}")

    if ci_model_path is None:
        ci_model_path = str(CENTERED_INSTANCE_MODEL_PATH)
        warnings.warn(f"No centered model specified, attempting to load from {ci_model_path}")
    
    if centroid_model_path is None:
        centroid_model_path = str(CENTROID_MODEL_PATH)
        warnings.warn(f"No centroid model specified, attempting to load from {centroid_model_path}")
    
    _predictions = run_inference(
        data_path=video_path,
        model_paths=[centroid_model_path, ci_model_path],
        output_path=output_path,
        make_labels=True,
        max_instances=max_instances,
        peak_threshold=peak_threshold,
        batch_size=batch_size,
    )

    
    return True
