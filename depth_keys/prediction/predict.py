"""
SLEAP-based pose estimation inference script.
Processes video files frame-by-frame with optional hole-filling preprocessing.
"""

from pathlib import Path

import torch

from sleap_nn.predict import run_inference

# Configuration constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CENTERED_INSTANCE_MODEL_PATH = PROJECT_ROOT / "models" / "centered_instance"
CENTROID_MODEL_PATH = PROJECT_ROOT / "models" / "centroid_unet"

MAX_INSTANCES = 1
PEAK_THRESHOLD = 0.0
BATCH_SIZE = 64

torch.set_default_dtype(torch.float32)

def run_inference_on_video(
    video_path: str,
    output_path: str,
    ci_model_path: str = None,
    centroid_model_path: str = None
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
    
    if centroid_model_path is None:
        centroid_model_path = str(CENTROID_MODEL_PATH)

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
