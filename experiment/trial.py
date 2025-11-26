import numpy as np
from typing import Dict, Any, List
from sleap_io import Labels
from prediction import predict
import os

import sys
sys.path.append("/storage/home/hcoda1/3/triesenmy3/r-jmarkowitz30-0/markovids/src")

from markovids.vid import util, io

def get_video_length(video_path, prepend_args="source ~/conda_activate ; conda activate ffmpeg"):

    avi_reader = io.AviReader(
        video_path, 
        prepend_args=prepend_args
    )

    total_frames = avi_reader.nframes

    return total_frames


class Trial:
    """
    Represents a single run or observation within an experiment.
    Holds video file paths, raw keypoint data, and metadata.
    """
    def __init__(
        self,
        trial_id: str, # associated with session
        video_paths: List[str],  # Changed from List[Video] to List[str]
        sleap_data: Dict[str, Labels] = None, 
        n_body_parts: int = None,
        metadata: Dict[str, Any] = None,
        keypoints: np.ndarray = None,
        video_extension: str = ".avi",
        inference_output_path: str = None
    ):
        # Essential identifying information
        self.trial_id: str = trial_id

        # Associated files: List of strings pointing to video locations
        self.video_paths: List[str] = video_paths

        # Keypoint data configuration
        self.n_body_parts: int = n_body_parts

        # Data fields
        # Dictionary for unstructured metadata (e.g., subject_id, environment_temp)
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

        # Raw keypoint data
        # Shape: (n_frames, n_body_parts, 3) where the 3 dimensions are (x, y, confidence/z)
        self.keypoints: np.ndarray = keypoints if keypoints is not None else np.empty((0, self.n_body_parts, 3))

    def get_keypoint_shape(self) -> str:
        """Helper to check the shape of the keypoint data."""
        return str(self.keypoints.shape)

    def calculate_n_frames(self) -> int:
        """Returns the number of frames based on the keypoints array."""
        return self.keypoints.shape[0]
    
    def predict_keypoints(self, n_splits=4):
        """Runs inference on videos in video_paths"""

        if self.inference_output_path is None:
            self.inference_output_path = "../_keypoints_2d"

        os.makedirs(self.inference_output_path, exists_ok=True)

        for video_path in self.video_paths:
            _file_name = os.path.basename(video_path).rstrip(self.video_extension)

            # get video length
            video_length = get_video_length(_file_name)

        


            # predict TODO




