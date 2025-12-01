import numpy as np
from typing import Dict, Any, List
from sleap_io import Labels
import h5py
import toml
from depth_keys.prediction.predict import run_inference_on_video
from depth_keys.post_processing.post_process import process_session
import depth_keys.visualization.config as config
import depth_keys.visualization.utils as utils
import depth_keys.visualization.viz as viz
import os
from glob import glob

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
        n_body_parts: int = 14,
        version_num: int = 1,
        base_dir: str = None,
        metadata: Dict[str, Any] = None,
        keypoints: np.ndarray = None,
        video_extension: str = ".avi",
        inference_output_path: str = None,
        keypoints_output_path : str = None,
        viz_output_dir : str = None,
        reference_camera : str = None,
        intrinsics_file : str = None
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

        self.sleap_data = sleap_data
        self.video_extension = video_extension

        self.inference_output_path = inference_output_path
        self.keypoints_output_path = keypoints_output_path
        self.viz_output_dir = viz_output_dir
        self.base_dir = base_dir
        self.version_num = version_num

        self.reference_camera = reference_camera
        self.intrinsics_file = intrinsics_file

    def get_keypoint_shape(self) -> str:
        """Helper to check the shape of the keypoint data."""
        return str(self.keypoints.shape)

    def calculate_n_frames(self) -> int:
        """Returns the number of frames based on the keypoints array."""
        return self.keypoints.shape[0]
    
    # def predict_keypoints(self, version_num=0, n_splits=4, 
    #                       ci_model_path=None, centroid_model_path=None):
    #     """Runs inference on videos in video_paths"""        

    #     if self.inference_output_path is None:
    #         self.inference_output_path = f"./_keypoints_v{version_num}_2d"

    #     os.makedirs(self.inference_output_path, exist_ok=True)

    #     for video_path in self.video_paths:
            
    #         save_name = os.path.basename(video_path).rstrip(self.video_extension)

    #         # predict
    #         success = run_inference_in_splits(video_path=video_path, 
    #                                           n_splits=n_splits, 
    #                                           session = self.trial_id,
    #                                           output_path=os.path.join(self.inference_output_path, f"{save_name}.slp"),
    #                                           ci_model_path=ci_model_path,
    #                                           centroid_model_path=centroid_model_path)

    #         if not success:
    #             print(f"Unable to process {os.path.basename(video_path)}")

    def predict_keypoints(self, version_num=0, n_splits=4, 
                          ci_model_path=None, centroid_model_path=None):
        """Runs inference on videos in video_paths"""        

        if self.inference_output_path is None:
            self.inference_output_path = f"./_keypoints_v{version_num}_2d"

        os.makedirs(self.inference_output_path, exist_ok=True)

        for video_path in self.video_paths:
            
            save_name = os.path.basename(video_path).rstrip(self.video_extension)

            # predict
            _ = run_inference_on_video(video_path=video_path, 
                                              output_path=os.path.join(self.inference_output_path, f"{save_name}.slp"),
                                              ci_model_path=ci_model_path,
                                              centroid_model_path=centroid_model_path)

    def compute_3d_keypoints(self, use_data_dir, version_num, intrinsics_file, cable, node_names, save_dir):

        if self.keypoints_output_path is None:
            self.keypoints_output_path = f"./_keypoints_v{version_num}_3d"

        avis = self.video_paths
        kpoint_root_dir = self.inference_output_path
        
        
        process_session(use_data_dir, 
                        avis, 
                        kpoint_root_dir, 
                        intrinsics_file, 
                        version_num, 
                        node_names, 
                        cable, 
                        save_dir)
        
    def visualize(self, save_name, output_dir=None, filename="merged_keypoints.h5", max_frames=None, **overlay_kwargs):
        """
        Visualizes the 3D keypoints saved in self.keypoints_output_path.
        
        Args:
            output_dir (str, optional): Directory to save the MP4. Defaults to {keypoints_output_path}/renders.
            filename (str): The name of the H5 file to load (default: merged_keypoints.h5).
        """
        if self.keypoints_output_path is None:
            print("Error: keypoints_output_path is not set. Run compute_3d_keypoints first.")
            return

        kpoints_path = self.keypoints_output_path
        h5_file = os.path.join(kpoints_path, filename)

        if not os.path.exists(h5_file):
            print(f"Error: 3D keypoint file not found at {h5_file}")
            return

        # Set default output directory
        if output_dir is None:
            output_dir = os.path.join(self.base_dir, self.trial_id, "_proc", "renders")

        print(f"Visualizing keypoints from: {h5_file}")
        print(f"Output target: {output_dir}")

        # 1. Load the Merged Keypoints
        try:
            with h5py.File(h5_file, "r") as f:
                # Load merged_keypoints_smooth preferably, fallback if needed
                merged_keys = f["merged_keypoints_smooth"][()]

        except Exception as e:
            print(f"Error loading H5 file: {e}")
            return

        # 2. Retrieve Node Names (needed for Skeleton)
        # We try to find any .toml file in the directory (usually cam1.toml etc saved by process_session)
        # to get the node name list.

        toml_files = glob(os.path.join(kpoints_path, "*.toml"))
        if not toml_files:
            print("Error: No metadata .toml files found to retrieve node names.")
            return
        
        try:
            kpoints_metadata = toml.load(toml_files[0])
            node_names = kpoints_metadata["node_names"]
        except Exception as e:
            print(f"Error loading metadata from {toml_files[0]}: {e}")
            return

        # 3. Build Skeleton Edges
        skeleton_edges = utils.get_skeleton_edges(config.SKELETON_DEFINITIONS, node_names)

        # 4. Render Video
        try:
            viz.render_3d_matplotlib(
                merged_keys,
                skeleton_edges,
                output_path=str(output_dir),
                save_name=save_name,
                max_frames=max_frames,
                fps=config.FPS
            )
            print("Visualization complete.")
        except Exception as e:
            print(f"Error during visualization rendering: {e}")

        # 5. Keypoint Overlay
        session_dir = os.path.join(self.base_dir, self.trial_id)
        try:
            viz.create_overlay_video(
                session_dir=session_dir,
                version_num=self.version_num,
                reference_camera=self.reference_camera,
                intrinsics_file=self.intrinsics_file,
                **overlay_kwargs
            )
            print("Visualization complete.")
        except Exception as e:
            print(f"Error during visualization rendering: {e}")
        

        
        
        






