import json
import os
from typing import Any, Dict, List

import h5py
import toml

from depth_keys.post_processing.post_process import process_session
from depth_keys.kpoints.predict import run_inference_on_video
import depth_keys.visualization.utils as utils
import depth_keys.visualization.viz as viz


class Trial:
    """
    Represents a single run or observation within an experiment.
    Acts as interface with file paths needed for inference, post processing, and visualization.
    """
    def __init__(
        self,
        trial_id: str, # associated with session
        video_paths: List[str],  # Changed from List[Video] to List[str]
        version_num: int = 1,
        base_dir: str = None,
        metadata: Dict[str, Any] = None,
        node_names: List = None,
        video_extension: str = ".avi",
        inference_output_path: str = None,
        keypoints_output_path : str = None,
        viz_output_dir : str = None,
        reference_camera : str = None,
        intrinsics_file : str = None,
        cable : bool = False,
        conda_env_name: str = None,
        transforms_path : str = None
    ):
        # Essential identifying information
        self.trial_id: str = trial_id

        # Associated files: List of strings pointing to video locations
        self.video_paths: List[str] = video_paths

        # Data fields
        # Dictionary for unstructured metadata (e.g., subject_id, environment_temp)
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

        self.video_extension = video_extension

        self.inference_output_path = inference_output_path
        self.keypoints_output_path = keypoints_output_path
        self.viz_output_dir = viz_output_dir
        self.base_dir = base_dir
        self.version_num = version_num

        self.reference_camera = reference_camera
        self.intrinsics_file = intrinsics_file
        self.cable = cable
        self.conda_env_name = conda_env_name
        self.node_names = [] if node_names is None else node_names

        self.transforms_path = transforms_path

    def predict_keypoints(self, ci_model_path=None, centroid_model_path=None):
        """Runs inference on videos in video_paths"""        

        if self.inference_output_path is None:
            self.inference_output_path = f"./_keypoints_v{self.version_num}_2d"

        os.makedirs(self.inference_output_path, exist_ok=True)

        for video_path in self.video_paths:
            
            save_name = os.path.splitext(os.path.basename(video_path))[0]

            _ = run_inference_on_video(video_path=video_path, 
                                        output_path=os.path.join(self.inference_output_path, f"{save_name}.slp"),
                                        ci_model_path=ci_model_path,
                                        centroid_model_path=centroid_model_path)

    def compute_3d_keypoints(self, config_path):
        """Compute 3d keypoints from 2d SLEAP predictions."""


        avis = self.video_paths
        kpoint_root_dir = self.inference_output_path
        version_num=self.version_num
        use_data_dir = os.path.join(self.base_dir, self.trial_id)
        version_num = self.version_num
        intrinsics_file = self.intrinsics_file
        cable = self.cable
        node_names = self.node_names
        transforms_path = self.transforms_path
        

        if self.keypoints_output_path is None:
            self.keypoints_output_path = os.path.join(self.base_dir, self.trial_id, "_proc", f"_kpoints_v{version_num}_3d")
            
        save_dir = self.keypoints_output_path
        
        if self.conda_env_name is None:
            raise ValueError("conda_env_name is required for 3D keypoint computation.")

        process_session(
            config_path,
            use_data_dir,
            avis,
            kpoint_root_dir,
            intrinsics_file,
            version_num,
            node_names,
            cable,
            save_dir,
            conda_env_name=self.conda_env_name,
            transforms_path=transforms_path,
        )
        
    def visualize(self, matplot_viz=True, overlay_viz=True, output_dir=None, filename="merged_keypoints.h5", 
                  max_frames_matplot=10000, matplot_save_name="matplotlib_render", skeleton_json_path="skeleton.json", alt_key_path=None, **overlay_kwargs):
        """
        Visualizes the 3D keypoints saved in self.keypoints_output_path.
        
        Args:
            matplot_viz (bool): Flag to control whether to render the matplotlib-based visualization.
            overlay_viz (bool): Flag to control whether to render the depth video kepyoint overlay visualization.
            output_dir (str, optional): Directory to save the MP4. Defaults to self.base_dir/self.trial_id/renders.
            filename (str): The name of the H5 file to load (default: merged_keypoints.h5).
            max_frames_matplot (int): number of frames to render for the matplotlib-based render.
            overlay_kwargs (dict): arguments for depth video keypoint overlay: (`n_frames`, `batch_size`, `raw`,
                                   `cam_by_conf`, `frame_start`, `frame_end`, `render_save_name`)
        """
        if self.keypoints_output_path is None:
            raise ValueError("keypoints_output_path must be set before visualization.")

        kpoints_path = self.keypoints_output_path
        h5_file = os.path.join(kpoints_path, filename)

        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"3D keypoint file not found at {h5_file}")

        if output_dir is None:
            output_dir = os.path.join(self.base_dir, self.trial_id, "_proc", "renders")

        print(f"Visualizing keypoints from: {h5_file}")
        print(f"Output target: {output_dir}")

        with h5py.File(h5_file, "r") as f:
            if "merged_keypoints_smooth" not in f:
                raise KeyError("Dataset 'merged_keypoints_smooth' not found in keypoint H5.")
            merged_keys = f["merged_keypoints_smooth"][()]

        toml_file = os.path.join(kpoints_path, "merged_keypoints.toml")

        kpoints_metadata = toml.load(toml_file)
        node_names = kpoints_metadata["kpoints"]["node_names"]

        with open(skeleton_json_path, "r") as f:
            skeleton_definitions = json.load(f)

        skeleton_edges = utils.get_skeleton_edges(skeleton_definitions, node_names)

        if matplot_viz:
            viz.render_3d_matplotlib(
                merged_keys,
                skeleton_edges,
                output_path=str(output_dir),
                save_name=matplot_save_name,
                max_frames=max_frames_matplot,
                fps=100,
            )
            print("Visualization complete.")

        if overlay_viz:
            if self.conda_env_name is None:
                raise ValueError("conda_env_name is required for overlay visualization.")
            session_dir = os.path.join(self.base_dir, self.trial_id)
            viz.create_overlay_video(
                session_dir=session_dir,
                version_num=self.version_num,
                reference_camera=self.reference_camera,
                intrinsics_file=self.intrinsics_file,
                conda_env_name=self.conda_env_name,
                output_path=str(output_dir),
                keypoint_file=alt_key_path,
                **overlay_kwargs
            )
        

        
        
        



