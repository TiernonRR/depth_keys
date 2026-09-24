import json
import os
import logging
import h5py
import toml
import depth_keys.visualization.utils as utils
import depth_keys.visualization.viz as viz
from typing import Any, Dict, List
# from depth_keys.post_processing.post_process import process_session, convert_2d_to_3d
from depth_keys.post_processing.conversion_computations import convert_2d_to_3d
from depth_keys.kpoints.predict import run_inference_on_video


class Trial:
    """Manage keypoint inference, 3D conversion, and visualization for one trial.

    A trial groups its camera videos with the paths and settings used by each
    processing stage. The stages run only when their corresponding methods are
    called.
    """

    def __init__(
        self,
        trial_id: str,  # associated with session
        video_paths: List[str],
        version_num: int = 1,
        base_dir: str = None,
        metadata: Dict[str, Any] = None,
        node_names: List = None,
        video_extension: str = ".avi",
        keypoints2d_output_path: str = None,
        keypoints3d_output_path: str = None,
        reference_camera: str = None,
        intrinsics_file: str = None,
        cable: bool = False,
        conda_env_name: str = None,
        transforms_path: str = None,
        verbose: bool = True,
        bundle_adjust: bool = False,
        logger = None,
        # registration_config_path: str = None,
    ):
        """Store the inputs and output locations for a trial.

        Args:
            trial_id: Trial identifier, joined to ``base_dir`` for 3D and
                visualization paths.
            video_paths: Paths to the camera videos to process.
            version_num: Version used in default keypoint output directory names.
            base_dir: Parent directory of the trial directory.
            metadata: Trial metadata. Defaults to an empty dictionary.
            node_names: Ordered keypoint names used during 2D-to-3D conversion.
            video_extension: Stored video filename extension.
            keypoints2d_output_path: Directory for SLEAP ``.slp`` predictions.
                If omitted, inference uses ``./_keypoints_v{version_num}_2d``.
            keypoints3d_output_path: Directory for converted and merged 3D
                keypoints. If omitted, conversion places it under the trial's
                ``_proc`` directory in subdir ``_kpoints_v{self.version_num}_3d``.
            reference_camera: Stored reference camera identifier.
            intrinsics_file: Path to the camera intrinsics TOML file.
            cable: Whether to use cable-specific depth processing settings.
            conda_env_name: Optional environment name passed to conversion and
                overlay rendering.
            transforms_path: Optional transforms passed to registration.
            verbose: Whether to log conversion and merging progress.
            bundle_adjust: Whether registration performs bundle adjustment.
            logger: Logger to use; defaults to this module's logger.
        """
        # Essential identifying information
        self.trial_id: str = trial_id
        # self.registration_config_path = registration_config_path

        # Associated files: List of strings pointing to video locations
        self.video_paths: List[str] = video_paths

        # Data fields
        # Dictionary for unstructured metadata (e.g., subject_id, environment_temp)
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

        self.video_extension = video_extension

        self.keypoints2d_output_path = keypoints2d_output_path
        self.keypoints3d_output_path = keypoints3d_output_path
        self.base_dir = base_dir
        self.version_num = version_num
        self.bundle_adjust = bundle_adjust

        self.reference_camera = reference_camera
        self.intrinsics_file = intrinsics_file
        self.cable = cable
        self.conda_env_name = conda_env_name
        self.node_names = [] if node_names is None else node_names

        self.transforms_path = transforms_path
        self.verbose = verbose
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
 

    def predict_keypoints(self, ci_model_path=None, centroid_model_path=None):
        """Write SLEAP predictions for each video in ``video_paths``.

        Creates ``keypoints2d_output_path`` if needed and writes one ``.slp``
        file per video, named after the video's filename stem.

        Args:
            ci_model_path: Centered-instance model directory. When omitted,
                the inference function uses its default model.
            centroid_model_path: Centroid model directory. When omitted, the
                inference function uses its default model.
        """

        if self.keypoints2d_output_path is None:
            self.keypoints2d_output_path = f"./_keypoints_v{self.version_num}_2d"

        os.makedirs(self.keypoints2d_output_path, exist_ok=True)

        for video_path in self.video_paths:

            save_name = os.path.splitext(os.path.basename(video_path))[0]

            _ = run_inference_on_video(
                video_path=video_path,
                output_path=os.path.join(
                    self.keypoints2d_output_path, f"{save_name}.slp"
                ),
                ci_model_path=ci_model_path,
                centroid_model_path=centroid_model_path,
            )

    def compute_3d_keypoints(self, registration_config_path):
        """Convert 2D predictions to depth keypoints and register camera views.

        Uses the trial's videos, node names, intrinsics, and conversion settings.
        If no 3D output directory was provided, sets it to
        ``base_dir/trial_id/_proc/_kpoints_v{version_num}_3d``. Registration
        writes merged keypoints to that directory.

        Args:
            registration_config_path: Path to the TOML configuration used for
                depth conversion and multiview registration.
        """
        from markovids.vid.io import format_intrinsics
        from markovids.pcl.pipeline import registration_pipeline

        use_data_dir = os.path.join(self.base_dir, self.trial_id)

        if self.keypoints3d_output_path is None:
            self.keypoints3d_output_path = os.path.join(
                self.base_dir, self.trial_id, "_proc", f"_kpoints_v{self.version_num}_3d"
            )

        for avi in self.video_paths:
            self.logger.info(f"-> {avi}")

        if self.verbose:
            self.logger.info("Converting 2D keypoints to 3D...")

        _ = convert_2d_to_3d(
            self.keypoints2d_output_path,
            self.video_paths,
            self.version_num,
            self.cable,
            self.node_names,
            registration_config_path,
            conda_env_name=self.conda_env_name,
        )

        if self.verbose:
            self.logger.info("Merging keypoints...")

        intrinsics_matrix, distortion_coeffs = format_intrinsics(
            toml.load(self.intrinsics_file)
        )
        registration_pipeline(
            registration_config_path,
            use_data_dir,
            kpoints_save_dir=os.path.basename(self.keypoints3d_output_path),
            intrinsics_matrix=intrinsics_matrix,
            distortion_coefficients=distortion_coeffs,
            alt_save_dir=self.keypoints3d_output_path,
            bundle_adjust=self.bundle_adjust,
            transforms_path=self.transforms_path,
        )

    def visualize(
        self,
        matplot_viz=True,
        overlay_viz=True,
        output_dir=None,
        filename="merged_keypoints.h5",
        max_frames_matplot=10000,
        matplot_save_name="matplotlib_render",
        skeleton_json_path="skeleton.json",
        alt_key_path=None,
        **overlay_kwargs,
    ):
        """Render merged 3D keypoints and optionally overlay them on video.

        Loads ``filename`` and ``merged_keypoints.toml`` from
        ``keypoints3d_output_path`` and reads the skeleton definition before
        starting either renderer. The Matplotlib renderer uses the
        ``merged_keypoints_smooth`` dataset; the overlay renderer receives
        ``alt_key_path`` when one is provided.

        Args:
            matplot_viz: Whether to render a 3D trajectory MP4.
            overlay_viz: Whether to render a keypoint overlay MP4.
            output_dir: Directory for rendered videos. Defaults to
                ``base_dir/trial_id/_proc/renders``.
            filename: Name of the merged keypoint HDF5 file to load.
            max_frames_matplot: Maximum frame index passed to the 3D renderer.
            matplot_save_name: Filename stem for the 3D trajectory MP4.
            skeleton_json_path: Path to the JSON skeleton edge definition.
            alt_key_path: Optional keypoint HDF5 file used by the overlay
                renderer instead of its version-based default.
            **overlay_kwargs: Additional arguments passed to the overlay
                processor, such as ``n_frames``, ``batch_size``, ``raw``,
                ``cam_by_conf``, ``frame_start``, ``frame_end``, and
                ``render_save_name``.

        Raises:
            ValueError: If ``keypoints3d_output_path`` is unset.
            FileNotFoundError: If the requested merged keypoint file is absent.
            KeyError: If the HDF5 file lacks ``merged_keypoints_smooth``.
        """
        if self.keypoints3d_output_path is None:
            raise ValueError(
                "keypoints3d_output_path must be set before visualization."
            )

        kpoints_path = self.keypoints3d_output_path
        h5_file = os.path.join(kpoints_path, filename)

        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"3D keypoint file not found at {h5_file}")

        if output_dir is None:
            output_dir = os.path.join(self.base_dir, self.trial_id, "_proc", "renders")

        self.logger.info(f"Visualizing keypoints from: {h5_file}")
        self.logger.info(f"Output target: {output_dir}")

        with h5py.File(h5_file, "r") as f:
            if "merged_keypoints_smooth" not in f:
                raise KeyError(
                    "Dataset 'merged_keypoints_smooth' not found in keypoint H5."
                )
            merged_keys = f["merged_keypoints_smooth"][()]

        toml_file = os.path.join(kpoints_path, "merged_keypoints.toml")

        kpoints_metadata = toml.load(toml_file)
        node_names = kpoints_metadata["kpoints"]["node_names"]
        reference_camera = kpoints_metadata["reference_camera"]

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
            self.logger.info("Visualization complete.")

        if overlay_viz:
            session_dir = os.path.join(self.base_dir, self.trial_id)
            viz.create_overlay_video(
                session_dir=session_dir,
                version_num=self.version_num,
                reference_camera=reference_camera,
                intrinsics_file=self.intrinsics_file,
                conda_env_name=self.conda_env_name,
                output_path=str(output_dir),
                keypoint_file=alt_key_path,
                **overlay_kwargs,
            )
