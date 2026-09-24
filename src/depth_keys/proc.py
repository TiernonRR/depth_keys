from glob import glob
import os
import logging

# DEFINE DEFAULTS HERE
DEFAULT_OUTPUT_DIRS = {
    "kpoints_2d": "_kpoints_v{version}_2d",
    "kpoints_3d": "_kpoints_v{version}_3d",
    "renders": "renders"
}

def check_directory(
        source_directory,
        version_num: str = 1,
        output_dirs: dict = {},
):
    """Check whether the expected 2D, 3D, and render artifacts exist.

    Args:
        source_directory: Directory containing camera AVI files and outputs.
        version_num: Version embedded in keypoint and overlay filenames.
        output_dirs: Optional overrides for keypoint directory names.

    Returns:
        Mapping with Boolean ``2d``, ``3d``, and ``render`` completion flags.
        The 2D check accepts either ``.slp`` or ``.pkl.gz`` for each camera;
        the 3D check also requires ``merged_keypoints.h5``.
    """
    avis = glob(os.path.join(source_directory, "*.avi"))
    avis_base = [os.path.splitext(os.path.basename(_avi))[0] for _avi in avis]
    use_output_dirs = DEFAULT_OUTPUT_DIRS | output_dirs
    keypoints2d_output_path = os.path.join(source_directory, use_output_dirs["kpoints_2d"].format(version=version_num))
    keypoints3d_output_path = os.path.join(source_directory, use_output_dirs["kpoints_3d"].format(version=version_num))
    renders_output_path = os.path.join(source_directory, "renders")
    
    iscomplete = {}
    exists_2d_output = []
    for _avi in avis_base:
        output_file = os.path.join(keypoints2d_output_path, f"{_avi}.slp")
        output_file2 = os.path.join(keypoints2d_output_path, f"{_avi}.pkl.gz")
        exists_2d_output.append(os.path.exists(output_file) | os.path.exists(output_file2))

    # print(exists_2d_output)
    iscomplete["2d"] = all(exists_2d_output)
     
    exists_3d_output = []
    for _avi in avis_base:
        output_file = os.path.join(keypoints3d_output_path, f"{_avi}.pkl.gz")
        exists_3d_output.append(os.path.exists(output_file))
    exists_3d_output.append(os.path.exists(os.path.join(keypoints3d_output_path, "merged_keypoints.h5")))
    
    iscomplete["3d"] = all(exists_3d_output)
    # isok["2d"] = not os.path.exists(keypoints2d_output_path)
    # isok["3d"] = not os.path.exists(keypoints3d_output_path)
    # isok["render"] = not os.path.exists(renders_output_path)

    renders_files = [f"keypoints_overlay_v{version_num}.mp4", "matplotlib_render.mp4"]
    exists_renders_output = [os.path.exists(os.path.join(renders_output_path, _file)) for _file in renders_files]
    iscomplete["render"] = all(exists_renders_output)    
    return iscomplete


# TODO:
# 1. more verbose logging of all parameters...
# 2. Try/Catch when force=False for existing dirs
def process_directory(
    source_directory,
    registration_config_path,
    ci_model_path,
    centroid_model_path,
    intrinsics_path,
    transforms_path,
    skeleton_path,
    node_names,
    glob_pattern="_proc/*.avi",
    version_num=1,
    reference_camera="",
    cable=False,
    compute_2d=True,
    compute_3d=True,
    render=True,
    force=False,
    output_dirs = {}
):
    """Run selected keypoint and visualization stages for a session directory.

    Finds camera AVI files with ``glob_pattern``, constructs a ``Trial``, and
    runs inference, 3D registration, and rendering according to the stage flags.

    Args:
        source_directory: Session directory containing the ``_proc`` folder.
        registration_config_path: TOML settings for depth conversion and registration.
        ci_model_path: Centered-instance model directory.
        centroid_model_path: Centroid model directory.
        intrinsics_path: Camera intrinsics TOML file.
        transforms_path: Optional transforms for registration.
        skeleton_path: JSON skeleton definition used by rendering.
        node_names: Ordered names of keypoints in the SLEAP predictions.
        glob_pattern: Video path pattern relative to ``source_directory``.
        version_num: Version embedded in keypoint output directory names.
        reference_camera: Reference camera identifier stored on the trial.
        cable: Whether to use cable-specific depth settings.
        compute_2d: Whether to run SLEAP inference.
        compute_3d: Whether to convert and register keypoints.
        render: Whether to create both visualization videos.
        force: Whether existing output directories may be reused.
        output_dirs: Optional overrides for keypoint directory names.
    """
    import warnings
    from depth_keys.experiment.trial import Trial
    logger = logging.getLogger(__name__)

    use_output_dirs = DEFAULT_OUTPUT_DIRS | output_dirs
    
    # do we want these hardcoded?
    video_paths = sorted(glob(os.path.join(source_directory, glob_pattern)))
    keypoints2d_output_path = os.path.join(source_directory, "_proc", use_output_dirs["kpoints_2d"].format(version=version_num))
    keypoints3d_output_path = os.path.join(source_directory, "_proc", use_output_dirs["kpoints_3d"].format(version=version_num))
    renders_output_path = os.path.join(source_directory, "_proc", "renders")
    
    if len(video_paths) > 0:
        logger.info(f"Processing videos in {source_directory}: {video_paths}")

    trial = Trial(
        trial_id=source_directory,
        video_paths=video_paths,
        version_num=version_num,
        base_dir=os.path.dirname(source_directory),
        node_names=node_names,
        video_extension=".avi",
        keypoints2d_output_path=keypoints2d_output_path,
        keypoints3d_output_path=keypoints3d_output_path,
        reference_camera=reference_camera,
        intrinsics_file=intrinsics_path,
        cable=cable,
        conda_env_name=None,
        transforms_path=transforms_path,
        # registration_config_path=registration_config_path,
    )

    # process_session: 2D keypoint prediction
    if compute_2d:
        os.makedirs(keypoints2d_output_path, exist_ok=force)
        trial.predict_keypoints(ci_model_path=ci_model_path, centroid_model_path=centroid_model_path)
    
    # post_process: 2D -> 3D conversion
    if compute_3d:
        os.makedirs(keypoints3d_output_path, exist_ok=force) 
        trial.compute_3d_keypoints(registration_config_path=registration_config_path)
    
    # visualize: render keypoint overlay + 3D matplotlib video
    if render:
        alt_key_path = os.path.join(keypoints3d_output_path, "merged_keypoints.h5")
        trial.visualize(
            matplot_viz=True,
            overlay_viz=True,
            output_dir=renders_output_path, # do we want custome output render dir
            skeleton_json_path=skeleton_path,
            alt_key_path=alt_key_path,
        )
