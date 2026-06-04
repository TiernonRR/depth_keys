from glob import glob
import os

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
    use_output_dirs = DEFAULT_OUTPUT_DIRS | output_dirs
    keypoints2d_output_path = os.path.join(source_directory, "_proc", use_output_dirs["kpoints_2d"].format(version=version_num))
    keypoints3d_output_path = os.path.join(source_directory, "_proc", use_output_dirs["kpoints_3d"].format(version=version_num))
    renders_output_path = os.path.join(source_directory, "_proc", "renders")
    
    isok = {}
    isok["2d"] = not os.path.exists(keypoints2d_output_path)
    isok["3d"] = not os.path.exists(keypoints3d_output_path)
    isok["render"] = not os.path.exists(renders_output_path)

    return isok


# TODO:
# 1. more verbose logging of all parameters...
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
    import warnings
    from depth_keys.experiment.trial import Trial

    use_output_dirs = DEFAULT_OUTPUT_DIRS | output_dirs
    
    # do we want these hardcoded?
    video_paths = sorted(glob(os.path.join(source_directory, glob_pattern)))
    keypoints2d_output_path = os.path.join(source_directory, "_proc", use_output_dirs["kpoints_2d"].format(version=version_num))
    keypoints3d_output_path = os.path.join(source_directory, "_proc", use_output_dirs["kpoints_3d"].format(version=version_num))
    renders_output_path = os.path.join(source_directory, "_proc", "renders")
    
    if len(video_paths) > 0:
        print(f"Processing videos in {source_directory}: {video_paths}")

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
        alt_key_path = os.path.join(trial.keypoints_output_path, "merged_keypoints.h5")
        trial.visualize(
            matplot_viz=True,
            overlay_viz=True,
            output_dir=renders_output_path,
            skeleton_json_path=skeleton_path,
            alt_key_path=alt_key_path,
        )