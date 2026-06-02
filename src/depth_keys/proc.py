from glob import glob
import os

def process_session(
    source_directory,
    config_path,
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
    compute_renders=True,
):
    import warnings
    from depth_keys.experiment.trial import Trial
    
    video_paths = sorted(glob(os.path.join(source_directory, glob_pattern)))
    inference_output_path = os.path.join(source_directory, "_proc", f"_key_points_v{version_num}")
    keypoints3d_output_path = os.path.join(source_directory, "_proc", f"_key_points_v{version_num}_3d")
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
        inference_output_path=inference_output_path,
        keypoints_output_path=keypoints3d_output_path,
        reference_camera=reference_camera,
        intrinsics_file=intrinsics_path,
        cable=cable,
        conda_env_name=None,
        transforms_path=transforms_path,
    )

    inference_exists = os.path.exists(inference_output_path)
    keypoints3d_exists = os.path.exists(keypoints3d_output_path)
    renders_exists = os.path.exists(renders_output_path)

    # process_session: 2D keypoint prediction
    if compute_2d and not inference_exists:
        trial.predict_keypoints(ci_model_path=ci_model_path, centroid_model_path=centroid_model_path)
    elif compute_2d and inference_exists:
        warnings.warn(f"Inference directory: {inference_output_path} already exists")
        return None
    # post_process: 2D -> 3D conversion
    if compute_3d and not keypoints3d_exists:
        trial.compute_3d_keypoints(config_path=config_path)
    elif compute_3d and keypoints3d_exists:
        warnings.warn(f"3D directory: {keypoints3d_output_path} already exists")
        return None
    
    # visualize: render keypoint overlay + 3D matplotlib video
    if compute_renders and not renders_exists:
        alt_key_path = os.path.join(trial.keypoints_output_path, "merged_keypoints.h5")
        trial.visualize(
            matplot_viz=True,
            overlay_viz=True,
            output_dir=renders_output_path,
            skeleton_json_path=skeleton_path,
            alt_key_path=alt_key_path,
        )
    elif compute_renders and renders_exists:
        warnings.warn(f"Renders directory: {renders_output_path} already exists")
        return None