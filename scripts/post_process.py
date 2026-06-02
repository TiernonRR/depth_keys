import sys
import os
from glob import glob

import argparse

# Get the path to the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to the parent directory (project root)
parent_dir = os.path.dirname(current_dir)

# Add parent directory to sys.path so Python can find 'depth_keys'
sys.path.append(parent_dir)

import depth_keys.experiment.trial as trial

node_names = [
    "tail_tip",
    "tail_middle",
    "tail_base",
    "back_bottom",
    "back_middle_lower",
    "back_middle_upper",
    "back_top",
    "left_ear",
    "right_ear",
    "snout",
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
]

centroid_model_path = "/storage/home/hcoda1/3/triesenmy3/lab_folder/sleap_nn_models/models/centroid_unet"
ci_model_path = '/storage/home/hcoda1/3/triesenmy3/lab_folder/sleap_nn_models/models/convnext-large_seed-4'

reference_camera = "Lucid Vision Labs-HTP003S-001-224500508"
intrinsics_file = "/storage/project/r-jmarkowitz30-0/shared/active_lab_members/markowitz_jeffrey/active_projects/mouse_open_field_lucid_rig_da_photometry/intrinsics_lucid_rig.toml" 


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SLEAP inference on video files with frame range support"
    )
    parser.add_argument('session', help="Session identifier")
    parser.add_argument('--project', required=True, help="Project name")
    parser.add_argument('--version', required=True, help="Output version identifier")
    parser.add_argument('--output_dir', required=False, default=None)
    # parser.add_argument('--camera', required=True, help="Output version identifier")
    
    # Mutually exclusive cable flags
    cable_group = parser.add_mutually_exclusive_group()
    cable_group.add_argument("--cable", dest="cable", action="store_true", help="Cable Present")
    cable_group.add_argument("--no-cable", dest="cable", action="store_false", help="Cable Not Present")
    parser.set_defaults(cable=True)
    
    return parser.parse_args()

def main():
    base_dir = "/storage/project/r-jmarkowitz30-0/shared/active_lab_members/markowitz_jeffrey/active_projects"

    args = parse_arguments()

    session = args.session
    project = args.project
    version = args.version
    cable   = args.cable
    output_dir = args.output_dir

    videos = glob(os.path.join(base_dir, project, session, "_proc", "*.avi"))
    print(videos)

    inference_output_path = os.path.join(base_dir, project, session, "_proc", f"_keypoints_v{version}")

    keypoint_output_path = os.path.join(base_dir, project, session, "_proc", output_dir) if output_dir else None

    # post_proc_config_path = "/storage/home/hcoda1/3/triesenmy3/r-jmarkowitz30-0/depth_and_da/Validation/param_sweep/configs/config_sweep_034.toml"# "./config.toml" TODO remove
    post_proc_config_path = "/storage/home/hcoda1/3/triesenmy3/r-jmarkowitz30-0/depth_and_da/depth_keys/example_scripts/config.toml"


    _trial = trial.Trial(trial_id=session, 
                         video_paths=videos, 
                         inference_output_path=inference_output_path,
                         version_num=version,
                         base_dir=os.path.join(base_dir, project),
                         intrinsics_file=intrinsics_file,
                         reference_camera=reference_camera,
                         node_names=node_names,
                         keypoints_output_path=keypoint_output_path, 
                         cable=cable,
                         conda_env_name="sleap-nn-env")

    _trial.compute_3d_keypoints(post_proc_config_path)

if __name__ == "__main__":
    main()

# TODO add code for viz
