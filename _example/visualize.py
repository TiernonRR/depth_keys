import sys
import os
from glob import glob

import argparse

import sys
import os

# Get the path to the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to the parent directory (project root)
parent_dir = os.path.dirname(current_dir)

# Add parent directory to sys.path so Python can find 'depth_keys'
sys.path.append(parent_dir)
import depth_keys.experiment.trial as trial

reference_camera = "Lucid Vision Labs-HTP003S-001-224500508"
intrinsics_file = "/storage/project/r-jmarkowitz30-0/shared/active_lab_members/markowitz_jeffrey/active_projects/mouse_open_field_lucid_rig_da_photometry/intrinsics_lucid_rig.toml" 

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


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SLEAP inference on video files with frame range support"
    )
    parser.add_argument('session', help="Session identifier")
    parser.add_argument('--project', required=True, help="Project name")
    parser.add_argument('--version', required=True, help="Output version identifier")
    parser.add_argument('--keyp_output_dir', required=False, default=None)
    
    return parser.parse_args()

def main():
    base_dir = "/storage/project/r-jmarkowitz30-0/shared/active_lab_members/markowitz_jeffrey/active_projects"

    args = parse_arguments()

    session = args.session
    project = args.project
    version = args.version
    keyp_output_dir = args.keyp_output_dir
    cable = False # not needed for viz

    videos = glob(os.path.join(base_dir, project, session, "_proc", "*.avi")) # TODO if use camera arg

    inference_output_path = os.path.join(base_dir, project, session, "_proc", f"_keypoints_v{version}")

    base_kout = os.path.join(base_dir, project, session, "_proc")
    keypoints_output_path = os.path.join(base_kout, keyp_output_dir) if keyp_output_dir else os.path.join(base_kout, f"_kpoints_v{version}_3d")


    _trial = trial.Trial(trial_id=session, 
                         video_paths=videos, 
                         inference_output_path=inference_output_path,
                         keypoints_output_path=keypoints_output_path,
                         version_num=version,
                         base_dir=os.path.join(base_dir, project),
                         intrinsics_file=intrinsics_file,
                         reference_camera=reference_camera,
                         node_names=node_names,
                         cable=cable,
                         conda_env_name="sleap-nn-env")
    
    overlay_kwargs = {
        "n_frames": 5000,
        "render_save_name" : f"depth_overlay_{keyp_output_dir}"
    }

    _trial.visualize(matplot_save_name=f"matplotlib_render_{keyp_output_dir}", matplot_viz=True, overlay_viz=True, max_frames_matplot=5000, skeleton_json_path="./skeleton.json", **overlay_kwargs)

if __name__ == "__main__":
    main()
