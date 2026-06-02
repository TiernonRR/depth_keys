"""
run_render.py
Main script to process merged keypoints and generate visualizations.
"""
import sys
import os
import argparse
import toml
import h5py
import numpy as np
from pathlib import Path

# Add project path for markovids
sys.path.append("/storage/home/hcoda1/3/triesenmy3/r-jmarkowitz30-0/markovids/src")
from markovids.vid.io import format_intrinsics

# Import local modules
import config
import utils
import viz

# Path to the external Viz script
VIZ_SCRIPT_PATH = '/storage/home/hcoda1/3/triesenmy3/r-jmarkowitz30-0/depth_and_da/analysis/EvaluatePipeline/2025-09-07-Make2dViz-fromMerged.py'
INTRINSICS_FILE = "/storage/project/r-jmarkowitz30-0/shared/active_lab_members/markowitz_jeffrey/active_projects/mouse_open_field_lucid_rig_da_photometry/intrinsics_lucid_rig.toml"
REF_CAMERA = "Lucid Vision Labs-HTP003S-001-224500508"

def main(args):
    # 1. Setup Paths
    base_dir = Path(args.base_dir)
    session_dir = base_dir / args.project / args.session
    proc_dir = session_dir / "_proc"
    kpoints_dir = proc_dir / "_kpoints_v1.1_3d"
    merged_keys_path = kpoints_dir / "merged_keypoints.h5"
    output_dir = proc_dir / "renders"

    if not merged_keys_path.exists():
        print(f"Error: Merged keypoints not found at {merged_keys_path}")
        sys.exit(1)

    print(f"Processing session: {args.session}")

    # 2. Load Intrinsics
    print("Loading intrinsics...")
    intrinsics_data = toml.load(INTRINSICS_FILE)
    intrinsics_matrix, _ = format_intrinsics(intrinsics_data)
    
    cam_matrix = intrinsics_matrix[REF_CAMERA]
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
    
    # 3. Load Keypoints Data
    print("Loading 3D keypoints...")
    with h5py.File(merged_keys_path, "r") as f:
        merged_keys = f["merged_keypoints_smooth"][()]

    # 4. Load Metadata for Skeleton
    # Try to find a camera toml file to get node names
    cameras = list(intrinsics_matrix.keys())
    meta_path = kpoints_dir / f"{cameras[0]}.toml"
    
    if not meta_path.exists():
        print(f"Error: Metadata file not found at {meta_path}")
        sys.exit(1)
        
    kpoints_metadata = toml.load(meta_path)
    node_names = kpoints_metadata["node_names"]
    
    # 5. Generate Skeleton Edges
    skeleton_edges = utils.get_skeleton_edges(config.SKELETON_DEFINITIONS, node_names)

    # 6. Render 3D Matplotlib Video
    viz.render_3d_matplotlib(
        merged_keys, 
        skeleton_edges, 
        output_path=str(output_dir),
        fps=config.FPS
    )

    # 7. Optional: 2D Overlay (Commented out logic from original script cleaned up)
    if args.run_overlay:
        print("Generating 2D Overlay...")
        
        # Project 3D -> 2D
        keypoints_2d = utils.inverse_project_3d_to_2d(
            merged_keys, cx, cy, fx, fy
        )
        
        # Load external processor
        keypoint_processor_mod = utils.load_external_module(VIZ_SCRIPT_PATH)
        
        # Initialize the processor class
        # Note: logic inferred from your snippet
        processor_instance = keypoint_processor_mod.KeypointVideoProcessor(
            use_data_dir=str(proc_dir.parent), # pass session root
            keyp_version=1,
            frame_start=0,
            frame_end=8000,
            batch_size=1000,
            raw=False,
            save_name="merged_keys_v1",
            output_path=str(output_dir)
        )
        
        viz.create_overlay_video(keypoint_processor_mod, processor_instance, keypoints_2d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 3D and 2D visualizations from merged keypoints.")
    
    parser.add_argument("--base_dir", type=str, 
                        default="/storage/project/r-jmarkowitz30-0/shared/active_lab_members/markowitz_jeffrey/active_projects/",
                        help="Base directory for projects")
    parser.add_argument("--project", type=str, default="mouse_open_field_lucid_rig_da_bipoles",
                        help="Project folder name")
    parser.add_argument("--session", type=str, required=True,
                        help="Session folder name (e.g., 'session_20250612...')")
    parser.add_argument("--run_overlay", action="store_true", 
                        help="Run the 2D overlay generation (requires the external module script)")

    args = parser.parse_args()
    
    main(args)