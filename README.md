# Python package for the depth rig processing pipeline

This repository provides a depth-video processing pipeline for:

- 2D keypoint prediction (`depth_keys/prediction`)
- 2D-to-3D conversion and session post-processing (`depth_keys/post_processing`)
- visualization and overlay rendering (`depth_keys/visualization`)

## Environment requirements

- Python 3.12+
- `ffmpeg` CLI available through a conda environment used by subprocess calls
- Package dependencies in `pyproject.toml`
- environment.yml provided

## Configuration

Post-processing depth extraction settings are config-driven. The example configs now include:

- `[post_processing.depth_patch_parameters]` per-node patch radius and aggregation percentile
- `[post_processing.depth_processing]` non-cable/height-spike settings
- `[post_processing.depth_processing_cable]` cable/height-spike settings

See:

- `notebooks/config.toml`
- `notebooks/config_cable.toml`

## Assumptions

Assumes that the session directory is of form:

*session_name*
 / - metadata.toml
   - *cam1*.txt
   - *cam2*.txt
   - *cam3*.txt
   - _bground
    / - *cam1*.toml
      - *cam2*.toml
      - *cam3*.toml
      - *cam1*.tiff
      - *cam2*.tiff
      - *cam3*.tiff
   - _proc
    / - timestamps.txt
      - sync_metadata.toml
      - *cam1*.avi
      - *cam2*.avi
      - *cam3*.avi

Assumes that ffmpeg is installed in environment via conda.

Assumes the presence of 

## command sequence 
Refer to _example directory
 - (optional) fill_holes.py -> This script iterates over each frame of the input video and outputs (I output to scratch) a video with any holes in the depth map filled in. 
 - process_session.py -> Obtains SLEAP predictions. Provide session name, project name, version number, camera name, and flag for whether a photometry cable was used in the experiment. Creates output at the project directory under _proc in _keypoints_v{version}
 - post_process.py -> Computes z from the 2d predictions and performs post processing stages. Provide session name, project name, version number, cable flag, and output directory.
 - visualize.py -> This renders keypoint overlay video and 3d matplotlib render of keypoints. Provide session name, project name, version number, and output directory.

notebooks dir contains 2026-05-08-run_pipeline.ipynb which demonstrates how to run the processing code on an example session.