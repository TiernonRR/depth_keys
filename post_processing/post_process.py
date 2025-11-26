import sys
sys.path.append("/storage/home/hcoda1/3/triesenmy3/r-jmarkowitz30-0/markovids/src")

from markovids import vid, pcl
from conversion_computations import *
import sys
import numpy as np
import toml


'''
    Global Variable Definitions
'''

smoothing_params = {
    "not_noisy": {"window_length": int(7), "poly_order": int(2)},
    "noisy": {"window_length": int(25), "poly_order": int(2)},
}

hampel_params = {
    "window": 100,
    "threshold": 3,
    "replace": False,
}


def process_session(use_data_dir, avis, kpoint_root_dir, intrinsics_file, version_num, node_names, cable, save_dir):
    print("Using the following avis: ")
    for avi in avis:
        print(f"-> {avi}")

    '''
        Convert 2d to 3d ...
    '''

    print("Converting 2D keypoints to 3D")

    _ = convert_2d_to_3d(intrinsics_file, 
                        kpoint_root_dir, 
                        avis, 
                        version_num, 
                        cable,
                        node_names)

    '''
        Merge keypoints across views...
    '''

    print("Merging keypoints...")


    intrinsics_matrix, distortion_coeffs = vid.io.format_intrinsics(toml.load(intrinsics_file))

    pcl.pipeline.registration_pipeline(
        use_data_dir,
        kpoints_save_dir=f"_kpoints_v{version_num}_3d",
        intrinsics_matrix=intrinsics_matrix,
        distortion_coefficients=distortion_coeffs,
        smoothing_params=smoothing_params,
        hampel_params=hampel_params,
        mp4_max_render_frames=5000,
        mp4_burn_in=200,
        mp4_renderer=None,
        cable=cable,
        constrain_bones=True,
        impute_pca=False,
        regularize_temporal=True,
        alt_save_dir=save_dir
    )
    

