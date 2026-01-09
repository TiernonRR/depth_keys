import sys # TODO replace when pip
sys.path.append("/storage/home/hcoda1/3/triesenmy3/r-jmarkowitz30-0/markovids/src")

from markovids import vid, pcl
from depth_keys.post_processing.conversion_computations import *
import sys
import numpy as np
import toml

def process_session(
    config_path,
    use_data_dir, 
    avis,
    kpoint_root_dir, 
    intrinsics_file, 
    version_num, 
    node_names, 
    cable, 
    save_dir, 
    bundle_adjust=False):
    
    print("Using the following avis: ")
    for avi in avis:
        print(f"-> {avi}")

    '''
        Convert 2d to 3d ...
    '''

    print("Converting 2D keypoints to 3D")

    _ = convert_2d_to_3d(kpoint_root_dir, 
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
        config_path,
        use_data_dir,
        kpoints_save_dir=f"_kpoints_v{version_num}_3d",
        intrinsics_matrix=intrinsics_matrix,
        distortion_coefficients=distortion_coeffs,
        alt_save_dir=save_dir,
        bundle_adjust=bundle_adjust
    )
    

