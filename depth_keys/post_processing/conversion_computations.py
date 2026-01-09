import sleap_io as sio
import os
import sys
import joblib
import numpy as np
import warnings
import cv2
import copy 
import toml

import sys
sys.path.append("/storage/home/hcoda1/3/triesenmy3/r-jmarkowitz30-0/markovids/src")
from markovids import vid

from tqdm.auto import tqdm

def cable_agg_func_with_cable(x):
    return np.nanpercentile(x, 75)

def cable_agg_func_no_cable(x):
    return np.nanmax(x)

smoothing_params = {
    "not_noisy": {"window_length": int(7), "poly_order": int(2)},
    "noisy": {"window_length": int(25), "poly_order": int(2)},
}

hampel_params = {
    "window": 100,
    "threshold": 3,
    "replace": False,
}

def replace_height_spikes(depth_map, threshold=30, ksize=5, z_scale=4):
    """
    Use OpenCV medianBlur to replace height spikes above a threshold with local median.
    Args:
        depth_map: 2D array of height values in mm (float32 or float64)
        threshold: max allowed height difference from local median
        ksize: kernel size for median blur (must be odd)
    Returns:
        filtered 2D height map
    """
    temp = depth_map.copy()
    original_dtype = depth_map.dtype

    
    if ksize > 5:
        median = (
            cv2.medianBlur((temp / z_scale).astype("uint8"), ksize=ksize).astype(
                original_dtype
            )
            * z_scale
        )
    else:
        median = cv2.medianBlur(temp, ksize=ksize)
    diff = np.abs(temp - median)
    replace_mask = diff > threshold
    temp[replace_mask] = median[replace_mask]
    return temp

def get_3d_kpoints(
    avi_file,
    batch_size=2000,
    kpoint_2d_save_dir="_kpoints_v0_2d",
    save_dir="_kpoints_v0_3d",
    patch_radius=3,
    agg_func=np.nanmax,
    z_valid_range = (1,200),
    reader_kwargs={"threads": 2, 
                   "prepend_args" : "source ~/conda_activate ; conda activate ffmpeg"},
    bilateral_kwargs={"d":5, "sigmaColor": 15, "sigmaSpace":3},
    replace_height_spikes_kwargs=None,
    force=False,
    new_save_dir = None
):

    avi_dir = os.path.dirname(avi_file)
    cam = os.path.splitext(os.path.basename(avi_file))[0]
    
    if new_save_dir is None:
        new_save_dir = os.path.join(avi_dir, save_dir)
        
    new_save_file = os.path.join(new_save_dir, f"{cam}.pkl.gz")
    
    new_metadata_file = os.path.join(new_save_dir, f"{cam}.toml")
    
    if os.path.exists(new_save_file) and not force:
        print(f"{new_save_file} exists, skipping...")
        return None
    
    os.makedirs(new_save_dir, exist_ok=True)

    read_obj = vid.io.AutoReader(avi_file, **reader_kwargs)
    nframes = read_obj.nframes
    width, height = read_obj.frame_size

    kpoint_file = os.path.join(avi_dir, kpoint_2d_save_dir, f"{cam}.pkl.gz")
        
    metadata_file = os.path.join(avi_dir, kpoint_2d_save_dir, f"{cam}.toml")
    
    metadata = toml.load(metadata_file)
    new_metadata = copy.deepcopy(metadata)
    new_metadata["agg_func"] = agg_func.__name__
    new_metadata["patch_radius"] = patch_radius
    new_metadata["z_valid_range"] = z_valid_range
    
    kpoints = joblib.load(kpoint_file)

    if kpoints.shape[0] == 0:
        print(f"2D Keypoints for {avi_file} empty...")
        return None

    nbody_parts = kpoints.shape[1]

    # move it to 3d...
    kpoints_3d = np.full((nframes, nbody_parts, 4), fill_value=np.nan, dtype=np.float32)
    kpoints_3d[..., 0] = kpoints[..., 0]
    kpoints_3d[..., 1] = kpoints[..., 1]
    kpoints_3d[..., 3] = kpoints[..., 2]

    batches = range(0, nframes, batch_size)

    for _batch in tqdm(batches):

        working_range = range(_batch, min(_batch + batch_size, nframes))
        # working_range_arr = np.array(list(working_range)).astype("int")

        frame_batch = read_obj.get_frames(working_range).astype("float32")
        kpoint_batch = kpoints[working_range]


        for i in range(len(frame_batch)):
            use_frame = frame_batch[i]

            if replace_height_spikes_kwargs is not None:
                use_frame = replace_height_spikes(use_frame, **replace_height_spikes_kwargs)
            use_frame = cv2.bilateralFilter(use_frame.astype("float32"), **bilateral_kwargs)

            # fill holes in depth map TODO tune params
            use_frame = vid.util.fill_holes(use_frame)

            for j, _kpoint in enumerate(kpoint_batch[i]):
                # now we're in each body part...
                try:
                    x = int(np.round(_kpoint[0]))
                    y = int(np.round(_kpoint[1]))
                except ValueError:
                    continue
                    
                xi = int(round(x))
                yi = int(round(y))

                x0 = max(xi - patch_radius, 0)
                x1 = min(xi + patch_radius + 1, width)
                y0 = max(yi - patch_radius, 0)
                y1 = min(yi + patch_radius + 1, height)

                patch = use_frame[y0:y1,x0:x1]
                if patch.size == 0:
                    continue
                
                patch_valid = np.logical_and(patch >= z_valid_range[0], patch <= z_valid_range[1])
                patch = patch[patch_valid]
                if patch.size == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    try:
                        kpoints_3d[working_range[i], j, 2] = agg_func(patch)
                    except ValueError as e:
                        pass
        
    read_obj.close()
    
    if not os.path.exists(new_metadata_file):
        with open(new_metadata_file, "w") as f:
            toml.dump(new_metadata, f)
    joblib.dump(kpoints_3d, new_save_file)
    
    return None

def convert_2d_to_3d(kpoint_root_dir, avis, version_num, cable, node_names):
    kpoint_save_dir = f"_kpoints_v{version_num}_2d"

    nbody_parts = len(node_names)
    body_part_mapping = {_name: i for i, _name in enumerate(node_names)}

    for _avi in tqdm(avis):
        cam = os.path.splitext(os.path.basename(_avi))[0]

        dirname = os.path.dirname(_avi)
        save_dir = os.path.join(dirname, kpoint_save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f"{cam}.pkl.gz")

        sleap_file = os.path.join(kpoint_root_dir, f"{cam}.slp")
    
        sleap_dat = sio.load_file(sleap_file)
        nframes = len(sleap_dat.labeled_frames)
        new_arr = np.full((nframes, nbody_parts, 3), fill_value=np.nan)

        for i, _frame in enumerate(sleap_dat.labeled_frames):

            if len(_frame.instances) == 0 : 
                continue

            points = _frame.instances[0].points # _points returns all points, points only returns labeled points
            for j in range(points.shape[0]):
                _point = points[j]
                new_arr[i][j][0] = _point['xy'][0]  
                new_arr[i][j][1] = _point['xy'][1]
                new_arr[i][j][2] = _point['score']
     
        # save a toml with relevant stuff...
        metadata = {}
        metadata["node_names"] = node_names
        metadata["node_mapping"] = body_part_mapping
        metadata["sleap_path"] = sleap_file
        metadata["avi_path"] = _avi
        metadata["camera"] = cam
        metadata["undistorted"] = True # data already undistorted, make sure we know it...

        with open(os.path.join(save_dir, f"{cam}.toml"), "w") as f:
            toml.dump(metadata, f)

        joblib.dump(new_arr, save_file)

    if cable: 
        bilateral_kwargs={"d":11, "sigmaColor": 30, "sigmaSpace": 5}
        replace_height_spikes_kwargs={"threshold": 100, "ksize": 11}
        cable_agg_func = cable_agg_func_with_cable

    else: 
        bilateral_kwargs={"d":9, "sigmaColor": 15, "sigmaSpace":3}
        replace_height_spikes_kwargs=None
        cable_agg_func = cable_agg_func_no_cable

    delays = []

    print("Processing files to get 3D keypoints...")
    for _avi in avis:
        delays.append(
            joblib.delayed(get_3d_kpoints)(
                _avi,
                force=False,
                bilateral_kwargs=bilateral_kwargs,
                replace_height_spikes_kwargs=replace_height_spikes_kwargs,
                batch_size=3000,
                patch_radius=4,
                agg_func=cable_agg_func, # max for data without cables, median for data with cables...
                z_valid_range=(1,200),
                reader_kwargs={"threads": 2},
                save_dir = f"_kpoints_v{version_num}_3d",
                kpoint_2d_save_dir = f"_kpoints_v{version_num}_2d"
            )
            
        )


    print(f"{len(delays)} jobs to process")
    results = joblib.Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(delays)

    return results