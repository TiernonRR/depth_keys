import sleap_io as sio
import os
import joblib
import numpy as np
import warnings
import cv2
import copy 
import toml

from markovids import vid

from tqdm.auto import tqdm


def _to_plain_types(obj):
    """Recursively copy mappings and sequences into plain Python containers."""
    if isinstance(obj, dict):
        return {k: _to_plain_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_types(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain_types(v) for v in obj)
    return obj


def _get_resolve_z_config(config, config_path):
    """Return depth settings from ``resolve_z`` or the legacy section.

    Raises:
        KeyError: If neither configuration section is a table.
    """
    resolve_z_cfg = config.get("resolve_z")
    if isinstance(resolve_z_cfg, dict):
        return resolve_z_cfg

    post_cfg = config.get("post_processing")
    if isinstance(post_cfg, dict):
        return post_cfg

    raise KeyError(
        "Missing [resolve_z] (or backward-compatible [post_processing]) section "
        f"in config: {config_path}"
    )


def _load_depth_params(config_path):
    """Load per-node depth patch settings from a TOML configuration.

    Raises:
        KeyError: If the required section or parameter table is missing.
        TypeError: If the parameter table is not a mapping.
    """
    config = toml.load(config_path)
    resolve_z_cfg = _get_resolve_z_config(config, config_path)
    depth_params = resolve_z_cfg.get("depth_patch_parameters")
    if depth_params is None:
        raise KeyError(
            "Missing [resolve_z.depth_patch_parameters] "
            "(or backward-compatible [post_processing.depth_patch_parameters]) "
            f"in config: {config_path}"
        )
    if not isinstance(depth_params, dict):
        raise TypeError(
            "[resolve_z.depth_patch_parameters] must be a table of node names to settings."
        )
    return _to_plain_types(depth_params)


def _load_depth_processing_overrides(config_path, cable):
    """Load bilateral and optional spike-filter settings for a video type.

    Args:
        config_path: Path to the depth-processing TOML configuration.
        cable: Whether to select the cable-specific processing section.

    Returns:
        The bilateral-filter arguments and optional spike-filter arguments.

    Raises:
        KeyError: If the selected section or bilateral settings are missing.
    """
    config = toml.load(config_path)
    resolve_z_cfg = _get_resolve_z_config(config, config_path)
    variant_key = "depth_processing_cable" if cable else "depth_processing"
    depth_cfg = resolve_z_cfg.get(variant_key)
    if depth_cfg is None:
        raise KeyError(
            f"Missing [resolve_z.{variant_key}] "
            f"(or backward-compatible [post_processing.{variant_key}]) in config: {config_path}"
        )

    bilateral_kwargs = depth_cfg.get("bilateral_kwargs")
    if bilateral_kwargs is None:
        raise KeyError(
            f"Missing [resolve_z.{variant_key}.bilateral_kwargs] in config: {config_path}"
        )
    bilateral_kwargs = _to_plain_types(bilateral_kwargs)

    replace_height_spikes_kwargs = depth_cfg.get("replace_height_spikes_kwargs")
    if not replace_height_spikes_kwargs:
        replace_height_spikes_kwargs = None
    else:
        replace_height_spikes_kwargs = _to_plain_types(replace_height_spikes_kwargs)
    return bilateral_kwargs, replace_height_spikes_kwargs


def _normalize_node_name(node_name):
    """Convert a point name to text, decoding bytes and preserving ``None``."""
    if node_name is None:
        return None
    if isinstance(node_name, bytes):
        return node_name.decode("utf-8", errors="ignore")
    return str(node_name)


def _map_instance_points_to_array(points, frame_arr, body_part_mapping, node_names):
    """Copy one instance's coordinates and scores into a frame array.

    Matches named points to ``body_part_mapping``. Unnamed points may use their
    position in ``node_names``; unknown names are skipped. Updates ``frame_arr``
    in place with ``(x, y, score)`` values.
    """
    for j in range(len(points)):
        point = points[j]

        point_name = None
        try:
            point_name = point["name"]
        except Exception:
            point_name = None

        normalized_name = _normalize_node_name(point_name)
        point_index = body_part_mapping.get(normalized_name)

        # Backward-compatible fallback: use positional assignment only when names
        # are missing or positional name matches expected node.
        if point_index is None and j < len(node_names):
            expected_name = node_names[j]
            if normalized_name is None or normalized_name == expected_name:
                point_index = j

        if point_index is None:
            continue
        frame_arr[point_index][0] = point["xy"][0]
        frame_arr[point_index][1] = point["xy"][1]
        frame_arr[point_index][2] = point["score"]

def replace_height_spikes(depth_map, threshold=30, ksize=5, z_scale=4):
    """Replace depth values far from their local median.

    For kernels larger than five, scales the depth map before an 8-bit median
    filter and scales the median back afterward.

    Args:
        depth_map: Two-dimensional depth array.
        threshold: Minimum absolute difference that triggers replacement.
        ksize: Odd median-filter kernel size.
        z_scale: Scale factor used for kernels larger than five.

    Returns:
        A filtered copy of ``depth_map`` with its original dtype.
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
    config_path,
    batch_size=2000,
    kpoint_2d_save_dir="_kpoints_v0_2d",
    save_dir="_kpoints_v0_3d",
    z_valid_range = (1,200),
    reader_kwargs=None,
    bilateral_kwargs={"d":5, "sigmaColor": 15, "sigmaSpace":3},
    replace_height_spikes_kwargs=None
):
    """Attach video depth values to 2D keypoints and save camera artifacts.

    Reads the matching 2D keypoint array and metadata beside ``avi_file``.
    Each valid point receives the configured percentile of a filtered depth
    patch. Writes ``<camera>.pkl.gz`` containing ``(x, y, z, score)`` values
    and ``<camera>.toml`` under ``save_dir`` beside the video.

    Args:
        avi_file: Path to the depth video.
        config_path: TOML file with per-node depth patch settings.
        batch_size: Number of video frames read per batch.
        kpoint_2d_save_dir: 2D artifact directory beside the video.
        save_dir: Destination directory beside the video.
        z_valid_range: Inclusive range of valid depth values.
        reader_kwargs: Additional arguments for ``AutoReader``.
        bilateral_kwargs: Arguments for OpenCV's bilateral filter.
        replace_height_spikes_kwargs: Optional arguments for spike filtering.

    Returns:
        ``None`` after writing files, or when output exists or 2D data is empty.

    Raises:
        KeyError: If a node lacks required depth patch settings.
    """
    if reader_kwargs is None:
        reader_kwargs = {"threads": 2}

    depth_patch_parameters = _load_depth_params(config_path)

    avi_dir = os.path.dirname(avi_file)
    cam = os.path.splitext(os.path.basename(avi_file))[0]
    
    new_save_dir = os.path.join(avi_dir, save_dir)
        
    new_save_file = os.path.join(new_save_dir, f"{cam}.pkl.gz")
    if os.path.exists(new_save_file):
        warnings.warn(f"3D keypoints already computed for {avi_file}, skipping...")
        return None 
    new_metadata_file = os.path.join(new_save_dir, f"{cam}.toml")
    
    os.makedirs(new_save_dir, exist_ok=True)

    read_obj = vid.io.AutoReader(avi_file, **reader_kwargs)
    nframes = read_obj.nframes
    width, height = read_obj.frame_size

    kpoint_file = os.path.join(avi_dir, kpoint_2d_save_dir, f"{cam}.pkl.gz")
        
    metadata_file = os.path.join(avi_dir, kpoint_2d_save_dir, f"{cam}.toml")
    
    print(kpoint_file)
    print(metadata_file)

    metadata = toml.load(metadata_file)
    new_metadata = copy.deepcopy(metadata)

    new_metadata["z_valid_range"] = z_valid_range
    new_metadata["depth_patch_parameters"] = depth_patch_parameters
    
    kpoints = joblib.load(kpoint_file)

    if kpoints.shape[0] == 0:
        print(f"2D Keypoints for {avi_file} empty...")
        return None

    nbody_parts = kpoints.shape[1]
    node_names = new_metadata.get("node_names", [])
    missing_depth_nodes = sorted(set(node_names) - set(depth_patch_parameters.keys()))
    if missing_depth_nodes:
        raise KeyError(
            "Missing depth patch parameters for node(s): "
            + ", ".join(missing_depth_nodes)
        )

    # move it to 3d...
    kpoints_3d = np.full((nframes, nbody_parts, 4), fill_value=np.nan, dtype=np.float32)
    kpoints_3d[..., 0] = kpoints[..., 0]
    kpoints_3d[..., 1] = kpoints[..., 1]
    kpoints_3d[..., 3] = kpoints[..., 2]

    batches = range(0, nframes, batch_size)

    for _batch in tqdm(batches):

        working_range = range(_batch, min(_batch + batch_size, nframes))

        frame_batch = read_obj.get_frames(working_range).astype("float32")
        kpoint_batch = kpoints[working_range]


        for i in range(len(frame_batch)):
            use_frame = frame_batch[i]

            if replace_height_spikes_kwargs is not None:
                use_frame = replace_height_spikes(use_frame, **replace_height_spikes_kwargs)
            use_frame = cv2.bilateralFilter(use_frame.astype("float32"), **bilateral_kwargs)

            use_frame = vid.util.fill_holes(use_frame)

            for j, _kpoint in enumerate(kpoint_batch[i]):
                node_name = node_names[j]

                patch_radius = depth_patch_parameters[node_name]["patch_radius"]
                percentile = depth_patch_parameters[node_name]["agg_func"]

                # now we're in each body part...
                # try:
                #     x = int(np.round(_kpoint[0]))
                #     y = int(np.round(_kpoint[1]))
                # except ValueError:
                #     continue

                if not np.isfinite(_kpoint[0]) or not np.isfinite(_kpoint[1]): 
                    continue
                    
                xi = int(np.round(_kpoint[0]))
                yi = int(np.round(_kpoint[1]))

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
                    kpoints_3d[working_range[i], j, 2] = np.nanpercentile(patch, percentile)
        
    read_obj.close()
    
    with open(new_metadata_file, "w") as f:
        toml.dump(new_metadata, f)
    joblib.dump(kpoints_3d, new_save_file)
    
    return None

def convert_2d_to_3d(
    kpoint_root_dir,
    avis,
    version_num,
    cable,
    node_names,
    config_path,
    conda_env_name=None,
):
    """Convert SLEAP predictions to per-camera 3D keypoint files.

    Creates ``(x, y, score)`` arrays and metadata beside each video, then
    processes depth for all videos in parallel. Existing 2D arrays are reused.

    Args:
        kpoint_root_dir: Directory containing camera-named ``.slp`` files.
        avis: Paths to depth videos, one per camera.
        version_num: Version included in 2D and 3D output directory names.
        cable: Whether to use cable-specific depth filtering settings.
        node_names: Ordered names defining output point positions.
        config_path: TOML file with depth patch and filtering settings.
        conda_env_name: Optional environment activated by the video reader.

    Returns:
        One result per video from the parallel ``get_3d_kpoints`` calls.
    """
    kpoint_save_dir = f"_kpoints_v{version_num}_2d"

    nbody_parts = len(node_names)
    body_part_mapping = {_name: i for i, _name in enumerate(node_names)}

    for _avi in tqdm(avis):
        cam = os.path.splitext(os.path.basename(_avi))[0]

        dirname = os.path.dirname(_avi)
        save_dir = os.path.join(dirname, kpoint_save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f"{cam}.pkl.gz")

        if not os.path.exists(save_file):
            sleap_file = os.path.join(kpoint_root_dir, f"{cam}.slp")
        
            sleap_dat = sio.load_file(sleap_file)
            nframes = len(sleap_dat.labeled_frames)
            new_arr = np.full((nframes, nbody_parts, 3), fill_value=np.nan)

            for i, _frame in enumerate(sleap_dat.labeled_frames):

                if len(_frame.instances) == 0 : 
                    continue

                points = _frame.instances[0].points
                _map_instance_points_to_array(points, new_arr[i], body_part_mapping, node_names)
        
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
        else:
            new_arr = joblib.load(save_file)

    bilateral_kwargs, replace_height_spikes_kwargs = _load_depth_processing_overrides(
        config_path, cable
    )

    delays = []
    reader_kwargs = {"threads": 2}
    if conda_env_name:
        reader_kwargs["prepend_args"] = (
            f"source ~/conda_activate ; conda activate {conda_env_name}"
        )
    reader_kwargs = _to_plain_types(reader_kwargs)

    print("Processing files to get 3D keypoints...")
    for _avi in avis:
        delays.append(
            joblib.delayed(get_3d_kpoints)(
                _avi,
                config_path=config_path,
                bilateral_kwargs=bilateral_kwargs,
                replace_height_spikes_kwargs=replace_height_spikes_kwargs,
                batch_size=3000,
                z_valid_range=(1,200),
                reader_kwargs=reader_kwargs,
                save_dir = f"_kpoints_v{version_num}_3d",
                kpoint_2d_save_dir = f"_kpoints_v{version_num}_2d"
            )
            
        )


    print(f"{len(delays)} jobs to process")
    results = joblib.Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(delays)

    return results
