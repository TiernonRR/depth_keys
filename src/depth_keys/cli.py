from depth_keys.slurm import build_slurm_command
from glob import glob
import click
import functools
import os
import logging
import shlex


VERSION_NUM = 1

@click.group()
def cli():
    """Group commands for keypoint processing and Slurm batch generation."""
    pass


def slurm_params(func):
    """Add Slurm resource and command-wrapper options to a Click command."""
    # fmt: off
    @click.option( "--ncpus", "-n", type=int, default=2, help="Number of CPUs", envvar="DEPTHKEYS_SLURM_NCPUS", show_envvar=True, )
    @click.option( "--memory", "-m", type=str, default="10GB", help="RAM string", envvar="DEPTHKEYS_SLURM_MEM", show_envvar=True, )
    @click.option( "--wall-time", "-w", type=str, default="3:00:00", help="Wall time", envvar="DEPTHKEYS_SLURM_WALLTIME", show_envvar=True, )
    @click.option( "--qos", type=str, default="inferno", help="QOS name", envvar="DEPTHKEYS_SLURM_QOS", show_envvar=True, )
    @click.option( "--prefix", type=str, default=None, help="Command prefix", envvar="DEPTHKEYS_SLURM_PREFIX", show_envvar=True, )
    @click.option( "--suffix", type=str, default=None, help="Command suffix", envvar="DEPTHKEYS_SLURM_SUFFIX", show_envvar=True, )
    @click.option( "--account", type=str, default=None, help="Account name", envvar="DEPTHKEYS_SLURM_ACCOUNT", show_envvar=True, )
    @click.option( "--ngpus", type=int, default=0, help="Number of GPUs to include in request", envvar="DEPTHKEYS_SLURM_NGPUS", show_envvar=True, )
    @click.option("--gpu-type", type=str, default=None, help="GPU type (e.g. A100, V100)", envvar="DEPTHKEYS_SLURM_GPUTYPE", show_envvar=True)
    @click.option("--constraint", type=str, default=None, multiple=True, help="constraint to add")
    @functools.wraps(func)
    # fmt: on
    def wrapper(*args, **kwargs):
        """Forward the Click options and arguments to the wrapped command."""
        return func(*args, **kwargs)
    
    return wrapper


def kpoint_params(func):
    """Add model, camera, processing-stage, and output options to a command."""
    # fmt: off
    @click.option("--config-path", type=click.Path(), help="Path to config file", envvar="DEPTHKEYS_CONFIG", show_envvar=True, )
    @click.option("--ci-model-path", type=click.Path(), help="Path to centered instance model", envvar="DEPTHKEYS_CI_MODEL", show_envvar=True, )
    @click.option("--centroid-model-path", type=click.Path(), help="Path to centroid model", envvar="DEPTHKEYS_CENTROID_MODEL", show_envvar=True, )
    @click.option("--intrinsics-path", type=click.Path(), help="Path to camera intrinsics", envvar="DEPTHKEYS_INTRINSICS", show_envvar=True, )
    @click.option("--transform-path", type=click.Path(), help="Path to transforms (ONLY NEEDED FOR MULTI-CAM REGISTRATION)", envvar="DEPTHKEYS_TRANSFORM", show_envvar=True, )
    @click.option("--skeleton-path", type=click.Path(), help="Path to sleap json skeleton definition", envvar="DEPTHKEYS_SKELETON", show_envvar=True, )
    @click.option("--node-path", type=click.Path(), help="Path to node names", envvar="DEPTHKEYS_NODES", show_envvar=True, )
    @click.option("--reference-camera", type=str, default="Lucid Vision Labs-HTP003S-001-224500508", envvar="DEPTHKEYS_REFERENCE_CAMERA", show_envvar=True, )
    @click.option("--cable", is_flag=True, help="Set flag if data contains a cable")
    @click.option("--compute-2d", is_flag=True, help="Process 2d keypoints")
    @click.option("--compute-3d", is_flag=True, help="Process 3d keypoints")
    @click.option("--render", is_flag=True, help="Render keypoint data")
    @click.option("--force", is_flag=True, help="Overwrite pre-existing data")
    @functools.wraps(func)
    # fmt: on
    def wrapper(*args, **kwargs):
        """Forward the Click options and arguments to the wrapped command."""
        return func(*args, **kwargs)

    return wrapper

def shell_join(args):
    """Quote and join command arguments for display as a shell command."""
    return shlex.join([str(a) for a in args])

# TODO:
# 2. Handle file inputs, have env var options...
# 3. Write out a batch so that each one can be processed...
# fmt: off
@cli.command( name="create-kpoint-batch", context_settings={"show_default": True, "auto_envvar_prefix": "DEPTHKEYS"}, )
@click.option("--chk_dir", type=click.Path(), default=None)
@click.option("--proc-sub-dir", type=str, default="_proc", help="Location with processed depth videos")
@click.option("--kpoint-job-file", type=click.Path(), help="Toml file that specifies parameters for keypoint computation (see compute-keypoints for options here)")
@kpoint_params
@slurm_params
def create_kpoint_batch(chk_dir, 
                        config_path,
                        kpoint_job_file,
                        proc_sub_dir,
                        ci_model_path,
                        centroid_model_path,
                        intrinsics_path,
                        transform_path,
                        skeleton_path,
                        node_path,
                        reference_camera,
                        cable,
                        compute_2d,
                        compute_3d,
                        render,
                        force,
                        ncpus, 
                        memory, 
                        wall_time, 
                        qos, 
                        prefix, 
                        suffix, 
                        account, 
                        ngpus, 
                        gpu_type, 
                        constraint):
    """Print Slurm commands for session directories needing keypoint work.

    Scans subdirectories of ``chk_dir`` for processed AVI files, checks their
    existing artifacts, and prints one ``sbatch`` command per selected session.
    This command does not submit the jobs.

    Args:
        chk_dir: Parent directory containing session directories.
        config_path: Registration and depth-processing TOML path.
        kpoint_job_file: Accepted job-file path; currently unused.
        proc_sub_dir: Processed-video subdirectory within each session.
        ci_model_path: Centered-instance model path.
        centroid_model_path: Centroid model path.
        intrinsics_path: Accepted intrinsics path; currently not forwarded.
        transform_path: Optional registration transforms path.
        skeleton_path: Skeleton JSON path for rendering.
        node_path: TOML file containing ordered node names.
        reference_camera: Reference camera identifier.
        cable: Whether sessions contain a cable.
        compute_2d: Whether to request 2D inference.
        compute_3d: Whether to request 3D conversion.
        render: Whether to request visualizations.
        force: Whether to regenerate outputs for requested stages.
        ncpus: CPUs requested from Slurm.
        memory: Slurm memory request.
        wall_time: Slurm time limit.
        qos: Slurm quality-of-service name.
        prefix: Optional shell text before the wrapped command.
        suffix: Optional text appended to the generated Slurm arguments.
        account: Slurm account name.
        ngpus: Number of requested GPUs.
        gpu_type: Optional GPU type.
        constraint: Optional Slurm constraints.
    """
    
    from depth_keys.proc import check_directory
    import shlex
    if chk_dir is None:
        chk_dir = os.getcwd()

    param_dct = {
        "--config-path": config_path,
        "--ci-model-path": ci_model_path,
        "--centroid-model-path": centroid_model_path,
        "--transform-path": transform_path,
        "--skeleton-path": skeleton_path,
        "--node-path": node_path,
        "--reference-camera": reference_camera,
        "--intrinsics-path": intrinsics_path
    }
    if cable:
        param_dct["--cable"] = None
    if compute_2d:
        param_dct["--compute-2d"] = None
    if compute_3d:
        param_dct["--compute-3d"] = None
    if render:
        param_dct["--render"] = None
    if force:
        param_dct["--force"] = None


    if (gpu_type is not None) and (ngpus > 0):
        gpu_cmd = f"{gpu_type}:{ngpus}"
    else:
        gpu_cmd = f"{ngpus}"

    cluster_prefix = [
        "sbatch",
        "--gpus-per-node", gpu_cmd,
        "--nodes", "1",
        "--ntasks-per-node", "1",
        "--cpus-per-task", ncpus,
        "--mem", memory,
        "-q", qos,
        "-t", wall_time,
        "-A", account,
    ]

    try:
        iter(constraint)
    except TypeError as te:
        if constraint is not None:
            constraint = [constraint]

    if constraint is not None:
        for _constraint in constraint:
            cluster_prefix += ["--constraint", _constraint]

    cluster_prefix.append("--wrap")
    
    if chk_dir is None:
        chk_dir = os.getcwd() 
        
    listing = sorted(os.listdir(chk_dir))
    listing = [os.path.join(chk_dir, _listing) for _listing in listing]
    listing = [_listing for _listing in listing if os.path.isdir(_listing)]
    
    include_dirs = []
    
    for _listing in listing:
        subdir = os.path.join(_listing, proc_sub_dir)
        if not os.path.exists(subdir):
            continue
        # possible candidate
        avis = glob(os.path.join(subdir, "*.avi"))
        if len(avis) == 0:
            continue
        
        iscomplete = check_directory(subdir, version_num=VERSION_NUM) # check what artefacts have been generated

        needs_2d = compute_2d and (force or not iscomplete["2d"])
        needs_3d = compute_3d and (force or not iscomplete["3d"] or needs_2d)
        needs_render = render and (force or not iscomplete["render"] or needs_3d)

        if not any((needs_2d, needs_3d, needs_render)):
            continue

        if needs_3d and not (needs_2d or iscomplete["2d"]):
            continue
        if needs_render and not (needs_3d or iscomplete["3d"]):
            continue

        include_dirs.append((_listing, needs_2d, needs_3d, needs_render))
    
    if prefix is not None and prefix[-1] != ";":
        prefix += ";"

    for _dir, needs_2d, needs_3d, needs_render in include_dirs:
        session_params = param_dct.copy()

        if not needs_2d:
            session_params.pop("--compute-2d", None)
        if not needs_3d:
            session_params.pop("--compute-3d", None)
        if not needs_render:
            session_params.pop("--render", None)

        command = ["depth-keys", "compute-keypoints", _dir]
        for option, value in session_params.items():
            command.append(option)
            if value is not None:
                command.append(value)

        wrap_str = shell_join(command)
        if prefix is not None:
            wrap_str = prefix + wrap_str

        use_list = cluster_prefix + [wrap_str]
        if suffix is not None:
            use_list.append(suffix)

        print(shell_join(use_list))





@cli.command( name="compute-keypoints", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOLABCLI_SLURM"}, )
@click.argument("proc_dir", type=click.Path())
@kpoint_params
# fmt: on
def compute_keypoints(
    proc_dir,
    config_path,
    ci_model_path,
    centroid_model_path,
    intrinsics_path,
    transform_path,
    skeleton_path,
    node_path,
    reference_camera,
    cable,
    compute_2d,
    compute_3d,
    render,
    force,
):    
    """Run selected processing stages for one session directory.

    Loads ordered node names from ``node_path`` and passes model, camera,
    registration, and rendering options to ``process_directory``.

    Args:
        proc_dir: Session directory to process.
        config_path: Depth-processing and registration TOML path.
        ci_model_path: Centered-instance model path.
        centroid_model_path: Centroid model path.
        intrinsics_path: Camera intrinsics TOML path.
        transform_path: Optional registration transforms path.
        skeleton_path: Skeleton JSON path for rendering.
        node_path: TOML file whose ``nodes`` key lists keypoint names.
        reference_camera: Reference camera identifier.
        cable: Whether the session contains a cable.
        compute_2d: Whether to run SLEAP inference.
        compute_3d: Whether to convert and register keypoints.
        render: Whether to render keypoint videos.
        force: Whether to regenerate existing files for requested stages.
    """
    if proc_dir is None:
        proc_dir = os.getcwd()
    proc_dir = os.path.normpath(proc_dir)
    cli_args = locals().copy()

    from depth_keys.proc import process_directory
    import toml

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    for k, v in cli_args.items():
        logger.info(f"CLI PARAMETERS {k}: {v}")

    node_names = toml.load(node_path)["nodes"]
    process_directory(
        source_directory=proc_dir,
        registration_config_path=config_path,
        ci_model_path=ci_model_path,
        centroid_model_path=centroid_model_path,
        intrinsics_path=intrinsics_path,
        transforms_path=transform_path,
        skeleton_path=skeleton_path,
        reference_camera=reference_camera,
        node_names=node_names,
        cable=cable,
        compute_2d=compute_2d,
        compute_3d=compute_3d,
        render=render,
        force=force,
        version_num=VERSION_NUM
    )
