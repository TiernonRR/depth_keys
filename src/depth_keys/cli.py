from depth_keys.slurm import build_slurm_command
import click
import functools
import os


@click.group()
def cli():
    pass


# TODO
# add code for applying sleap inference and conversion to 3d via
# command line
def slurm_params(func):
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
        return func(*args, **kwargs)


def kpoint_params(func):
    # fmt: off
    @click.option("--config-path", "-c", type=click.Path(), help="Path to config file", envvar="DEPTHKEYS_CONFIG", show_envvar=True, )
    @click.option("--ci-model-path", "-i", type=click.Path(), help="Path to centered instance model", envvar="DEPTHKEYS_CI_MODEL", show_envvar=True, )
    @click.option("--centroid-model-path", "-m", type=click.Path(), help="Path to centroid model", envvar="DEPTHKEYS_CENTROID_MODEL", show_envvar=True, )
    @click.option("--intrinsics-path", "-t", type=click.Path(), help="Path to camera intrinsics", envvar="DEPTHKEYS_INTRINSICS", show_envvar=True, )
    @click.option("--transform-path", "-t", type=click.Path(), help="Path to average transforms", envvar="DEPTHKEYS_AVG_TRANSFORM", show_envvar=True, )
    @click.option("--skeleton-path", "-t", type=click.Path(), help="Path to sleap json skeleton definition", envvar="DEPTHKEYS_SKELETON", show_envvar=True, )
    @click.option("--node-path", "-n", type=click.Path(), help="Path to node names", envvar="DEPTHKEYS_NODES", show_envvar=True, )
    @click.option("--reference-camera", "-c", type=str, default="Lucid Vision Labs-HTP003S-001-224500508", envvar="DEPTHKEYS_REFERENCE_CAMERA", show_envvar=True, )
    @click.option("--cable", is_flag=True, help="Set flag if data contains a cable")
    @click.option("--compute-2d", is_flag=True, help="Process 2d keypoints")
    @click.option("--compute-3d", is_flag=True, help="Process 3d keypoints")
    @click.option("--render", is_flag=True, help="Render keypoint data")
    @functools.wraps(func)
    # fmt: on
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)


# TODO:
# 1. Check directory for avis...
# 2. Handle file inputs, have env var options...
# 3. Write out a batch so that each one can be processed...
# fmt: off
@cli.command( name="create-kpoint-batch", context_settings={"show_default": True, "auto_envvar_prefix": "DEPTHKEYS"}, )
@click.argument("chk_dir", type=click.Path())
@kpoint_params
@slurm_params
def create_kpoint_batch(chk_dir, 
                        config_path, 
                        ci_model_path, 
                        intrinsics_path, 
                        centroid_model_path, 
                        transform_path, 
                        skeleton_path,
                        node_path,
                        reference_camera, 
                        cable, 
                        compute_2d,
                        compute_3d,
                        render,
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
    
    command = "depth_keys compute-keypoints {process_dir}"
    command += f" --config-path {config_path}"
    command += f" --ci-model-path {ci_model_path}"
    command += f" --centroid-model-path {centroid_model_path}"
    command += f" --transform-path {transform_path}"
    command += f" --skeleton-path {skeleton_path}"
    command += f" --node-path {node_path}"
    command += f" --reference-camera {reference_camera}"

    if cable:
        command += " --cable"

    if compute_2d:
        command += " --compute-2d"

    if compute_3d:
        command += " --compute-3d"

    if render:
        command += " --render" 

    if prefix is not None:
        base_command = f"{prefix};"
    else:
        base_command = ""

    if (gpu_type is not None) and (ngpus > 0):
        gpu_cmd = f"{gpu_type}:{ngpus}"
    else:
        gpu_cmd = f"{ngpus}"

    cluster_prefix = f'sbatch --gpus-per-node={gpu_cmd} --nodes 1 --ntasks-per-node 1 --cpus-per-task {ncpus:d} --mem={memory} -q {qos} -t {wall_time} -A {account} '

    try:
        iter(constraint)
    except TypeError as te:
        if constraint is not None:
            constraint = [constraint]

    if constraint is not None:
        for _constraint in constraint:
            cluster_prefix += f'--constraint="{_constraint}" '

    cluster_prefix += '--wrap "'

    issue_command = f"{cluster_prefix}{base_command}"
    
    if suffix is not None:
        run_command = f'{issue_command}{command}{suffix}"'
    else:
        run_command = f'{issue_command}{command}"'

    if chk_dir is None:
        chk_dir = os.getcwd() 
        
    # now walk through directories and ensure we have what we need etc...  
    # will need separate directory checks for 2d 3d, etc.




@cli.command( name="compute-keypoints", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOLABCLI_SLURM"}, )
@click.argument("proc_dir", type=click.Path())
@click.option("--config-path", "-c", type=click.Path(), help="Path to config file", envvar="DEPTHKEYS_CONFIG", show_envvar=True, )
@click.option("--ci-model-path", "-i", type=click.Path(), help="Path to centered instance model", envvar="DEPTHKEYS_CI_MODEL", show_envvar=True, )
@click.option("--centroid-model-path", "-m", type=click.Path(), help="Path to centroid model", envvar="DEPTHKEYS_CENTROID_MODEL", show_envvar=True, )
@click.option("--intrinsics-path", "-t", type=click.Path(), help="Path to camera intrinsics", envvar="DEPTHKEYS_INTRINSICS", show_envvar=True, )
@click.option("--transform-path", "-t", type=click.Path(), help="Path to average transforms", envvar="DEPTHKEYS_AVG_TRANSFORM", show_envvar=True, )
@click.option("--skeleton-path", "-t", type=click.Path(), help="Path to sleap json skeleton definition", envvar="DEPTHKEYS_SKELETON", show_envvar=True, )
@click.option("--node-path", "-n", type=click.Path(), help="Path to node names", envvar="DEPTHKEYS_NODES", show_envvar=True, )
@click.option("--reference-camera", "-c", type=str, default="Lucid Vision Labs-HTP003S-001-224500508", envvar="DEPTHKEYS_REFERENCE_CAMERA", show_envvar=True, )
@click.option("--cable", is_flag=True, help="Set flag if data contains a cable")
@click.option("--compute-2d", is_flag=True, help="Process 2d keypoints")
@click.option("--compute-3d", is_flag=True, help="Process 3d keypoints")
@click.option("--render", is_flag=True, help="Render keypoint data")
@click.option("--force", is_flag=True, help="Overwrite pre-existing data")
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
    from depth_keys.proc import process_directory
    import toml

    if proc_dir is None:
        proc_dir = os.getcwd()

    node_names = toml.load(node_path)
    process_directory(
        source_directory=proc_dir,
        config_path=config_path,
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
    )
