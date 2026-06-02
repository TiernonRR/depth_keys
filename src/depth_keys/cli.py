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

# TODO:
# 1. Check directory for avis...
# 2. Handle file inputs, have env var options...
# 3. Write out a batch so that each one can be processed...
# fmt: off
@cli.command( name="create-2dkpoint-batch", context_settings={"show_default": True, "auto_envvar_prefix": "MARKOLABCLI_SLURM"}, )
@click.argument("command", type=str)
@click.argument("chk_dir", type=click.Path())
@click.option("--config-path", "-c", type=click.Path(), default="", help="Path to config file", envvar="DEPTHKEYS_CONFIG_FILE", show_envvar=True, )
@click.option("--ci-model-path", "-i", type=click.Path(), default="", help="Path to config file", envvar="DEPTHKEYS_CI_MODEL", show_envvar=True, )
@click.option("--centroid-model-path", "-m", type=click.Path(), default="", help="Path to config file", envvar="DEPTHKEYS_CENTROID_MODEL", show_envvar=True, )
@click.option("--reference-camera", "-c", type=str, default="Lucid Vision Labs-HTP003S-001-224500508", envvar="DEPTHKEYS_REFERENCE_CAMERA", show_envvar=True, )
@click.option("--cable", is_flag=True, help="Set flag if data contains a cable")
# fmt: on
@slurm_params
def create_slurm_cli(command, chk_dir, config_path, ci_model_path, centroid_model_path, reference_camera, cable, ncpus, memory, wall_time, qos, prefix, suffix, account, ngpus, gpu_type, constraint):
 	
	if chk_dir is None:
        chk_dir = os.getcwd() 
        
	

	# now walk through directories and ensure we have what we need etc...  

