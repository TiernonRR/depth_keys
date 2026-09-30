def build_slurm_command(command: str = "", 
                        ncpus: int = 8, 
                        memory: str = "20GB", 
                        wall_time: str = "05:00:00", 
                        qos: str = "embers", 
                        prefix: str = None, 
                        suffix: str = None, 
                        account: str = "gts-XXX", 
                        ngpus: int = 0, 
                        gpu_type: str = None, 
                        constraint: str = None):
    """Build an ``sbatch`` command string that wraps a shell command.

    Args:
        command: Shell command to run in the submitted job.
        ncpus: CPUs requested for the job.
        memory: Slurm memory request, such as ``20GB``.
        wall_time: Slurm time limit.
        qos: Slurm quality-of-service name.
        prefix: Optional shell text placed before ``command``.
        suffix: Optional shell text appended after ``command``.
        account: Slurm charge account name.
        ngpus: Number of GPUs to request.
        gpu_type: Optional GPU type prepended to the GPU count.
        constraint: Optional iterable of constraint values to append.

    Returns:
        An ``sbatch --wrap`` command string.
    """
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
        
    return run_command
