Process many sessions on Slurm
==============================

``create-kpoint-batch`` scans each immediate child of a parent directory for
``_proc/*.avi``. It prints one ``sbatch`` command per session with requested
work still to do; it does not submit jobs. Set the paths from
:doc:`processing` in the shell, then generate commands such as::

   depth-keys create-kpoint-batch \
     --chk_dir /path/to/sessions \
     --compute-2d --compute-3d --render \
     --ngpus 1 --gpu-type A100 \
     --ncpus 4 --memory 20GB --wall-time 05:00:00 \
     --qos YOUR_QOS --account YOUR_ACCOUNT

Review the printed ``sbatch`` commands and submit the ones you want to run.
Use ``--prefix`` if the wrapped command must first load modules or activate
the environment on the compute node. For example::

   --prefix "module load anaconda3/2023.03; conda activate depth-keys"

Request a GPU for 2D inference. If you are generating commands for 3D
conversion or rendering alone, request the resources those stages need.
``--force`` includes sessions even when the selected outputs already exist.
Use ``depth-keys create-kpoint-batch --help`` for the full list of Slurm
resource options.
