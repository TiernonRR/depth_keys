import pytest

from depth_keys.slurm import build_slurm_command


def test_build_slurm_command_defaults():
    result = build_slurm_command("depth-keys compute-keypoints /tmp/session")
    assert result == 'sbatch --gpus-per-node=0 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --mem=20GB -q embers -t 05:00:00 -A gts-XXX --wrap "depth-keys compute-keypoints /tmp/session"'


def test_build_slurm_command_gpu_prefix_suffix_account_and_constraints():
    result = build_slurm_command(
        "echo 'hello world'", ncpus=4, memory="8GB", wall_time="01:02:03",
        qos="gpu", prefix="module load cuda", suffix=" ; echo done",
        account="research", ngpus=2, gpu_type="A100", constraint=("gpu", "large"),
    )
    assert result.startswith("sbatch --gpus-per-node=A100:2 --nodes 1 --ntasks-per-node 1 --cpus-per-task 4 --mem=8GB -q gpu -t 01:02:03 -A research")
    assert '--constraint="gpu"' in result
    assert '--constraint="large"' in result
    assert '--wrap "module load cuda;echo' in result
    assert result.endswith("echo done\"")


def test_build_slurm_command_gpu_type_is_ignored_without_gpu_count():
    result = build_slurm_command("cmd", ngpus=0, gpu_type="A100", account="acct")
    assert "--gpus-per-node=0" in result
    assert "-A acct" in result


@pytest.mark.xfail(strict=True, reason="build_slurm_command always interpolates account, including account=None")
def test_build_slurm_command_omits_account_when_none():
    result = build_slurm_command("cmd", account=None)
    assert "-A" not in result


@pytest.mark.xfail(strict=True, reason="build_slurm_command treats a string constraint as an iterable of characters")
def test_build_slurm_command_single_string_constraint_is_one_constraint():
    result = build_slurm_command("cmd", constraint="a100")
    assert '--constraint="a100"' in result
    assert '--constraint="a"' not in result


def test_build_slurm_command_preserves_shell_sensitive_command_text():
    result = build_slurm_command("python script.py --name 'a b'", prefix="source env.sh", suffix="; touch output")
    assert "python script.py --name 'a b'" in result
    assert "source env.sh;" in result
    assert result.endswith("; touch output\"")
