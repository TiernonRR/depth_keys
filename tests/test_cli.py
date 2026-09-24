from pathlib import Path

import click
from click.testing import CliRunner
import pytest
import toml

import depth_keys.cli as cli_module


def test_shell_join_quotes_values_and_coerces_paths():
    assert cli_module.shell_join(["depth-keys", Path("folder with spaces"), "a'b", ""]) == "depth-keys 'folder with spaces' 'a'\"'\"'b' ''"


def test_cli_help_and_compute_help():
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["--help"])
    assert result.exit_code == 0
    assert "create-kpoint-batch" in result.output
    assert "compute-keypoints" in result.output
    result = runner.invoke(cli_module.cli, ["compute-keypoints", "--help"])
    assert result.exit_code == 0
    assert "--compute-2d" in result.output


def test_cli_rejects_unknown_command_and_missing_proc_dir():
    runner = CliRunner()
    unknown = runner.invoke(cli_module.cli, ["not-a-command"])
    assert unknown.exit_code != 0
    assert "No such command" in unknown.output
    missing = runner.invoke(cli_module.cli, ["compute-keypoints"])
    assert missing.exit_code != 0
    assert "Missing argument" in missing.output


def test_compute_keypoints_loads_nodes_and_forwards_all_options(tmp_path, monkeypatch):
    nodes = tmp_path / "nodes.toml"
    nodes.write_text('nodes = ["snout", "tail_tip"]\n')
    calls = []
    monkeypatch.setattr("depth_keys.proc.process_directory", lambda **kwargs: calls.append(kwargs))
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, [
        "compute-keypoints", str(tmp_path / "proc"),
        "--config-path", "config.toml", "--ci-model-path", "ci",
        "--centroid-model-path", "centroid", "--intrinsics-path", "intrinsics.toml",
        "--transform-path", "transforms.toml", "--skeleton-path", "skeleton.json",
        "--node-path", str(nodes), "--reference-camera", "cam", "--cable",
        "--compute-2d", "--compute-3d", "--render", "--force",
    ])
    assert result.exit_code == 0, result.output
    assert calls == [{
        "source_directory": str(tmp_path / "proc"), "registration_config_path": "config.toml",
        "ci_model_path": "ci", "centroid_model_path": "centroid", "intrinsics_path": "intrinsics.toml",
        "transforms_path": "transforms.toml", "skeleton_path": "skeleton.json",
        "reference_camera": "cam", "node_names": ["snout", "tail_tip"], "cable": True,
        "compute_2d": True, "compute_3d": True, "render": True, "force": True, "version_num": 1,
    }]


def test_compute_keypoints_accepts_explicit_environment_options(tmp_path, monkeypatch):
    nodes = tmp_path / "nodes.toml"
    nodes.write_text('nodes = ["snout"]\n')
    calls = []
    monkeypatch.setattr("depth_keys.proc.process_directory", lambda **kwargs: calls.append(kwargs))
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["compute-keypoints", str(tmp_path)], env={
        "DEPTHKEYS_CONFIG": "env-config.toml", "DEPTHKEYS_NODES": str(nodes),
        "DEPTHKEYS_INTRINSICS": "env-intrinsics.toml",
    })
    assert result.exit_code == 0, result.output
    assert calls[0]["registration_config_path"] == "env-config.toml"
    assert calls[0]["node_names"] == ["snout"]
    assert calls[0]["intrinsics_path"] == "env-intrinsics.toml"


def _candidate(parent, name):
    """Create a session candidate with one processed camera video."""
    proc = parent / name / "_proc"
    proc.mkdir(parents=True)
    (proc / "cam.avi").touch()
    return parent / name


def test_create_kpoint_batch_selects_incomplete_candidates_and_builds_slurm_command(tmp_path, monkeypatch):
    first = _candidate(tmp_path, "first")
    second = _candidate(tmp_path, "second")
    checks = {str(first / "_proc"): {"2d": False, "3d": False, "render": False}, str(second / "_proc"): {"2d": True, "3d": True, "render": True}}
    monkeypatch.setattr("depth_keys.proc.check_directory", lambda directory, **kwargs: checks[directory])
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, [
        "create-kpoint-batch", "--chk_dir", str(tmp_path), "--compute-2d", "--compute-3d", "--render",
        "--config-path", "config with space.toml", "--node-path", "nodes.toml",
        "--intrinsics-path", "intrinsics.toml", "--constraint", "gpu", "--constraint", "large",
        "--ngpus", "2", "--gpu-type", "A100", "--ncpus", "4", "--prefix", "module load x",
        "--suffix", "; echo done", "--account", "acct",
    ])
    assert result.exit_code == 0, result.output
    assert "first" in result.output
    assert "second" not in result.output
    assert "--gpus-per-node A100:2" in result.output
    assert "--constraint gpu" in result.output and "--constraint large" in result.output
    assert "intrinsics.toml" not in result.output


def test_create_kpoint_batch_force_includes_complete_candidate(tmp_path, monkeypatch):
    candidate = _candidate(tmp_path, "complete")
    monkeypatch.setattr("depth_keys.proc.check_directory", lambda *args, **kwargs: {"2d": True, "3d": True, "render": True})
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["create-kpoint-batch", "--chk_dir", str(tmp_path), "--compute-2d", "--force"])
    assert result.exit_code == 0, result.output
    assert "complete" in result.output


@pytest.mark.xfail(strict=True, reason="batch selection skips a directory when 2D is complete even though requested 3D remains incomplete")
def test_create_kpoint_batch_runs_both_stages_when_only_2d_is_complete(tmp_path, monkeypatch):
    candidate = _candidate(tmp_path, "needs-3d")
    monkeypatch.setattr("depth_keys.proc.check_directory", lambda *args, **kwargs: {"2d": True, "3d": False, "render": False})
    result = CliRunner().invoke(cli_module.cli, [
        "create-kpoint-batch", "--chk_dir", str(tmp_path), "--compute-2d", "--compute-3d",
    ])
    assert result.exit_code == 0, result.output
    assert "needs-3d" in result.output


@pytest.mark.xfail(strict=True, reason="create_kpoint_batch builds param_dct without --intrinsics-path")
def test_create_kpoint_batch_forwards_intrinsics_path(tmp_path, monkeypatch):
    _candidate(tmp_path, "candidate")
    monkeypatch.setattr("depth_keys.proc.check_directory", lambda *args, **kwargs: {"2d": False, "3d": False, "render": False})
    result = CliRunner().invoke(cli_module.cli, ["create-kpoint-batch", "--chk_dir", str(tmp_path), "--compute-2d", "--intrinsics-path", "intrinsics.toml"])
    assert "--intrinsics-path intrinsics.toml" in result.output


def test_create_kpoint_batch_kpoint_job_file_is_accepted_but_currently_unused(tmp_path, monkeypatch):
    _candidate(tmp_path, "candidate")
    monkeypatch.setattr("depth_keys.proc.check_directory", lambda *args, **kwargs: {"2d": False, "3d": False, "render": False})
    job = tmp_path / "job.toml"
    job.write_text("compute_2d = true\n")
    result = CliRunner().invoke(cli_module.cli, ["create-kpoint-batch", "--chk_dir", str(tmp_path), "--compute-2d", "--kpoint-job-file", str(job)])
    assert result.exit_code == 0, result.output
