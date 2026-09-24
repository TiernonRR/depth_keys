"""Run SLEAP-NN tracking on videos and report command-line progress."""

from pathlib import Path

import torch
import warnings
from sleap_nn.predict import run_inference

# Configuration constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CENTERED_INSTANCE_MODEL_PATH = PROJECT_ROOT / "models" / "centered_instance"
CENTROID_MODEL_PATH = PROJECT_ROOT / "models" / "centroid_unet"

# MAX_INSTANCES = 1
# PEAK_THRESHOLD = 0.0
# BATCH_SIZE = 64

torch.set_default_dtype(torch.float32)

def run_inference_on_video(
    video_path: str,
    output_path: str,
    ci_model_path: str = None,
    centroid_model_path: str = None,
    batch_size: int = 16,
    max_instances: int = 1,
    peak_threshold: float = 0.0,
) -> bool:
    """Run SLEAP-NN tracking on a video and save its predictions.

    Uses the configured default model paths when paths are omitted, and emits a
    warning for each default selected. Progress is printed by the command
    runner.

    Args:
        video_path: Path to the input video.
        output_path: Path for the output SLEAP ``.slp`` file.
        ci_model_path: Path to the centered-instance model directory.
        centroid_model_path: Path to the centroid model directory.
        batch_size: Number of frames to process in each inference batch.
        max_instances: Maximum number of instances to track per frame.
        peak_threshold: Minimum peak score used by SLEAP-NN for tracking.

    Returns:
        ``True`` after the SLEAP-NN command exits successfully.

    Raises:
        RuntimeError: If the SLEAP-NN command exits with a nonzero status.
    """
    print(f"Processing video: {video_path}")

    if ci_model_path is None:
        ci_model_path = str(CENTERED_INSTANCE_MODEL_PATH)
        warnings.warn(f"No centered model specified, attempting to load from {ci_model_path}")
    
    if centroid_model_path is None:
        centroid_model_path = str(CENTROID_MODEL_PATH)
        warnings.warn(f"No centroid model specified, attempting to load from {centroid_model_path}")

    cmd = make_sleap_nn_track_cmd(
        video_path=video_path,
        centroid_model_path=centroid_model_path,
        ci_model_path=ci_model_path,
        output_path=output_path,
        max_instances=max_instances,
        peak_threshold=peak_threshold,
        batch_size=batch_size
    )
    _ = run_sleap_nn_with_progress(cmd)

    
    return True


def make_sleap_nn_track_cmd(
    video_path,
    centroid_model_path,
    ci_model_path,
    output_path,
    max_instances=None,
    peak_threshold=None,
    batch_size=None,
    device="cuda",
):
    """Build the argument list for a SLEAP-NN tracking command.

    The command includes both model paths and ``--gui``. Optional flags are
    omitted when their values are ``None``.

    Args:
        video_path: Path to the input video.
        centroid_model_path: Path to the centroid model directory.
        ci_model_path: Path to the centered-instance model directory.
        output_path: Path for the output SLEAP ``.slp`` file.
        max_instances: Maximum instances per frame, if specified.
        peak_threshold: Minimum peak score, if specified.
        batch_size: Inference batch size, if specified.
        device: Device passed to SLEAP-NN; ``None`` omits the flag.

    Returns:
        Command and arguments as separate strings, suitable for a subprocess.
    """
    cmd = [
        "sleap-nn",
        "track",
        "--data_path", str(video_path),
        "--model_paths", str(centroid_model_path),
        "--model_paths", str(ci_model_path),
        "--output_path", str(output_path),
        "--gui",
    ]

    if max_instances is not None:
        cmd.extend(["--max_instances", str(max_instances)])

    if peak_threshold is not None:
        cmd.extend(["--peak_threshold", str(peak_threshold)])

    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])

    if device is not None:
        cmd.extend(["--device", device])

    return cmd



def run_sleap_nn_with_progress(
    cmd: list[str],
    *,
    log_every_s: float = 30.0,
) -> None:
    """Run a command and print SLEAP-NN progress from its output.

    Reads line-delimited JSON progress events from combined standard output and
    standard error. Prints other nonempty lines as text, reports the first and
    final progress events, and limits intermediate reports by ``log_every_s``.

    Args:
        cmd: Command and arguments to execute without a shell.
        log_every_s: Minimum seconds between intermediate progress reports.

    Raises:
        RuntimeError: If the command exits with a nonzero status.
    """
    import json
    import subprocess
    import time
    print("Running command:")
    print(" ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    last_log_t = 0.0
    last_progress = None

    assert proc.stdout is not None

    for raw_line in proc.stdout:
        line = raw_line.rstrip()

        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(line, flush=True)
            continue

        n_processed = event.get("n_processed")
        n_total = event.get("n_total")
        rate = event.get("rate")
        eta = event.get("eta")

        if n_processed is None or n_total is None:
            print(line, flush=True)
            continue

        last_progress = event

        now = time.monotonic()
        is_done = n_processed >= n_total
        should_log = (
            now - last_log_t >= log_every_s
            or is_done
            or last_log_t == 0.0
        )

        if should_log:
            pct = 100 * n_processed / n_total if n_total else 0.0

            msg = f"SLEAP-NN progress: {n_processed}/{n_total} frames ({pct:.1f}%)"

            if rate is not None:
                msg += f", {rate:.1f} fps"

            if eta is not None:
                eta_min = float(eta) / 60
                msg += f", ETA {eta_min:.1f} min"

            print(msg, flush=True)
            last_log_t = now

    return_code = proc.wait()

    if return_code != 0:
        raise RuntimeError(
            f"SLEAP-NN failed with exit code {return_code}. "
            f"Last progress event: {last_progress}"
        )
