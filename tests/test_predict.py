import json
import subprocess
import sys
import types

import pytest

# These primarily test that commands get properly passed and output format is desired. 
# Scope of these tests should be acceptable, as predict burden is on sleap_io.

try:
    import depth_keys.kpoints.predict as predict
except ModuleNotFoundError as exc:
    # The project imports the optional SLEAP runtime at module import time.  The
    # unit tests replace the command runner, so a tiny import stub keeps these
    # tests runnable in a lightweight development environment.
    if exc.name not in {"torch", "sleap_nn", "sleap_nn.predict"}:
        raise
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.float32 = object()
        torch.set_default_dtype = lambda dtype: None
        sys.modules["torch"] = torch
    sleap_nn = types.ModuleType("sleap_nn")
    sleap_nn_predict = types.ModuleType("sleap_nn.predict")
    sleap_nn_predict.run_inference = lambda *args, **kwargs: None
    sleap_nn.predict = sleap_nn_predict
    sys.modules["sleap_nn"] = sleap_nn
    sys.modules["sleap_nn.predict"] = sleap_nn_predict
    import depth_keys.kpoints.predict as predict


def test_make_sleap_nn_track_cmd_preserves_tokens_and_options():
    command = predict.make_sleap_nn_track_cmd(
        "/data/video with spaces.avi",
        "/models/centroid",
        "/models/instance",
        "/out/predictions.slp",
        max_instances=2,
        peak_threshold=0.25,
        batch_size=8,
        device="cpu",
    )

    assert command == [
        "sleap-nn", "track", "--data_path", "/data/video with spaces.avi",
        "--model_paths", "/models/centroid", "--model_paths", "/models/instance",
        "--output_path", "/out/predictions.slp", "--gui",
        "--max_instances", "2", "--peak_threshold", "0.25",
        "--batch_size", "8", "--device", "cpu",
    ]


@pytest.mark.parametrize(
    "kwargs, omitted",
    [
        ({"max_instances": None, "peak_threshold": None, "batch_size": None, "device": None},
         {"--max_instances", "--peak_threshold", "--batch_size", "--device"}),
        ({}, set()),
    ],
)
def test_make_sleap_nn_track_cmd_optional_arguments(kwargs, omitted):
    command = predict.make_sleap_nn_track_cmd("v.avi", "centroid", "instance", "out.slp", **kwargs)
    assert omitted.isdisjoint(command)


def test_run_inference_on_video_uses_custom_models(monkeypatch):
    seen = {}
    def make_cmd(**kwargs):
        seen["args"] = kwargs
        return ["cmd"]
    def run_cmd(cmd):
        seen["cmd"] = cmd
    monkeypatch.setattr(predict, "make_sleap_nn_track_cmd", make_cmd)
    monkeypatch.setattr(predict, "run_sleap_nn_with_progress", run_cmd)
    import warnings
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        assert predict.run_inference_on_video("in.avi", "out.slp", "ci", "centroid", batch_size=4) is True

    # making sure argument are passed correctly
    assert records == []
    assert seen["args"]["ci_model_path"] == "ci"
    assert seen["args"]["centroid_model_path"] == "centroid"
    assert seen["args"]["batch_size"] == 4
    assert seen["cmd"] == ["cmd"]


def test_run_inference_on_video_warns_and_uses_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(predict, "make_sleap_nn_track_cmd", lambda **kwargs: seen.update(args=kwargs) or [])
    monkeypatch.setattr(predict, "run_sleap_nn_with_progress", lambda cmd: None)
    with pytest.warns(UserWarning) as records:
        predict.run_inference_on_video("in.avi", "out.slp")
    assert len(records) == 2

    # if we don't pass anything, does it still load in the default models
    assert seen["args"]["ci_model_path"] == str(predict.CENTERED_INSTANCE_MODEL_PATH)
    assert seen["args"]["centroid_model_path"] == str(predict.CENTROID_MODEL_PATH)


def test_run_inference_on_video_propagates_runner_error(monkeypatch):
    monkeypatch.setattr(predict, "run_sleap_nn_with_progress", lambda cmd: (_ for _ in ()).throw(RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        predict.run_inference_on_video("in.avi", "out.slp", "ci", "centroid")


class _FakeProcess:
    def __init__(self, lines, return_code=0):
        self.stdout = lines
        self.return_code = return_code
        self.wait_called = False

    def wait(self):
        self.wait_called = True
        return self.return_code


def test_progress_parser_passes_non_json_and_throttles(monkeypatch, capsys):
    lines = [
        "ordinary output\n", "\n", "{\"partial\": true}\n",
        json.dumps({"n_processed": 1, "n_total": 10, "rate": 2.0, "eta": 60}) + "\n",
        json.dumps({"n_processed": 2, "n_total": 10, "rate": 2.0}) + "\n",
        json.dumps({"n_processed": 10, "n_total": 10, "rate": 2.0, "eta": 0}) + "\n",
    ]
    process = _FakeProcess(lines)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    times = iter([1.0, 1.1, 2.0])
    monkeypatch.setattr("time.monotonic", lambda: next(times))

    predict.run_sleap_nn_with_progress(["sleap-nn", "track"], log_every_s=1.0)
    output = capsys.readouterr().out
    assert "ordinary output" in output
    assert '{"partial": true}' in output
    assert "1/10 frames (10.0%), 2.0 fps, ETA 1.0 min" in output
    assert "2/10 frames" not in output
    assert "10/10 frames (100.0%), 2.0 fps, ETA 0.0 min" in output
    assert process.wait_called


@pytest.mark.parametrize("return_code", [1, 17])
def test_progress_parser_reports_failed_process(monkeypatch, return_code):
    process = _FakeProcess([json.dumps({"n_processed": 3, "n_total": 5}) + "\n"], return_code)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(RuntimeError, match=f"exit code {return_code}"):
        predict.run_sleap_nn_with_progress(["cmd"])


def test_progress_parser_handles_zero_total(monkeypatch, capsys):
    process = _FakeProcess([json.dumps({"n_processed": 0, "n_total": 0}) + "\n"])
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    predict.run_sleap_nn_with_progress(["cmd"])
    assert "0/0 frames (0.0%)" in capsys.readouterr().out
