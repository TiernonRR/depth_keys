"""Lightweight import shims for optional runtime-only dependencies.

The unit tests replace SLEAP execution with fakes.  Keep collection usable on
machines that do not install the heavyweight SLEAP packages, while preserving
real modules whenever they are available.
"""

import importlib
import sys
import types


def _install_optional_import_stubs():
    """Provide import shims only for unavailable optional SLEAP dependencies."""
    try:
        importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        torch = types.ModuleType("torch")
        torch.float32 = object()
        torch.set_default_dtype = lambda dtype: None
        sys.modules["torch"] = torch

    try:
        importlib.import_module("sleap_io")
    except ModuleNotFoundError as exc:
        if exc.name != "sleap_io":
            raise
        sleap_io = types.ModuleType("sleap_io")
        # Conversion tests replace this callable with a fake loader; defining
        # the attribute keeps monkeypatch.setattr(..., raising=True) useful.
        sleap_io.load_file = None
        sys.modules["sleap_io"] = sleap_io

    try:
        importlib.import_module("sleap_nn.predict")
    except ModuleNotFoundError as exc:
        # Only synthesize the package when it is genuinely absent.  If a real
        # sleap_nn package exists but lacks predict, add the narrow submodule
        # without replacing the installed package object.
        if not (exc.name == "sleap_nn" or str(exc.name).startswith("sleap_nn.")):
            raise
        try:
            sleap_nn = importlib.import_module("sleap_nn")
        except ModuleNotFoundError as package_exc:
            if package_exc.name != "sleap_nn":
                raise
            sleap_nn = types.ModuleType("sleap_nn")
            sys.modules["sleap_nn"] = sleap_nn
        sleap_nn_predict = types.ModuleType("sleap_nn.predict")
        sleap_nn_predict.run_inference = lambda *args, **kwargs: None
        sleap_nn.predict = sleap_nn_predict
        sys.modules["sleap_nn.predict"] = sleap_nn_predict


_install_optional_import_stubs()
