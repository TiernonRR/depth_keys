import json
from pathlib import Path

import pytest
import toml


ROOT = Path(__file__).parents[1]
DEFAULT = ROOT / "example_configs" / "default"
CABLE = ROOT / "example_configs" / "default_cable"


@pytest.mark.parametrize("config_path", [DEFAULT / "config.toml", CABLE / "config_cable.toml"])
def test_example_config_loads_and_has_global_contract(config_path):
    config = toml.load(config_path)
    assert isinstance(config["fps"], int)
    assert config["reference_camera"]
    assert config["skeleton"]
    assert config["incl_kpoints_fit_transform"]
    assert config["plt_kpoints"]
    assert config["renderer_kwargs"]["xlim"] == [-300, 300]


def test_default_nodes_and_skeleton_endpoints_match():
    nodes = toml.load(DEFAULT / "nodes.toml")["nodes"]
    skeleton = json.loads((DEFAULT / "skeleton.json").read_text())
    config_skeleton = toml.load(DEFAULT / "config.toml")["skeleton"]
    assert len(nodes) == len(set(nodes))
    assert {endpoint for edge in skeleton for endpoint in edge}.issubset(nodes)
    assert {edge[0] for edge in config_skeleton} | {edge[1] for edge in config_skeleton} <= set(nodes)


def test_cable_skeleton_endpoints_match_declared_keypoints():
    config = toml.load(CABLE / "config_cable.toml")
    skeleton = json.loads((CABLE / "skeleton.json").read_text())
    declared = set(config["noisy_keypoints"]) | set(config["plt_kpoints"]) | set(config["incl_kpoints_fit_transform"])
    assert {endpoint for edge in skeleton for endpoint in edge}.issubset(declared | {"left_ear", "right_ear"})


@pytest.mark.parametrize(
    "config_path, section",
    [(DEFAULT / "config.toml", "resolve_z"), (CABLE / "config_cable.toml", "post_processing")],
)
def test_example_processing_schema_has_patch_and_bilateral_parameters(config_path, section):
    config = toml.load(config_path)
    processing = config[section]
    assert processing["depth_patch_parameters"]
    for node, params in processing["depth_patch_parameters"].items():
        assert isinstance(node, str)
        assert params["patch_radius"] > 0
        assert 0 < params["agg_func"] <= 100
    for variant in ("depth_processing", "depth_processing_cable"):
        assert processing[variant]["bilateral_kwargs"]["d"] > 0
        assert "replace_height_spikes_kwargs" in processing[variant]


@pytest.mark.parametrize("transforms_path", [DEFAULT / "avg_transforms.toml", CABLE / "avg_transforms.toml"])
def test_example_transforms_parse_and_include_reference_identity(transforms_path):
    transforms = toml.load(transforms_path)
    identity_key = "('Lucid Vision Labs-HTP003S-001-224500508', 'Lucid Vision Labs-HTP003S-001-224500508')"
    assert identity_key in transforms
    rotation, translation = transforms[identity_key]
    assert rotation == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert translation == [0.0, 0.0, 0.0]
