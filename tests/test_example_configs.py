import json
from pathlib import Path

import pytest
import toml


ROOT = Path(__file__).parents[1]
DEFAULT = ROOT / "example_configs" / "default"
CABLE = ROOT / "example_configs" / "default_cable"
CONFIG_PATHS = [DEFAULT / "config.toml", CABLE / "config_cable.toml"]


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_example_config_loads_and_has_global_contract(config_path):
    config = toml.load(config_path)
    assert isinstance(config["fps"], int)
    assert config["reference_camera"]
    assert config["skeleton"]
    assert config["incl_kpoints_fit_transform"]
    assert config["plt_kpoints"]
    assert config["renderer_kwargs"]["xlim"] == [-300, 300]
    assert config["proc_order"]


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_example_nodes_and_skeleton_endpoints_match(config_path):
    nodes = toml.load(config_path.parent / "nodes.toml")["nodes"]
    skeleton = json.loads((config_path.parent / "skeleton.json").read_text())
    config = toml.load(config_path)
    config_skeleton = config["skeleton"]
    assert len(nodes) == len(set(nodes))
    assert {endpoint for edge in skeleton for endpoint in edge}.issubset(nodes)
    assert {edge[0] for edge in config_skeleton} | {edge[1] for edge in config_skeleton} <= set(nodes)
    assert set(config["post_processing"]["incl_kpoints_post_processing"]) <= set(nodes)
    assert set(config["incl_kpoints_fit_transform"]) <= set(nodes)
    assert set(config["plt_kpoints"]) <= set(nodes)
    assert set(config["index_conf_map"]) == {str(i) for i in range(len(nodes))}


def test_cable_nodes_omit_snout_and_preserve_remaining_confidence_thresholds():
    default_nodes = toml.load(DEFAULT / "nodes.toml")["nodes"]
    cable_nodes = toml.load(CABLE / "nodes.toml")["nodes"]
    assert cable_nodes == [node for node in default_nodes if node != "snout"]
    default_config = toml.load(DEFAULT / "config.toml")
    cable_config = toml.load(CABLE / "config_cable.toml")
    thresholds_by_node = {
        node: default_config["index_conf_map"][str(i)]
        for i, node in enumerate(default_nodes)
    }
    for i, node in enumerate(cable_nodes):
        assert cable_config["index_conf_map"][str(i)] == thresholds_by_node[node]


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_example_processing_schema_has_patch_and_bilateral_parameters(config_path):
    config = toml.load(config_path)
    processing = config["resolve_z"]
    assert set(processing) == {"depth_processing", "depth_patch_parameters"}
    assert processing["depth_patch_parameters"]
    nodes = toml.load(config_path.parent / "nodes.toml")["nodes"]
    assert set(nodes) <= set(processing["depth_patch_parameters"])
    for node, params in processing["depth_patch_parameters"].items():
        assert isinstance(node, str)
        assert params["patch_radius"] > 0
        assert 0 < params["agg_func"] <= 100
    assert processing["depth_processing"]["bilateral_kwargs"]["d"] > 0
    assert "replace_height_spikes_kwargs" in processing["depth_processing"]
    assert not set(processing) & set(config["post_processing"])


def test_example_configs_share_parameter_names():
    default = toml.load(DEFAULT / "config.toml")
    cable = toml.load(CABLE / "config_cable.toml")

    def parameter_paths(table, prefix=()):
        for key, value in table.items():
            if prefix == ("index_conf_map",):
                key = "<node_index>"
            path = (*prefix, key)
            if isinstance(value, dict) and key != "replace_height_spikes_kwargs":
                yield from parameter_paths(value, path)
            else:
                yield path

    assert set(parameter_paths(default)) == set(parameter_paths(cable))
    assert default["proc_order"] == ["temporal", "bone", "interpolate", "bone"]
    assert cable["proc_order"] == ["temporal", "bone", "bone", "final_smooth"]


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_example_post_processing_matches_markovids_contract(config_path):
    post_processing = pytest.importorskip("markovids.pcl.post_processing")
    config = toml.load(config_path)
    params = dict(config["post_processing"])
    params.pop("incl_kpoints_post_processing")
    params["temporal_regularization"]["fps"] = config["fps"]
    post_processing.PostProcessingConfig(**params)
    assert set(config["proc_order"]) <= post_processing.KeypointPostProcessor.VALID_PROC_STAGES


@pytest.mark.parametrize("transforms_path", [DEFAULT / "avg_transforms.toml", CABLE / "avg_transforms.toml"])
def test_example_transforms_parse_and_include_reference_identity(transforms_path):
    transforms = toml.load(transforms_path)
    identity_key = "('Lucid Vision Labs-HTP003S-001-224500508', 'Lucid Vision Labs-HTP003S-001-224500508')"
    assert identity_key in transforms
    rotation, translation = transforms[identity_key]
    assert rotation == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert translation == [0.0, 0.0, 0.0]
