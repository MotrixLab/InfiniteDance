import os
from typing import Dict
from pathlib import Path

from yacs.config import CfgNode as CN


def to_lower(x: Dict) -> Dict:
    """
    Convert all dictionary keys to lowercase
    Args:
      x (dict): Input dictionary
    Returns:
      dict: Output dictionary with all keys converted to lowercase
    """
    return {k.lower(): v for k, v in x.items()}

_C = CN(new_allowed=True)

def default_config() -> CN:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

def get_config(config_file: str, merge: bool = True) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
      merge (bool): Whether to merge with the default config or not.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    if merge:
      cfg = default_config()
    else:
      cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)
    data_root = str(Path(os.environ.get(
        "INFINITEDANCE_DATA_ROOT", Path(__file__).resolve().parents[3] / "InfiniteDanceData"
    )).expanduser().resolve())
    def resolve_data_paths(node):
        for key, value in node.items():
            if isinstance(value, CN):
                resolve_data_paths(value)
            elif isinstance(value, str):
                node[key] = value.replace("${INFINITEDANCE_DATA_ROOT}", data_root)
    resolve_data_paths(cfg)
    cfg.freeze()
    return cfg

def dataset_config() -> CN:
    """
    Get dataset config file
    Returns:
      CfgNode: Dataset config as a yacs CfgNode object.
    """
    cfg = CN(new_allowed=True)
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets.yaml')
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

