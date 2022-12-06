import yaml
from typing import Dict
from yaml import CLoader
from types import SimpleNamespace


def load_config(config_filepath: str) -> Dict:
    """Read config from YAML into a dictionary"""
    with open(config_filepath, "r") as f:
        config = yaml.load(stream=f, Loader=CLoader)
    return config


def make_config_namespace(config: Dict) -> SimpleNamespace:
    """Make config dict into simple namespace"""
    return SimpleNamespace(**config)
