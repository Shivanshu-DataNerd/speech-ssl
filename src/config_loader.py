import os
import yaml


def load_config(config_name):
    """
    Load YAML config file safely.

    Args:
        config_name (str): name of yaml file inside configs/

    Returns:
        dict
    """

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root, "configs", config_name)

    if not os.path.exists(config_path):
        print(f"Config {config_name} not found, using defaults")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config