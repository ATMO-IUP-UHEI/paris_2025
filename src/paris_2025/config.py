import yaml
from pathlib import Path
import logging


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    logging.info(f"Loading config file {config_path} for project `paris_2025`")
    with config_path.open("r") as file:
        return yaml.safe_load(file)
