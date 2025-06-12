import yaml
from pathlib import Path


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with config_path.open("r") as file:
        return yaml.safe_load(file)
