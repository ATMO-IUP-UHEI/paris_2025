from pathlib import Path
import logging
import ggpymanager as ggp


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    logging.info(f"Loading config file {config_path} for project `paris_2025`")
    return ggp.io.read_project_yaml_file(config_path)
