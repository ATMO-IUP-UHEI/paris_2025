import logging
from functools import lru_cache
from pathlib import Path

import ggpymanager as ggp


@lru_cache(maxsize=1)
def _load_config():
    try:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        logging.info(f"Loading config file {config_path} for project `paris_2025`")
        return ggp.io.read_project_yaml_file(config_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Configuration file 'config.yaml' not found in the current directory."
        ) from e


CONFIG = _load_config()
