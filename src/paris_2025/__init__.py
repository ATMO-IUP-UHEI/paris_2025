"""
Search for the config file "config.yaml" in the current directory and load it as a
global variable `CONFIG`. If the file does not exist, it raises a `FileNotFoundError`.
"""

from paris_2025 import config

try:
    CONFIG = config.load_config()
except FileNotFoundError as e:
    raise FileNotFoundError(
        "Configuration file 'config.yaml' not found in the current directory."
    ) from e

from paris_2025 import tracers, domain, meteo, model, background, model_input

__all__ = ["tracers", "domain", "meteo", "model", "background", "CONFIG", "model_input"]
