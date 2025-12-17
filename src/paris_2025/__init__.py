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

from paris_2025 import (
    background,
    domain,
    meteo,
    model,
    model_input,
    model_timeseries,
    plotting,
    tracers,
)

__all__ = [
    "tracers",
    "domain",
    "meteo",
    "model",
    "background",
    "CONFIG",
    "model_input",
    "model_timeseries",
    "plotting",
]
