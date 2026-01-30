"""
Search for the config file "config.yaml" in the current directory and load it as a
global variable `CONFIG`. If the file does not exist, it raises a `FileNotFoundError`.
"""

from paris_2025 import (
    background,
    domain,
    google_earth_files,
    meteo,
    model,
    model_input,
    plotting,
    tracers,
)
from paris_2025.config import CONFIG

__all__ = [
    "tracers",
    "domain",
    "google_earth_files",
    "meteo",
    "model",
    "background",
    "CONFIG",
    "model_input",
    "plotting",
]
