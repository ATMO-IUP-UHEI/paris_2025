"""Plotting modules for paris_2025.

For conventions (function signatures, data-loading helpers, saving, path
sourcing, and registration in create_figures.py) see CLAUDE.md at the repo root.
"""

from . import (
    common,
    fluxes,
    gradient_for_matching,
    matching_methods,
    meteo_from_catalog,
    meteo_measurements,
    tracer_background,
    tracer_comparison,
    tracer_from_catalog,
    tracer_measurements,
    _loaders,
)

__all__ = [
    "tracer_from_catalog",
    "tracer_background",
    "gradient_for_matching",
    "tracer_comparison",
    "meteo_measurements",
    "meteo_from_catalog",
    "tracer_measurements",
    "common",
    "fluxes",
    "matching_methods",
    "_loaders",
]

RC_PARAMS = {
    "figure.figsize": (12, 5),
    "savefig.dpi": 300,
    # "grid.alpha": 0.5,
    # "font.family": "sans-serif",
    # "font.sans-serif": "Arial",
    # "axes.grid": True,
    # "axes.spines.right": False,
    # "axes.spines.top": False,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.8,
    "lines.linewidth": 1.2000000000000002,
    "lines.markersize": 4.800000000000001,
    "patch.linewidth": 0.8,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.major.size": 4.800000000000001,
    "ytick.major.size": 4.800000000000001,
    "xtick.minor.size": 3.2,
    "ytick.minor.size": 3.2,
    "font.size": 9.600000000000001,
    "axes.labelsize": 9.600000000000001,
    "axes.titlesize": 9.600000000000001,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 8.8,
    "legend.title_fontsize": 9.600000000000001,
}

INVENTORY_SECTORS = {
    "Origins.earth 2023 energie": "Power",
    "Origins.earth 2023 industrie": "Industry",
    "Origins.earth 2023 residentiel": "Combustion",
    "Origins.earth 2023 respiration_humaine": "Human respiration",
    "Origins.earth 2023 tertiaire": "Services",
    "Origins.earth 2023 transport_routier": "Traffic",
    "TNO 2018 Combustion": "Combustion",
    "TNO 2018 Industry": "Industry",
    "TNO 2018 Power": "Power",
    "TNO 2018 Traffic": "Traffic",
    "VPRM GEE": "GEE",
    "VPRM R": "R",
}
INVENTORY_COLORS = {
    "Origins.earth 2023 energie": "tab:blue",
    "Origins.earth 2023 industrie": "tab:orange",
    "Origins.earth 2023 residentiel": "tab:red",
    "Origins.earth 2023 respiration_humaine": "tab:green",
    "Origins.earth 2023 tertiaire": "tab:cyan",
    "Origins.earth 2023 transport_routier": "tab:purple",
    "TNO 2018 Combustion": "tab:red",
    "TNO 2018 Industry": "tab:orange",
    "TNO 2018 Power": "tab:blue",
    "TNO 2018 Traffic": "tab:purple",
    "VPRM GEE": "#4c9121",
    "VPRM R": "#c41a7c",
}
