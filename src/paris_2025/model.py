from pathlib import Path

import ggpymanager as ggpy
import xarray as xr

from paris_2025.config import load_config

CONFIG = load_config()

GRAMM_METEO_FILE = Path(CONFIG["gramm_meteo_path"]) / "meteo.nc"
CO2_DATA_PATH = Path(CONFIG["gral_co2_path"])
GRAL_CO2_FILE = CO2_DATA_PATH / "co2.nc"
GRAL_CO2_TIMESERIES_FILE = CO2_DATA_PATH / "rmse_co2_timeseries.nc"
GRAL_METEO_FILE = Path(CONFIG["gral_meteo_path"]) / "meteo.nc"


def get_gramm_meteo_data():
    if not GRAMM_METEO_FILE.exists():
        raise FileNotFoundError(f"Meteo data file not found: {GRAMM_METEO_FILE}")

    meteo_gramm = xr.open_dataset(GRAMM_METEO_FILE)
    # Fix some issues with the dataset
    # Drop var "speed" and rename "u" and "v" to "ux" and "vy" respectively
    meteo_gramm = meteo_gramm.drop_vars(["speed"])
    meteo_gramm = meteo_gramm.rename({"u": "ux", "v": "vy"})
    # Add "wind_speed" and "wind_direction" variables
    meteo_gramm["wind_speed"] = ggpy.utils.wind_speed_from_vector(
        meteo_gramm.ux, meteo_gramm.vy
    )
    meteo_gramm["wind_direction"] = ggpy.utils.direction_from_vector(
        meteo_gramm.ux, meteo_gramm.vy
    )
    return meteo_gramm


def get_gral_meteo_data():
    """Load meteorological data from GRAL."""
    if not GRAL_METEO_FILE.exists():
        raise FileNotFoundError(f"Meteo data file not found: {GRAL_METEO_FILE}")

    meteo_gral = xr.open_dataset(GRAL_METEO_FILE)
    # Fix some issues with the dataset
    # Rename "direction" and "speed" to "synoptic_wind_direction" and
    # "synoptic_wind_speed"
    meteo_gral = meteo_gral.rename({"direction": "synoptic_wind_direction"})
    meteo_gral = meteo_gral.rename({"speed": "synoptic_wind_speed"})
    # Add "wind_speed" and "wind_direction" variables
    meteo_gral["wind_speed"] = ggpy.utils.wind_speed_from_vector(
        meteo_gral.ux, meteo_gral.vy
    )
    meteo_gral["wind_direction"] = ggpy.utils.direction_from_vector(
        meteo_gral.ux, meteo_gral.vy
    )
    return meteo_gral


def get_co2_data():
    """Load CO2 data from GRAL."""
    if not GRAL_CO2_FILE.exists():
        raise FileNotFoundError(f"CO2 data file not found: {GRAL_CO2_FILE}")

    co2_gral = xr.open_dataset(GRAL_CO2_FILE)
    return co2_gral


def get_co2_time_series():
    """Load CO2 time series data from GRAL."""
    if not GRAL_CO2_TIMESERIES_FILE.exists():
        raise FileNotFoundError(
            f"CO2 time series file not found: {GRAL_CO2_TIMESERIES_FILE}"
        )

    co2_gral_timeseries = xr.open_dataset(GRAL_CO2_TIMESERIES_FILE)
    return co2_gral_timeseries
