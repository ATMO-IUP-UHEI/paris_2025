from pathlib import Path

import ggpymanager as ggpy
import xarray as xr

from paris_2025.config import load_config

CONFIG = load_config()

CO2_DATA_PATH = Path(CONFIG["gral_co2_path"])
MODEL_CO2_FILE = CO2_DATA_PATH / "co2.nc"
MODEL_CO2_TIMESERIES_FILE = CO2_DATA_PATH / "compound_co2_timeseries.nc"
MODEL_METEO_FILE = Path(CONFIG["gral_meteo_path"]) / "meteo.nc"


def get_meteo_data():
    """Load meteorological data from the model."""
    if not MODEL_METEO_FILE.exists():
        raise FileNotFoundError(f"Meteo data file not found: {MODEL_METEO_FILE}")

    meteo_model = xr.open_dataset(MODEL_METEO_FILE)
    # Fix some issues with the dataset
    # Rename "direction" and "speed" to "synoptic_wind_direction" and
    # "synoptic_wind_speed"
    if "direction" in meteo_model:
        meteo_model = meteo_model.rename({"direction": "synoptic_wind_direction"})
    if "speed" in meteo_model:
        meteo_model = meteo_model.rename({"speed": "synoptic_wind_speed"})
    # Add "wind_speed" and "wind_direction" variables
    meteo_model["wind_speed"] = ggpy.utils.wind_speed_from_vector(
        meteo_model.ux, meteo_model.vy
    )
    meteo_model["wind_direction"] = ggpy.utils.direction_from_vector(
        meteo_model.ux, meteo_model.vy
    )
    return meteo_model


def get_co2_data():
    """Load CO2 data from the model."""
    if not MODEL_CO2_FILE.exists():
        raise FileNotFoundError(f"CO2 data file not found: {MODEL_CO2_FILE}")

    co2_model = xr.open_dataset(MODEL_CO2_FILE)
    return co2_model


def get_co2_time_series():
    """Load CO2 time series data from the model."""
    if not MODEL_CO2_TIMESERIES_FILE.exists():
        raise FileNotFoundError(
            f"CO2 time series file not found: {MODEL_CO2_TIMESERIES_FILE}"
        )

    co2_timeseries = xr.open_dataset(MODEL_CO2_TIMESERIES_FILE)
    return co2_timeseries
