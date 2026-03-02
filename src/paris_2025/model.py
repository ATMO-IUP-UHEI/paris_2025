from pathlib import Path

import ggpymanager as ggpy
import xarray as xr

from paris_2025.config import CONFIG

GRAMM_METEO_FILE = Path(CONFIG["gramm_meteo_path"]) / "meteo.nc"
CO2_DATA_PATH = Path(CONFIG["gral_co2_path"])
GRAL_CO2_FILE = CO2_DATA_PATH / "co2.nc"
GRAL_CO2_TIMESERIES_FILE = CO2_DATA_PATH / "rmse_co2_timeseries.nc"
GRAL_METEO_FILE = Path(CONFIG["gral_meteo_path"]) / "meteo.nc"


def get_gramm_meteo_data() -> xr.Dataset:
    return ggpy.load("gramm_meteo_catalog", CONFIG)


def get_gral_meteo_data() -> xr.Dataset:
    """Load meteorological data from GRAL."""
    return ggpy.load("gral_meteo_catalog", CONFIG)


def get_model_meteo_data() -> xr.Dataset:
    return ggpy.load("model_meteo", CONFIG)


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
