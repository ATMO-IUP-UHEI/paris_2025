# import os
from pathlib import Path

import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib as mpl
import xarray as xr
import pandas as pd

# import geopandas as gpd
import pyproj

# import shapely
# from tqdm import tqdm

from paris_2025.config import load_config

CONFIG = load_config()

TRACER_PATH = Path(CONFIG["data_path"]) / "6_measurements/6_2_tracers"
print(f"Using tracer path: {TRACER_PATH}")
TRACER_CO2_FILE = TRACER_PATH / "co2.nc"


def get_co2_measurement() -> xr.Dataset | None:
    """
    Load the CO2 measurement dataset from file, if it exists and is not empty.
    Returns
    -------
    xr.Dataset or None
        The CO2 measurement dataset, or None if not found or empty.
    """
    if not TRACER_CO2_FILE.exists():
        print(
            "CO2 measurement file not found. Please run the script to create it "
            "tracers.create_co2_measurement()"
        )
        return None
    co2 = xr.open_dataset(TRACER_CO2_FILE)
    if co2.station.size == 0:
        print(
            "CO2 measurement file is empty. Please run the script to create it. "
            "tracers.create_co2_measurement()"
        )
        return None
    return co2


def create_co2_measurement() -> None:
    """
    Create and save a CO2 measurement dataset from high-cost and mid-cost measurements.
    """
    if TRACER_CO2_FILE.exists():
        print("CO2 measurement file already exists. Please delete it to recreate it.")
        return
    TRACER_CO2_FILE.parent.mkdir(parents=True, exist_ok=True)
    high_cost = process_high_cost()
    mid_cost = process_mid_cost()
    co2 = xr.concat([high_cost, mid_cost], dim="station")
    # Add labels to the station dimension
    co2["station"] = [
        "{}_{}".format(code, type_)
        for code, type_ in zip(co2.code.values, co2.height.values)
    ]
    # Add attributes
    co2["station"].attrs = {
        "long_name": "Station code and height",
        "description": "Unique identifier for each CO2 measurement station, "
        "combining station code and sampling height.",
    }
    co2["co2"].attrs = {"units": "ppm", "long_name": "CO2 concentration"}
    co2["stdev"].attrs = {"units": "ppm", "long_name": "Standard deviation"}
    co2["nbpoints"].attrs = {"units": "1", "long_name": "Number of points"}
    co2["flag"].attrs = {"long_name": "Flag"}
    co2["altitude"].attrs = {"units": "m asl", "long_name": "Altitude"}
    co2["height"].attrs = {"units": "m agl", "long_name": "Height"}
    co2["x"].attrs = {"units": "m", "long_name": "x coordinate"}
    co2["y"].attrs = {"units": "m", "long_name": "y coordinate"}
    co2["latitude"].attrs = {"units": "degrees", "long_name": "latitude"}
    co2["longitude"].attrs = {"units": "degrees", "long_name": "longitude"}
    co2["station"].attrs = {"long_name": "Station code"}
    co2 = co2.set_coords(
        ["x", "y", "latitude", "longitude", "code", "type", "height", "altitude"]
    )
    co2.to_netcdf(TRACER_CO2_FILE, mode="w", unlimited_dims="time")


def process_high_cost() -> xr.Dataset:
    """
    Process high-cost CO2 measurement files and return as a single xarray.Dataset.
    """
    data_path = TRACER_PATH / "6_2_1_high-cost/"
    if not data_path.exists():
        raise FileNotFoundError(
            f"High-cost CO2 measurement data path does not exist: {data_path}"
        )
    measurement_type = "high-cost"
    read_function = read_co2_measurement_high_cost
    return process_files(data_path, measurement_type, read_function)


def process_mid_cost() -> xr.Dataset:
    data_path = TRACER_PATH / "6_2_2_mid-cost/2025_version/"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Mid-cost CO2 measurement data path does not exist: {data_path}"
        )
    measurement_type = "mid-cost"
    read_function = read_co2_measurement_mid_cost
    return process_files(data_path, measurement_type, read_function)


def process_files(data_path: Path, measurement_type: str, read_function) -> xr.Dataset:
    """
    Process mid-cost CO2 measurement files and return as a single xarray.Dataset.
    """
    xds_list = []
    if measurement_type == "high-cost":
        file_list = data_path.glob("*.co2")
    elif measurement_type == "mid-cost":
        file_list = data_path.glob("*_co2.csv")
    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")
    file_list = sorted(file_list)
    print(f"Processing {len(file_list)} {measurement_type} files...")
    for file in file_list:
        code, lat, lon, alt, height = read_location(file, measurement_type)
        print(code, height)
        x, y = pyproj.Proj(CONFIG["domain"]["crs"])(lon, lat)
        co2_measured = read_function(file)
        xds_list.append(
            xr.Dataset(
                {
                    "co2": (
                        ["time", "station"],
                        co2_measured["co2"].values[:, np.newaxis],
                    ),
                    "stdev": (
                        ["time", "station"],
                        co2_measured["Stdev"].values[:, np.newaxis],
                    ),
                    "nbpoints": (
                        ["time", "station"],
                        co2_measured["NbPoints"].values[:, np.newaxis],
                    ),
                    "flag": (
                        ["time", "station"],
                        co2_measured["Flag"].values[:, np.newaxis],
                    ),
                    "height": (["station"], [height]),
                    "altitude": (["station"], [alt]),
                    "latitude": (["station"], [lat]),
                    "longitude": (["station"], [lon]),
                    "x": (["station"], [x]),
                    "y": (["station"], [y]),
                    "code": (["station"], [code]),
                    "type": (["station"], [measurement_type]),
                },
                coords={"time": co2_measured.index.values},
            )
        )
    return xr.concat(xds_list, dim="station")


def read_location(path: Path, measurement_type: str) -> pd.Series:
    """
    Read the location and metadata from the header of a measurement file.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    pd.Series
        Series with code, lat, lon, alt, height.
    """
    with open(path) as file:
        lines = file.readlines()[:20]
    code, lat, lon, elevation, height = None, None, None, None, None
    for line in lines:
        param = line.split(":")[0]
        if param == "# STATION CODE":
            code = line.split(":")[1].split()[0]
        elif param == "# LATITUDE":
            lat = float(line.split(":")[1].split()[0])
        elif param == "# LONGITUDE":
            lon = float(line.split(":")[1].split()[0])
        elif param == "# ALTITUDE":
            elevation = float(line.split(":")[1].split()[0])
        elif param == "# SAMPLING HEIGHTS":
            height = float(line.split(":")[1].split()[0])
    if None in (code, lat, lon, elevation, height):
        raise ValueError(
            "Could not read all required parameters from the file header. "
            "Please check the file format."
        )
    if measurement_type == "mid-cost":
        # For mid-cost measurements, the altitude is not given, so we set it to NaN
        elevation = np.nan
    elif measurement_type == "high-cost":
        # For high-cost measurements, the altitude is given as elevation + height
        # and height is the sampling height above ground level.
        pass
    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")
    alt = elevation + height  # type: ignore
    return pd.Series(
        {"code": code, "lat": lat, "lon": lon, "alt": alt, "height": height}
    )


def read_co2_measurement_high_cost(path: Path) -> pd.DataFrame:
    """
    Read high-cost CO2 measurement data from file.
    """
    columns = [
        "Site",
        "SamplingHeight",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "DecimalDate",
        "co2",
        "Stdev",
        "NbPoints",
        "Flag",
        "InstrumentId",
        "QualityId",
        "InternalFlag",
        "AutoDescriptiveFlag",
        "ManualDescriptiveFlag",
    ]
    co2_measured = pd.read_csv(path, sep=";", comment="#", names=columns)
    co2_measured["Datetime"] = pd.to_datetime(
        co2_measured[["Year", "Month", "Day", "Hour", "Minute"]]
    )
    co2_measured = co2_measured.set_index("Datetime")
    co2_measured = filter_valid_flags(co2_measured, flag_column="Flag")
    return co2_measured


def read_co2_measurement_mid_cost(path: Path) -> pd.DataFrame:
    """
    Read mid-cost CO2 measurement data from file.
    """
    columns = [
        "Datetime",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "Second",
        "DecimalDate",
        "co2",
        "NbPoints",
        "Flag",
        "OriginalFlag",
        "Stdev",
    ]
    co2_measured = pd.read_csv(path, sep=";", comment="#", names=columns)
    co2_measured["Datetime"] = pd.to_datetime(co2_measured["Datetime"])
    co2_measured = co2_measured.set_index("Datetime")
    co2_measured = filter_valid_flags(co2_measured, flag_column="Flag")
    # Adjust sampling time from mid hour to start of hour
    co2_measured.index = co2_measured.index - pd.Timedelta("30min")
    return co2_measured


def filter_valid_flags(df: pd.DataFrame, flag_column: str = "Flag") -> pd.DataFrame:
    mask = (df[flag_column] == "O") | (df[flag_column] == "U")
    df.loc[~mask, :] = np.nan
    return df
