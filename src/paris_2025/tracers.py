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


def get_co2_measurements() -> xr.Dataset:
    """
    Load the CO2 measurement dataset from file, if it exists and is not empty.

    Returns
    -------
    xr.Dataset
        The CO2 measurement dataset.
    """
    if not TRACER_CO2_FILE.exists():
        raise FileNotFoundError(
            "CO2 measurement file not found. Please run the script to create it "
            "tracers.create_co2_measurement()"
        )
    co2 = xr.open_dataset(TRACER_CO2_FILE)
    if co2.station.size == 0:
        raise ValueError(
            "CO2 measurement file is empty. Please run the script to create it. "
            "tracers.create_co2_measurement()"
        )
    return co2


def create_co2_measurements() -> None:
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
    # Add coordinates in GRAL projection
    x, y = pyproj.Proj(CONFIG["domain"]["crs"])(
        co2.longitude.values, co2.latitude.values
    )
    co2["x"] = (["station"], x)
    co2["y"] = (["station"], y)
    # Add labels to the station dimension
    co2["station"] = [
        "{}_{:.0f}".format(code[:3], type_)
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
    co2["code"].attrs = {
        "long_name": "Station code",
        "description": "Short code for the CO2 measurement station",
    }
    co2["name"].attrs = {
        "long_name": "Station name",
        "description": "Name of the CO2 measurement station",
    }
    co2["type"].attrs = {
        "long_name": "Measurement type",
        "description": "Type of CO2 measurement (high-cost or mid-cost)",
    }
    co2["time"].attrs = {
        "long_name": "Time of measurement (UTC)",
        "description": "Time when the CO2 measurement was taken",
        "standard_name": "time",
    }
    co2["instrument"].attrs = {
        "long_name": "Instrument used for measurement",
        "description": "Identifier for the instrument used to measure CO2",
    }
    co2["HPP_ID|K96_ID"].attrs = {
        "long_name": "HPP_ID or K96_ID",
        "description": "Identifier for the HPP or K96 station",
    }
    co2["box_id"].attrs = {
        "long_name": "Box ID",
        "description": (
            "Identifier for the box containing the CO2 measurement instrument",
        ),
    }
    # Set coordinates
    co2 = co2.set_coords(
        [
            "x",
            "y",
            "latitude",
            "longitude",
            "code",
            "type",
            "height",
            "altitude",
            "name",
            "instrument",
            "HPP_ID|K96_ID",
            "box_id",
        ]
    )
    # Add global attributes
    co2.attrs = {
        "title": "CO2 Measurements for Paris",
        "description": (
            "This dataset contains CO2 measurements from high-cost and mid-cost "
            "stations in Paris."
        ),
        "source": "High-cost and mid-cost CO2 measurement stations",
        "date_created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "creator": "Robert Maiwald",
        "creator_email": "Robert.Maiwald@uni-heidelberg.de",
    }
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
    high_cost = process_files(data_path, measurement_type, read_function)
    # Add instrument name
    high_cost["instrument"] = (
        ["station"],
        ["Picarro" for code in high_cost.code.values],
    )
    return high_cost


def process_mid_cost() -> xr.Dataset:
    data_path = TRACER_PATH / "6_2_2_mid-cost/2025_version/"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Mid-cost CO2 measurement data path does not exist: {data_path}"
        )
    measurement_type = "mid-cost"
    read_function = read_co2_measurement_mid_cost
    mid_cost = process_files(data_path, measurement_type, read_function)
    # Update the metadata with the correct position and additional data
    mid_cost = update_metadata(data_path, mid_cost)
    return mid_cost


def update_metadata(data_path, mid_cost):
    df = pd.read_csv(
        data_path.parent / "co2_metadata.csv",
        index_col=0,
        comment="#",
    )
    df["Name"] = df["Name"].astype(str)
    # print(df)
    found = False
    indexes = []
    for code in mid_cost.code.values:
        # Check if the code string matches any part of an index string and raise
        # an error if it does not match
        for index in df.index:
            found = False
            if code in index + df.loc[index, "Name"]:
                # print(
                #     f"Warning: Code {code} found in metadata index {index}, "
                #     "but not as a standalone code."
                # )
                found = True
                indexes.append(index)
                break
        if not found:
            raise ValueError(
                f"Code {code} not found in metadata. Please check the metadata file."
            )
    df = df.drop("Name", axis=1)
    for column in df.columns:
        mid_cost[column] = (
            ["station"],
            df.loc[indexes, column].values,
        )
    return mid_cost


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
        code, name, lat, lon, alt, height = read_location(file, measurement_type)
        print(code, height)
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
                    "code": (["station"], [code]),
                    "name": (["station"], [name]),
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
    code, name, lat, lon, elevation, height = None, None, None, None, None, None
    for line in lines:
        param = line.split(":")[0]
        if param == "# STATION CODE":
            code = line.split(":")[1].split()[0]
        elif param == "# STATION NAME":
            name = line.split(":")[1].strip()
        elif param == "# LATITUDE":
            lat = float(line.split(":")[1].split()[0])
        elif param == "# LONGITUDE":
            lon = float(line.split(":")[1].split()[0])
        elif param == "# ALTITUDE":
            elevation = float(line.split(":")[1].split()[0])
        elif param == "# SAMPLING HEIGHTS":
            height = float(line.split(":")[1].split()[0])
    if None in (code, name, lat, lon, elevation, height):
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
        {
            "code": code,
            "name": name,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "height": height,
        }
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
