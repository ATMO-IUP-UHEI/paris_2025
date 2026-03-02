# import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# import geopandas as gpd
import pyproj

# import matplotlib.pyplot as plt
# import matplotlib as mpl
import xarray as xr

import paris_2025 as p
from paris_2025.config import CONFIG

# import shapely
# from tqdm import tqdm


TRACER_PATH = Path(CONFIG["data_path"]) / "6_measurements/6_2_tracers/"
print(f"Using tracer path: {TRACER_PATH}")
TRACER_CO2_FILE = TRACER_PATH / "co2.nc"


def get_co2_measurements() -> xr.Dataset:
    """
    Load the CO2 measurement dataset from file, if it exists and is not empty.

    .. deprecated::
        Use ``ggpymanager.load("co2_measurements", CONFIG)`` instead.
        This function will be removed once all callers have been migrated.

    Returns
    -------
    xr.Dataset
        The CO2 measurement dataset.
    """
    warnings.warn(
        "p.tracers.get_co2_measurements() is deprecated. "
        "Use ggp.load('co2_measurements', CONFIG) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not TRACER_CO2_FILE.exists():
        raise FileNotFoundError(
            "CO2 measurement file not found. Please run the script to create it "
            "tracers.create_co2_measurement()"
        )
    co2 = xr.load_dataset(TRACER_CO2_FILE)
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
    co2 = xr.concat([high_cost, mid_cost], dim="station", join="outer")
    # Add coordinates in GRAL projection
    x, y = pyproj.Proj(CONFIG["domain"]["crs"])(
        co2.longitude.values, co2.latitude.values
    )
    co2["x"] = (["station"], x)
    co2["y"] = (["station"], y)

    # Add flag for GRAMM and GRAL domains
    co2 = co2.assign_coords(
        in_gramm_domain=("station", p.domain.checking_domain("gramm", x, y)),
        in_gral_domain=("station", p.domain.checking_domain("gral", x, y)),
    )

    # Add labels to the station dimension
    station_indices = []
    for code, height, instrument in zip(
        co2.code.values, co2.height.values, co2.instrument.values
    ):
        station_index = "{}_{:.0f}".format(code[:3], height)
        if station_index in station_indices:
            station_index = f"{station_index}_{instrument}"
        station_indices.append(station_index)
    co2["station"] = station_indices

    # Set coordinates
    co2 = co2.set_coords(
        [
            "x",
            "y",
            "time",
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

    # Add attributes [unit, long_name, standard_name, description]
    attrs = {
        "station": [
            None,
            "Station code and height",
            "station",
            "Unique identifier for each CO2 measurement station, combining station "
            "code and sampling height.",
        ],
        "co2": ["ppm", "CO2 concentration", "co2", np.nan],
        "stdev": ["ppm", "Standard deviation", "stdev", np.nan],
        "nbpoints": ["1", "Number of points", "nbpoints", np.nan],
        "flag": [
            None,
            "Flag",
            "flag",
            "Flag indicating the quality of the measurement. 'O' for valid, 'U' "
            "for uncertain, other values indicate invalid measurements.",
        ],
        "altitude": [
            "m asl",
            "Altitude",
            "altitude",
            "Altitude of the station above sea level in meters.",
        ],
        "height": [
            "m agl",
            "Height",
            "height",
            "Height of the station above ground level in meters.",
        ],
        "x": ["m", f"X coordinate in GRAL projection: {CONFIG['domain']['crs']}", "x"],
        "y": ["m", f"Y coordinate in GRAL projection: {CONFIG['domain']['crs']}", "y"],
        "in_gramm_domain": [
            "boolean",
            "Station in gramm domain",
            "in_gramm_domain",
            "True if the station is in the gramm domain, False otherwise",
        ],
        "in_gral_domain": [
            "boolean",
            "Station in gral domain",
            "in_gral_domain",
            "True if the station is in the gral domain, False otherwise",
        ],
        "latitude": [
            "degrees_north",
            "Latitude of the station",
            "latitude",
            "Latitude of the meteorological station in degrees north",
        ],
        "longitude": [
            "degrees_east",
            "Longitude of the station",
            "longitude",
            "Longitude of the meteorological station in degrees east",
        ],
        "code": [
            None,
            "Station code",
            "code",
            "Short code for the CO2 measurement station",
        ],
        "name": [
            None,
            "Station name",
            "name",
            "Name of the CO2 measurement station",
        ],
        "type": [
            None,
            "Measurement type",
            "type",
            "Type of CO2 measurement (high-cost or mid-cost)",
        ],
        "time": [
            "UTC",
            "Time",
            "time",
            "Time of measurement in UTC",
        ],
        "instrument": [
            None,
            "Instrument used for measurement",
            "instrument",
            "Identifier for the instrument used to measure CO2",
        ],
        "HPP_ID|K96_ID": [
            None,
            "HPP_ID or K96_ID",
            "HPP_ID|K96_ID",
            "Identifier for the HPP or K96 station",
        ],
        "box_id": [
            None,
            "Box ID",
            "box_id",
            "Identifier for the box containing the CO2 measurement instrument",
        ],
    }
    for var in attrs.keys():
        co2[var].attrs.update(
            {
                k: v
                for k, v in zip(
                    ["unit", "long_name", "standard_name", "description"], attrs[var]
                )
                if v is not None
            }
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
    data_path = TRACER_PATH / "6_2_1_high-cost/ICOS-CITIES-Paris-Tower-L2-2024/"
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
    # # Add instrument name
    # mid_cost["instrument"] = (
    #     ["station"],
    #     ["HPP_ID|K96_ID" for _ in mid_cost.code.values],
    # )
    return mid_cost


def update_metadata(data_path, mid_cost):
    file_path = data_path.parent / "co2_metadata.csv"
    if not file_path.exists():
        warnings.warn(
            f"Mid-cost CO2 measurement metadata file does not exist: {file_path}. "
            "Using default metadata from measurement files."
        )
        return mid_cost
    df = pd.read_csv(
        file_path,
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
    Process CO2 measurement files and return as a single xarray.Dataset.
    """
    xds_list = []
    if measurement_type == "high-cost":
        file_list = data_path.glob("*.CO2")
    elif measurement_type == "mid-cost":
        file_list = data_path.glob("*_co2.csv")
    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")
    file_list = sorted(file_list)
    print(f"Processing {len(file_list)} {measurement_type} files...")
    for file in file_list:
        code, name, lat, lon, alt, height = read_header(file, measurement_type)
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
    return xr.concat(xds_list, dim="station", join="outer")


def read_header(path: Path, measurement_type: str) -> pd.Series:
    """
    Read the location and metadata from the header of a measurement file.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    pd.Series
        Series with code, name, lat, lon, alt, height.
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
    with open(path) as f:
        columns_line = [l for l in f.readlines() if l.startswith("#")][-1]
    columns = columns_line.lstrip("# ").strip().split(";")
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
