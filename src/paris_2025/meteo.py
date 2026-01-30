import logging
from pathlib import Path

import ggpymanager as ggpy
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

import paris_2025 as p
from paris_2025.config import CONFIG

METEO_PATH = Path(CONFIG["meteo_path"])
logging.info(f"Using meteorological path: {METEO_PATH}")
METEO_FILE = METEO_PATH / "meteo.nc"

METEO_SUBPATHS = {
    "MeteoFrance": "6_1_1_MeteoFrance/",
    "NCAR": "6_1_2_NCAR/",
    "QUALAIR": "6_1_3_Qualair/",
    "mid-cost": "6_1_4_mid-cost/",
    "lidar": "6_1_5_Lidar/",
    "high-cost": "6_1_6_crds_co-located/",
}


def get_meteo_measurements() -> xr.Dataset:
    """
    Load the meteorological measurement dataset from file, if it exists and is not
    empty.

    Returns
    -------
    xr.Dataset
        The meteorological measurement dataset.
    """
    if not METEO_FILE.exists():
        raise FileNotFoundError(
            "Meteorological measurement file not found. "
            "Please run the script to create it "
            "tracers.create_meteo_measurements()"
        )
    meteo = xr.open_dataset(METEO_FILE)
    if meteo.station.size == 0:
        raise ValueError(
            "Meteorological measurement file is empty. "
            "Please run the script to create it. "
            "tracers.create_meteo_measurements()"
        )
    return meteo


def get_mean_wind_vars(
    meteo: xr.Dataset | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Get the mean wind speed from the meteorological measurements.

    Parameters
    ----------
    meteo : xarray.Dataset | None
        The meteorological measurement dataset. If None, it will be loaded from file.

    Returns
    -------
    mean_u_wind : xarray.DataArray
        Mean u-wind component across all stations.
    mean_v_wind : xarray.DataArray
        Mean v-wind component across all stations.
    mean_wind_speed : xarray.DataArray
        Mean wind speed across all stations.
    mean_wind_direction : xarray.DataArray
        Mean wind direction across all stations.
    """
    if meteo is None:
        meteo = p.meteo.get_meteo_measurements()
    mean_u_wind = meteo.u_wind.mean("station")
    mean_v_wind = meteo.v_wind.mean("station")
    mean_wind_speed = ggpy.processing.wind_speed_from_vector(mean_u_wind, mean_v_wind)
    mean_wind_direction = ggpy.processing.direction_from_vector(
        mean_u_wind, mean_v_wind
    )
    return mean_u_wind, mean_v_wind, mean_wind_speed, mean_wind_direction


def create_meteo_measurements() -> None:
    if METEO_FILE.exists():
        logging.info(
            "Meteo measurement file already exists. Please delete it to recreate it."
        )
        return
    METEO_FILE.parent.mkdir(parents=True, exist_ok=True)
    meteo = xr.concat(
        [process_meteo_measurements(source) for source in METEO_SUBPATHS],
        dim="station",
        join="outer",
    )
    # Add coordinates in GRAL projection
    x, y = pyproj.Proj(CONFIG["domain"]["crs"])(
        meteo.longitude.values, meteo.latitude.values
    )
    meteo = meteo.assign_coords(x=("station", x), y=("station", y))

    # Add flag for GRAMM and GRAL domains
    meteo = meteo.assign_coords(
        in_gramm_domain=("station", p.domain.checking_domain("gramm", x, y)),
        in_gral_domain=("station", p.domain.checking_domain("gral", x, y)),
    )

    # Add wind as vector
    meteo["u_wind"], meteo["v_wind"] = ggpy.processing.vector_from_direction_and_speed(
        meteo["wind_direction"], meteo["wind_speed"]
    )

    # Add attributes [unit, long_name, standard_name, description]
    attrs = {
        "time": [
            "UTC",
            "Time",
            "time",
            "Time of measurement in UTC",
        ],
        "station": [
            None,
            "Station name",
            "station",
            "Name of the meteorological station",
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
        "altitude": [
            "m",
            "Altitude of the station",
            "altitude",
            "Altitude of the meteorological station in meters above sea level",
        ],
        "operator": [
            None,
            "Operator of the meteorological station",
            "operator",
            "Operator of the meteorological station",
        ],
        "x": [
            "m",
            f"X coordinate in GRAL projection: {CONFIG['domain']['crs']}",
            "x",
        ],
        "y": [
            "m",
            f"Y coordinate in GRAL projection: {CONFIG['domain']['crs']}",
        ],
        "in_gramm_domain": [
            "boolean",
            "Station in gramm domain",
            "True if the station is in the gramm domain, False otherwise",
        ],
        "in_gral_domain": [
            "boolean",
            "Station in gral domain",
            "True if the station is in the gral domain, False otherwise",
        ],
        "temperature": [
            "K",
            "Air temperature",
            "air_temperature",
            "Air temperature in Kelvin",
        ],
        "pressure": ["Pa", "Air pressure", "air_pressure", "Air pressure in Pascals"],
        "relative_humidity": [
            "%",
            "Relative humidity",
            "relative_humidity",
            "Relative humidity in percentage",
        ],
        "wind_speed": [
            "m/s",
            "Wind speed",
            "wind_speed",
            "Wind speed in meters per second",
        ],
        "wind_direction": [
            "degrees",
            "Wind direction",
            "wind_direction",
            "Wind direction in degrees",
        ],
        "global_radiation": [
            "W/m^2",
            "Global radiation",
            "global_radiation",
            "Global radiation in Watts per square meter",
        ],
        "u_wind": [
            "m/s",
            "U component of wind",
            "eastward_wind",
            "U component of wind in meters per second",
        ],
        "v_wind": [
            "m/s",
            "V component of wind",
            "northward_wind",
            "V component of wind in meters per second",
        ],
    }
    for var in attrs.keys():
        meteo[var].attrs.update(
            {
                k: v
                for k, v in zip(
                    ["unit", "long_name", "standard_name", "description"], attrs[var]
                )
                if v is not None
            }
        )

    # Add global attributes
    meteo.attrs = {
        "title": "Meteorological Measurements in Paris",
        "description": (
            "This dataset contains meteorological measurements from various sources "
        ),
        "source": "MeteoFrance, NCAR, QUALAIR, mid-cost, high-cost",
        "date_created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "creator": "Robert Maiwald",
        "creator_email": "Robert.Maiwald@uni-heidelberg.de",
    }
    meteo.to_netcdf(METEO_FILE, mode="w", unlimited_dims="time")


def process_meteo_measurements(source: str) -> xr.Dataset:
    """
    Processes meteorological measurements from the specified source.

    Parameters:
        source (str): The name of the meteorological data source.

    Returns:
        xr.Dataset: Processed meteorological measurements.
    """
    logging.info(f"Processing meteorological measurements from {source}")
    funcs = {
        "MeteoFrance": process_meteofrance,
        "NCAR": process_ncar,
        "QUALAIR": process_qualair,
        "mid-cost": process_mid_cost,
        "lidar": process_lidar,
        "high-cost": process_high_cost,
    }
    return funcs[source]()


def is_variable_static_over_time(xds, var) -> bool:
    """
    Check if a variable in an xarray Dataset is static over the time dimension for all
    stations. This function iterates through all stations in the dataset and checks
    whether the specified variable has only one unique value across the time dimension
    (excluding NaNs). It handles cases where the time series might be entirely full of
    NaNs by immediately returning False in that scenario.

    Parameters
    ----------
    xds : xarray.Dataset
        The dataset containing the variable and dimensions 'time' and 'station'.
    var : str
        The name of the variable within the dataset to check.

    Returns
    -------
    bool
        True if the variable has exactly one unique value over time for every station
        (i.e., it is static). False if the variable changes over time for any station,
        or if any station's time series consists entirely of NaNs.
    """

    # If one other dimension only contains nans, skip it
    if xds[var].isnull().all("time").any():
        return False
    nunique = {
        s: xds[var].sel(station=s).to_series().nunique() for s in xds.station.values
    }
    if all(v == 1 for v in nunique.values()):
        return True
    else:
        return False


def squeeze_static_dims(xds: xr.Dataset) -> xr.Dataset:
    # If a variable does not change with time, convert it to a scalar
    for var in xds.data_vars:
        if "time" in xds[var].dims:
            # Check if the variable is static over time
            var_is_static = is_variable_static_over_time(xds, var)
            if var_is_static:
                xds[var] = xds[var].mean("time", keep_attrs=True)
    return xds


def test_dataset_structure(xds: xr.Dataset) -> None:
    """
    Tests the structure of the dataset to ensure it meets expected criteria.

    Parameters:
        xds (xr.Dataset): The dataset to test.
    """
    # Check that the dataset has a 'station' dimension
    assert "station" in xds.dims, "Dataset does not have a 'station' dimension."

    # Check that the dataset has a 'time' dimension
    assert "time" in xds.dims, "Dataset does not have a 'time' dimension."

    # Check that all variables have the 'station' and 'time' dimensions
    for var in xds.data_vars:
        assert (
            "station" in xds[var].dims
        ), f"Variable '{var}' does not have 'station' dimension."
        assert (
            "time" in xds[var].dims
        ), f"Variable '{var}' does not have 'time' dimension."
    # Check that the coordinates 'latitude', 'longitude', and 'altitude' are present
    assert "latitude" in xds.coords, "Dataset does not have 'latitude' coordinate."
    assert "longitude" in xds.coords, "Dataset does not have 'longitude' coordinate."
    assert "altitude" in xds.coords, "Dataset does not have 'altitude' coordinate."
    # Check that the 'time' coordinate is a datetime64 type
    assert xds["time"].dtype == np.dtype(
        "datetime64[ns]"
    ), "Time coordinate is not of type datetime64[ns]."
    # Check that the 'time' coordinate has a regular frequency
    assert np.all(
        xds["time"].diff("time") == np.timedelta64(1, "h")
    ), "Time index is not hourly."


def refine_dataset(xds: xr.Dataset, variable_names: dict, operator: str) -> xr.Dataset:
    """
    Refines the dataset by renaming variables, removing unused variables, and
    ensuring that the dataset is properly structured.
    Parameters:
        xds (xr.Dataset): The input dataset to refine.
        variable_names (dict): A dictionary mapping old variable names to new names.
    Returns:
        xr.Dataset: The refined dataset with renamed variables and proper structure.
    """
    # Only keep variables listed in variable_names, and rename them accordingly
    logging.info("Renaming and selecting variables")
    xds = xds.rename({k: v for k, v in variable_names.items() if k in xds.data_vars})
    xds = xds[[v for v in variable_names.values() if v in xds.data_vars]]
    xds = squeeze_static_dims(xds)

    # Move all variables with only the 'station' dimension to the 'station' coordinate
    for var in xds.data_vars:
        if len(xds[var].dims) == 1 and "station" in xds[var].dims:
            xds = xds.assign_coords({var: xds[var]})

    # Assign operator metadata
    xds = xds.assign_coords(
        operator=(("station",), np.repeat([operator], len(xds["station"])))
    )

    # Convert to each data variable to float32 to save space
    for var in xds.data_vars:
        xds[var] = xds[var].astype("float32")

    # Set time index to UTC (Not possible for netCDF files, as they are already in UTC)
    # xds = xds.assign_coords(time=xds["time"].to_index().tz_localize("UTC"))

    # Sort by time
    xds = xds.sortby("time")

    # Take hourly mean
    xds = xds.resample(time="1h").mean(dim="time", keep_attrs=True)

    # Ensure the dataset has the correct structure
    test_dataset_structure(xds)
    return xds


def process_meteofrance() -> xr.Dataset:
    """
    Processes meteorological measurements from MeteoFrance.

    Returns:
        xr.Dataset: Processed meteorological measurements from MeteoFrance.
    """
    data_path = (
        Path(CONFIG["data_path"])
        / "6_measurements/6_1_meteo"
        / METEO_SUBPATHS["MeteoFrance"]
    )
    file_list = list(data_path.glob("./H_??_*????-????.csv"))
    df = pd.concat(
        [pd.read_csv(file, index_col=0, sep=";") for file in tqdm(file_list)],
        ignore_index=True,
    )
    df["time"] = pd.to_datetime(df["AAAAMMJJHH"], format="%Y%m%d%H")
    mindex = pd.MultiIndex.from_arrays([df["NOM_USUEL"], df["time"]])
    xds = df.to_xarray()
    xds["index"] = mindex
    xds = xds.unstack("index")
    # Rename the index to 'station'
    xds = xds.rename({"NOM_USUEL": "station"})
    xds["station"] = xds["station"].astype("str")

    # Convert global radiation to W/m^2
    xds["GLO"] = xds["GLO"] / 3600 * 10000  # J cm-2 to W/m^2

    # Convert temperature to Kelvin
    xds["T"] = xds["T"] + 273.15  # C to K
    # Convert pressure to Pa
    xds["PSTAT"] = xds["PSTAT"] * 100  # hPa to Pa

    variable_names = {
        "LAT": "latitude",
        "LON": "longitude",
        "ALTI": "altitude",
        "DD": "wind_direction",
        "FF": "wind_speed",
        "T": "temperature",
        "PSTAT": "pressure",
        "GLO": "global_radiation",  # Global radiation in W/m^2
    }
    xds = refine_dataset(xds, variable_names, operator="MeteoFrance")
    return xds


def is_valid(string, index):
    if string[index] != " ":
        return True
    else:
        if string[index - 1] != " " and string[index + 1] != " ":
            return True
    return False


def read_txt_file(file):
    with open(file, "r") as f:
        lines = f.readlines()[1:3]
    names = []
    pre_label = ""
    label = ""
    for i in range(1, len(lines[1]) - 1):
        if is_valid(lines[0], i):
            label += lines[1][i]
            pre_label += lines[0][i]
        elif is_valid(lines[1], i):
            label += lines[1][i]
        elif len(label) > 0:
            if len(pre_label) > 0:
                names.append(pre_label + " " + label)
                pre_label = ""
            else:
                names.append(label)
            label = ""
    # If "-" in name, remove it and all whitespaces before and after
    names = [
        name.strip() if name.find("-") == -1 else name.replace("-", "").replace(" ", "")
        for name in names
    ]
    no_data = "-9999.9"
    df = pd.read_table(
        file,
        sep=" +",
        engine="python",
        header=None,
        skiprows=3,
        names=names,
        na_values=no_data,
    )
    # Rename columns to harmonized names
    to_rename = {
        "ELEV  (M)": "ELEVATION",
    }
    df = df.rename(columns=to_rename)

    # Check that wind speed and wind direction are included
    if "WDIR" not in df.columns:
        logging.warning(f"WDIR not in columns of file {file}: {df.columns}")
    if "WSPD" not in df.columns:
        logging.warning(f"WSPD not in columns of file {file}: {df.columns}")
    return df


def ncar_exemptions(xds: xr.Dataset) -> xr.Dataset:
    """
    Apply exemptions for known issues in NCAR data.

    Parameters:
        xds (xr.Dataset): The input dataset to apply exemptions to.

    Returns:
        xr.Dataset: The dataset with exemptions applied.
    """
    position_vars = [
        "LATITUDE",
        "LONGITUDE",
        "ELEVATION",
    ]
    datasets: list[xr.Dataset] = []
    for s in xds.station.values:
        sub_ds = xds.sel(station=s)
        problem_with_position = False

        for var in position_vars:
            if sub_ds[var].to_series().nunique() > 1:
                problem_with_position = True
                unique_values = sub_ds[var].to_series().unique()
                logging.warning(
                    f"Station {s} has varying {var} values."
                    f" Unique values: {unique_values}"
                )

        if problem_with_position:
            # Find all unique combinations of the position variables
            pos_df = sub_ds[position_vars].to_dataframe()
            # Drop NaN values before finding unique combinations as we only want valid
            # positions
            unique_positions = pos_df.dropna().drop_duplicates()

            for i, (_, row) in enumerate(unique_positions.iterrows()):
                # Create mask for this position
                mask = (
                    (sub_ds["LATITUDE"] == row["LATITUDE"])
                    & (sub_ds["LONGITUDE"] == row["LONGITUDE"])
                    & (sub_ds["ELEVATION"] == row["ELEVATION"])
                )

                # Apply mask
                new_sub_ds = sub_ds.where(mask, drop=True)

                # Rename station
                new_station_name = f"{s}_{i+1}"
                new_sub_ds = new_sub_ds.assign_coords(station=new_station_name)

                datasets.append(new_sub_ds)
        else:
            datasets.append(sub_ds)

    if datasets:
        return xr.concat(datasets, dim="station", join="outer")
    return xds


def process_ncar() -> xr.Dataset:
    """
    Processes meteorological measurements from NCAR.

    Returns:
        xr.Dataset: Processed meteorological measurements from NCAR.
    """
    logging.info("Starting NCAR data processing")
    data_path = METEO_PATH / METEO_SUBPATHS["NCAR"]
    files = list(data_path.glob("ncar_????/downloads/data/*.txt"))
    logging.info(f"Found {len(files)} NCAR files to process")
    # Check if there are duplicate files
    file_names = [file.name for file in files]
    if len(file_names) != len(set(file_names)):
        raise ValueError("Duplicate files found in NCAR data.")

    logging.info("Reading NCAR text files")
    # dfs = []
    # for file in tqdm(files, desc="Reading files"):
    #     dfs.append(read_txt_file(file))

    dfs = Parallel(n_jobs=-1, verbose=10, batch_size=128)(  # type: ignore
        delayed(read_txt_file)(file) for file in files
    )
    df = pd.concat(dfs, ignore_index=True, copy=False)
    logging.info(f"Read {len(df)} records from NCAR files")
    # Convert YYYYMMDDHHMM to numpy datetime64
    logging.info("Converting timestamps and filtering METAR entries")
    df["time"] = pd.to_datetime(df["REPORT TIME YYYYMMDDHHMM"], format="%Y%m%d%H%M")
    df = df.drop(columns=["REPORT TIME YYYYMMDDHHMM"])
    # Select only 'OBS TYPE' 'METAR' entries
    df = df[df["OBS TYPE"] == "METAR"]
    logging.info(f"Filtered to {len(df)} METAR entries")
    xds = df.to_xarray()
    # Find unique stations
    station_grouper = xds.groupby(
        ["STATION BBSSS", "LATITUDE", "LONGITUDE", "ELEVATION"]
    )
    logging.info(f"Found {len(station_grouper.groups)} unique stations in NCAR data")
    for group in station_grouper:
        logging.info(
            f"Station: "
            f"{group[0][0]}, Lat: {group[0][1]}, Lon: {group[0][2]}, Alt: {group[0][3]}"
        )
    mindex = pd.MultiIndex.from_arrays(
        [
            df[var]
            for var in [
                "STATION BBSSS",
                "time",
            ]
        ],
    )
    xds["index"] = mindex
    # Find duplicated indices and check that data is also duplicated
    logging.info("Removing duplicate entries and restructuring dataset")
    mask = xds.index.to_pandas().duplicated(keep=False)
    duplicated_entries = xds.sel(index=mask.values)
    duplicated_entries = duplicated_entries.sortby("index")
    for idx in tqdm(
        duplicated_entries.index,
        total=len(duplicated_entries.index),
        desc="Checking duplicates",
    ):
        df = duplicated_entries.sel(index=idx).to_dataframe()  # .dropna(how="all")
        # Test if all rows in df are identical
        if (df.nunique() > 1).all():
            logging.warning(f"Duplicated index {idx} has differing data:\n{df}")
    # Drop duplicated indices
    mask = xds.index.to_pandas().duplicated(keep="first")
    xds = xds.sel(index=~mask.values)
    xds = xds.unstack("index")
    # Rename the index to 'station'
    xds = xds.rename({"STATION BBSSS": "station"})
    xds["station"] = xds["station"].astype("str")
    logging.info(f"Dataset contains {len(xds.station)} stations")

    # Add altimeter setting to the pressure variable
    # xds["PRES"] = xds["PRES"] + xds["ALSE"]

    logging.info("Applying NCAR exemptions")
    xds = ncar_exemptions(xds)
    assert len(xds.station) == len(
        station_grouper.groups
    ), "Number of stations in dataset does not match number of unique stations."

    logging.info("Refining dataset with standardized variable names")
    variable_names = {
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "ELEVATION": "altitude",
        "ALSE": "pressure",
        "TMDB": "temperature",
        "WDIR": "wind_direction",
        "WSPD": "wind_speed",
    }
    xds = refine_dataset(xds, variable_names, operator="NCAR")

    logging.info("NCAR data processing completed successfully")
    return xds


def convert_to_datetime(Yr, Mo, Dy, Hr, Mn, Sd):
    """
    Convert Yr, Mo, Dy, Hr, Mn, Sd to a pandas datetime
    """
    return pd.to_datetime(
        {"year": Yr, "month": Mo, "day": Dy, "hour": Hr, "minute": Mn, "second": Sd}
    )


def read_qualair_file(file_path):
    """
    Read a Qualair file and return a pandas dataframe
    """
    with open(file_path, "r") as f:
        data = f.read().strip()
    # Replace ";" by ","
    data = data.replace(";", ",")
    # Remove "0D," on all lines
    data = data.replace("\n0D,", "\n")
    # Split in lines
    lines = data.split("\n")
    # Read column names
    columns = lines[0].split(",")
    columns = [c.split("=")[0] for c in columns if "=" in c]
    # Create pandas dataframe
    n_lines = len(lines)
    df = pd.DataFrame(index=range(n_lines), columns=columns)
    for i, line in enumerate(lines):
        # Split by "=" and remove last character which is the unit indicator
        values = [field.split("=")[1][:-1] for field in line.split(",") if "=" in field]
        if len(values) == len(columns):
            df.loc[i] = values
        else:
            # print(f"Line {i} has {len(values)} values instead of {len(columns)}")
            # print(line)
            pass
    # Drop rows with missing values
    df = df.dropna()
    # Convert to datetime from Yr	Mo	Dy	Hr	Mn	Sd
    df["time"] = convert_to_datetime(
        df["Yr"].astype(int),
        df["Mo"].astype(int),
        df["Dy"].astype(int),
        df["Hr"].astype(int),
        df["Mn"].astype(int),
        df["Sd"].astype(int),
    )
    df = df.drop(columns=["Yr", "Mo", "Dy", "Hr", "Mn", "Sd"])
    # Make datetime the index
    df.set_index("time", inplace=True)
    # Make all columns numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def process_qualair() -> xr.Dataset:
    """
    Processes meteorological measurements from Qualair.

    Returns:
        xr.Dataset: Processed meteorological measurements from Qualair.
    """
    data_path = METEO_PATH / METEO_SUBPATHS["QUALAIR"]
    files = list(data_path.glob("QUALAIR/meteo_QUALAIR_202?/*.dat"))
    df = pd.concat(
        [read_qualair_file(file) for file in tqdm(files, desc="Reading files")],
    )
    df = df.sort_index()
    xds = df.to_xarray()
    # Drop data with wind speeds over 40 m/s
    mask = xds.Sm > 40
    xds = xds.sel(time=~mask)
    # Add station information
    xds = xds.expand_dims("station")
    xds["station"] = ["QUALAIR"]
    xds["latitude"] = (("station"), [48.8464])
    xds["longitude"] = (("station"), [2.3561])
    xds["altitude"] = (("station"), [75])

    variable_names = {
        "latitude": "latitude",
        "longitude": "longitude",
        "altitude": "altitude",
        "Dm": "wind_direction",
        "Sm": "wind_speed",
    }
    xds = refine_dataset(xds, variable_names, operator="QUALAIR")
    return xds


def read_mid_cost_files(dir_path: Path) -> pd.DataFrame:
    """
    Read mid-cost files and extract relevant information.
    Parameters:
        dir_path (Path): Path to the directory containing mid-cost files.
    Returns:
        pd.DataFrame: Concatenated DataFrame of all files.
        str: Name of the station.
        float: Altitude of the station.
        float: Latitude of the station.
        float: Longitude of the station.
    """
    file_list = sorted(list(dir_path.glob("*.csv")))
    location_df = pd.DataFrame(
        columns=[
            "name",
            "location",
            "altitude",
            "latitude",
            "longitude",
            "number of satellites",
        ]
    )
    name = ""
    location = ""
    alt = 0.0
    lat = 0.0
    lon = 0.0
    number_of_satellites = 0
    # Read location information from each file
    for file in file_list:
        name = file.parent.name
        with open(file) as f:
            while True:
                line = f.readline()
                if line.startswith("date GPS : "):
                    parts = line.split(",")
                    alt = float(parts[1].split(":")[1])
                    lon = float(parts[2].split(":")[1])
                    lat = float(line.split(",")[3].split(" ")[2])
                    number_of_satellites = int(line.split(",")[4].split(":")[1])
                if line.startswith("location"):
                    location = line.split(":")[1].strip()
                    break
        location_df.loc[len(location_df)] = pd.Series(
            {
                "name": name,
                "location": location,
                "altitude": alt,
                "latitude": lat,
                "longitude": lon,
                "number of satellites": number_of_satellites,
            }
        )
    # Select position from row with the most satellites
    location_df = location_df.sort_values(
        by="number of satellites", ascending=False
    ).iloc[0]
    name = location_df["name"]
    location = location_df["location"]
    alt = location_df["altitude"]
    lat = location_df["latitude"]
    lon = location_df["longitude"]

    # Concatenate all files
    dfs = []
    for file in file_list:
        with open(file) as f:
            line = f.readline()
            while True:
                line = f.readline()
                if line.startswith("---"):
                    f.readline()
                    break
            file_df = pd.read_csv(f, sep=";", index_col=0)
        file_df.index = pd.to_datetime(file_df.index, format="%Y-%m-%d %H:%M:%S")
        dfs.append(file_df)
    df = pd.concat(dfs)
    # Sort by index
    df = df.sort_index()
    # Resample to hourly mean
    df = df.resample("h").mean()
    df["name"] = name
    df["location"] = location
    df["altitude"] = alt
    df["latitude"] = lat
    df["longitude"] = lon
    df = df.reset_index().rename(columns={"Time_sys": "time"})
    return df


def process_mid_cost() -> xr.Dataset:
    """
    Processes meteorological measurements from mid-cost.
    Returns:
        xr.Dataset: Processed meteorological measurements from mid-cost.
    """
    data_path = METEO_PATH / METEO_SUBPATHS["mid-cost"]
    directories = list(data_path.glob("???/"))
    df = pd.concat(
        [read_mid_cost_files(dir_path) for dir_path in tqdm(directories)],
        ignore_index=True,
    )

    mindex = pd.MultiIndex.from_arrays(
        [
            df[var]
            for var in [
                "name",
                "time",
            ]
        ],
    )
    xds = df.to_xarray()
    xds["index"] = mindex
    xds = xds.unstack("index")
    # Rename the index to 'station'
    xds = xds.rename({"name": "station"})
    xds["station"] = xds["station"].astype("str")

    # Set wrong readouts to NaN if the measurement stays constant for more than 1 hour
    mask = (xds["vit_anemo"] - xds["vit_anemo"].shift(time=1) == 0) | (
        xds["dir_anemo"] - xds["dir_anemo"].shift(time=1) == 0
    )
    xds["vit_anemo"] = xds["vit_anemo"].where(~mask)
    xds["dir_anemo"] = xds["dir_anemo"].where(~mask)

    variable_names = {
        "latitude": "latitude",
        "longitude": "longitude",
        "altitude": "altitude",
        "dir_anemo": "wind_direction",
        "vit_anemo": "wind_speed",
    }
    xds = refine_dataset(xds, variable_names, operator="mid-cost")

    return xds


def process_lidar() -> xr.Dataset:
    """
    Processes meteorological measurements from lidar.
    Returns:
        xr.Dataset: Processed meteorological measurements from mid-cost.
    """
    data_path = METEO_PATH / METEO_SUBPATHS["lidar"]
    path_list = sorted(list(data_path.glob("paris_dwl_L3V1.39_202*/*.nc")))
    # Open the files and combine them for faster access
    file_name = data_path / "paris_dwl_L3V1.39_all.nc"
    if not file_name.exists():
        xds = xr.open_mfdataset(
            path_list,
            # parallel=True,
        )
        xds.to_netcdf(file_name)
    else:
        xds = xr.open_dataset(file_name)
    # Drop measurement locations above 1000 m altitude
    xds = xds.sel(altitude=slice(None, 1000))
    xds_stacked = xds.stack(measurement_location=("station", "altitude"))
    # Drop all measurement locations which are below the measurement altitude and are
    # always NaN
    xds_stacked = xds_stacked.where(
        xds_stacked.station_altitude.mean("time") <= xds_stacked.altitude, drop=True
    )
    stations = xds_stacked.station.values
    altitudes = xds_stacked.altitude.values
    xds_stacked = xds_stacked.drop_vars(["measurement_location", "station", "altitude"])
    xds_stacked = xds_stacked.assign_coords(
        measurement_location=[
            f"{station}_{altitude}m" for station, altitude in zip(stations, altitudes)
        ]
    )
    xds_stacked["altitude"] = ("station"), altitudes
    xds = xds_stacked.rename(measurement_location="station")
    variable_names = {
        "station_lat": "latitude",
        "station_lon": "longitude",
        "altitude": "altitude",
        "wd": "wind_direction",
        "ws": "wind_speed",
    }
    xds = refine_dataset(xds, variable_names, operator="lidar")
    return xds


def read_header(file) -> dict:
    """
    Read the header of a file and return the header as a dictionary
    """
    with open(file, "r") as f:
        lines = f.readlines()
    # Create a dictionary with the header
    header = {}
    for line in lines:
        if not line.startswith("#"):
            break
        data = line.lstrip("# ").rstrip("\n").split(":")
        if len(data) == 2:
            key = data[0].strip()
            value = data[1].strip()
            header[key] = value
    return header


def read_column_names(file) -> list:
    """
    Read the column names of a file and return them as a list
    """
    with open(file, "r") as f:
        lines = f.readlines()
    # Create a list with the column names
    columns = []
    for line in lines:
        if line.startswith("# "):
            continue
        if len(line) < 3:
            continue
        if line.startswith("#"):
            line = line.lstrip("# ")
            data = line.split(";")
            columns = data
        break
    return columns


def to_unitless_float(string):
    """
    Convert a string to a float and remove the unit
    """
    return float(string.split(" ")[0])


def convert_to_datetime_high_cost(df):
    return pd.to_datetime(
        (
            df["Year"].astype(str)
            + df["Month"].astype(str)
            + df["Day"].astype(str)
            + "-"
            + df["Hour"].astype(str)
            + df["Minute"].astype(str)
        ),
        format="%Y%m%d-%H%M",
    )


def read_crds_file(file) -> pd.DataFrame:
    """
    Read a CRDS file and return a pandas dataframe
    """
    header = read_header(file)
    columns = read_column_names(file)
    df = pd.read_csv(
        file,
        sep=";",
        header=None,
        names=columns,
        comment="#",
        na_values=["-999.990"],
        low_memory=False,
    )
    # Convert time to datetime in UTC
    df["time"] = convert_to_datetime_high_cost(df)
    # Set time as index
    if header["TIME ZONE"] == "UTC +1":
        df["time"] = df["time"] - pd.Timedelta(hours=1)
    df = df.set_index("time")
    # Select only quality controlled data
    relevant_columns = {
        "AP": "pressure",
        "AT": "temperature",
        "RH": "relative humidity",
        "WS": "speed",
        "WD": "direction",
    }
    # units = {
    #     "AP": "hPa",
    #     "AT": "°C",
    #     "RH": "%",
    #     "WS": "m/s",
    #     "WD": "°",
    # }
    for var in relevant_columns:
        df = df[df[f"{var}-Flag"] == "O"]  # Only keep rows with quality flag 'O'
    df = df[relevant_columns.keys()]
    # Take hourly mean
    df = df.resample("h").mean().reset_index()

    # Add station information
    df["latitude"] = to_unitless_float(header["LATITUDE"])
    df["longitude"] = to_unitless_float(header["LONGITUDE"])
    df["altitude"] = to_unitless_float(header["ALTITUDE"])
    df["station"] = header["STATION NAME"]
    return df


def process_high_cost() -> xr.Dataset:
    """
    Processes meteorological measurements from CRDS co-located stations.
    Returns:
    -------
    xr.Dataset
        The processed dataset.
    """
    data_path = (
        Path(CONFIG["data_path"])
        / "6_measurements/6_1_meteo"
        / METEO_SUBPATHS["high-cost"]
    )
    file_list = sorted(list(data_path.glob("*.mto")))

    df = pd.concat(
        [read_crds_file(file) for file in tqdm(file_list)],
        ignore_index=True,
    )
    mindex = pd.MultiIndex.from_arrays(
        [
            df[var]
            for var in [
                "station",
                "time",
            ]
        ],
    )
    xds = df.to_xarray()
    xds["index"] = mindex
    xds = xds.unstack("index")

    xds["station"] = xds["station"].astype("str")

    # Limit to the time period of interest
    xds = xds.sel(time=slice("2010", None))

    # Convert pressure to Pa
    xds["AP"] = xds["AP"] * 100  # hPa to Pa
    # Convert temperature to Kelvin
    xds["AT"] = xds["AT"] + 273.15  # °C to K

    variable_names = {
        "latitude": "latitude",
        "longitude": "longitude",
        "altitude": "altitude",
        "WD": "wind_direction",
        "WS": "wind_speed",
        "AP": "pressure",
        "AT": "temperature",
        "RH": "relative_humidity",
    }
    xds = refine_dataset(xds, variable_names, operator="high-cost")
    return xds
