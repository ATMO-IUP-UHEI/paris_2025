import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import ggpymanager as ggp
import paris_2025 as p
from paris_2025.config import CONFIG


def get_wind_direction():
    """
    Get the mean wind direction from the meteorological measurements.

    Returns
    -------
    normalized_mean_u_wind : xarray.DataArray
        Normalized mean u-wind component.
    normalized_mean_v_wind : xarray.DataArray
        Normalized mean v-wind component.
    """
    meteo = p.meteo.get_meteo_measurements()
    # Only select stations used for matching
    meteo = meteo.sel(station=list(CONFIG["matching"]["stations"].keys()))
    mean_u_wind, mean_v_wind, mean_wind_speed, _ = p.meteo.get_mean_wind_vars(meteo)
    # Normalize mean_u_wind and mean_v_wind
    normalized_mean_u_wind = mean_u_wind / mean_wind_speed
    normalized_mean_v_wind = mean_v_wind / mean_wind_speed
    return normalized_mean_u_wind, normalized_mean_v_wind


def get_dynamic_background_co2() -> xr.Dataset:
    """
    Get the background CO2 measurements.

    Returns
    -------
    dynamic_background : xarray.Dataset
        The dynamic background CO2 measurements.
    """
    # Get mean wind direction
    logging.info("Getting wind direction for background CO2 computation...")
    normalized_mean_u_wind, normalized_mean_v_wind = get_wind_direction()
    normalized_mean_u_wind = normalized_mean_u_wind
    normalized_mean_v_wind = normalized_mean_v_wind

    # Get GRAL domain centroid
    logging.info("Getting GRAL domain centroid...")
    centroid_x, centroid_y = p.domain.get_centroid_of_domain("gral")

    # Get CO2 measurements
    logging.info("Getting background CO2 measurements...")
    co2 = p.tracers.get_co2_measurements().load()

    logging.info("Filtering background CO2 stations...")
    is_high_cost = co2.instrument == "Picarro"
    background_co2 = co2.sel(station=is_high_cost & ~co2.in_gral_domain)
    # Filter out OVS_20 and GNS_36 due to local contamination
    # (Doc et al., 2024, ‘The Monitoring Network of Greenhouse Gas (CO2, CH4) in the
    # Paris’ Region’.)
    background_co2 = background_co2.sel(
        station=~background_co2.station.isin(["OVS_20", "GNS_36"])
    )

    logging.info("Computing normalized distances of background CO2 stations...")
    x = background_co2.x - centroid_x
    y = background_co2.y - centroid_y
    length = np.sqrt(x**2 + y**2)
    background_co2["normalized_distance_x"] = x / length
    background_co2["normalized_distance_y"] = y / length
    for dim in ["x", "y"]:
        background_co2[f"normalized_distance_{dim}"].attrs = {
            "long_name": f"Normalized distance in {dim}-direction",
            "units": "1",
            "description": (
                f"Normalized distance in {dim}-direction of the wind direction from "
                "the GRAL domain centroid to the station location."
            ),
        }

    # Compute closest station
    distances = np.sqrt(
        (background_co2["normalized_distance_x"] + normalized_mean_u_wind) ** 2
        + (background_co2["normalized_distance_y"] + normalized_mean_v_wind) ** 2
    )

    # Set distances to NaN if no co2 measurements are available
    logging.info("Selecting dynamic background CO2 stations...")
    distances = distances.where(background_co2["co2"].notnull())  # type: ignore

    dynamic_background_index = distances.idxmin(dim="station")
    hours_without_background = dynamic_background_index.isnull().sum().item()
    logging.info(
        f"Hours without dynamic background station: {hours_without_background}"
    )
    logging.info(
        f"Unique dynamic background stations: "
        f"{dynamic_background_index.to_pandas().unique()}"
    )

    # Create placeholder for nan-values
    placeholder = xr.full_like(background_co2.isel(station=0), fill_value=np.nan)
    for coord in placeholder.coords:
        if coord == "time":
            continue
        else:
            placeholder[coord] = np.full(placeholder[coord].shape, np.nan)

    selection = xr.concat(
        [background_co2, placeholder],
        dim="station",
    )
    dynamic_background = selection.sel(station=dynamic_background_index)

    # Rename station to selected_station
    dynamic_background = dynamic_background.rename({"station": "background_station"})
    return dynamic_background


def get_minimum_background_co2() -> xr.DataArray:
    """Get the minimum background CO2 levels across all stations in the GRAL domain.

    Returns
    -------
    background_min : xarray.DataArray
        Minimum background CO2 levels across all stations in the GRAL domain.
    """

    co2 = p.tracers.get_co2_measurements()
    co2 = co2.where(co2.instrument == "Picarro")
    # co2 = co2.sel(station=co2.in_gral_domain)
    background_min = co2.co2.min("station")
    background_stations = co2.co2.idxmin("station")
    background_min["background_station"] = background_stations
    return background_min


def get_binned_background_co2(
    bins: int | list[int] = [0, 40, 80, 120, 200]
) -> xr.DataArray:
    """
    Get background CO2 levels binned by height.

    Parameters
    ----------
    bins : int | list[int]
        Number of bins or list of bin edges for height binning.

    Returns
    -------
    background_levels : xarray.DataArray
        Background CO2 levels corresponding to measurement heights.
    """

    co2 = p.tracers.get_co2_measurements()
    co2_height_bins = (
        co2.co2.where(co2.instrument == "Picarro")
        .groupby_bins("height", bins=bins)
        .min()
    )
    co2_height_bins = co2_height_bins.interpolate_na(
        dim="height_bins",
        method="nearest",
        use_coordinate=False,
        fill_value="extrapolate",
    )
    selected_stations = (
        co2.co2.where(co2.instrument == "Picarro")
        .groupby_bins("height", bins=bins)
        .map(lambda x: x.idxmin("station"))
    )
    co2_height_bins = co2_height_bins.assign_coords(
        background_station=selected_stations
    )
    return co2_height_bins


def _stations_to_str(raw: np.ndarray) -> np.ndarray:
    """Convert a numpy array of station names (possibly containing float NaN) to str.

    Missing values (NaN / None) are replaced with an empty string so the array
    can be stored as a plain string variable in NetCDF.
    """
    flat = np.array(raw).ravel()
    result = np.array(
        ["" if pd.isna(v) else str(v) for v in flat],
        dtype=object,
    )
    return result.reshape(raw.shape)


def create_background_co2() -> None:
    """Compute all background CO2 methods and save them to a single NetCDF file.

    Creates ``output_path/background_co2.nc`` containing three variables:

    * ``dynamic_background`` (time) — CO2 from the upwind Picarro station
      nearest to the GRAL domain centroid at each timestep.
    * ``minimum_background`` (time) — minimum CO2 across all Picarro stations
      outside the GRAL domain at each timestep.
    * ``binned_background`` (time, height_bins) — minimum CO2 across Picarro
      stations grouped by measurement height, using the bin edges from
      ``config.yaml`` → ``background.height_bins``.

    Each variable has a companion ``*_station`` string variable recording which
    station was selected (empty string when no station is available).

    This function is idempotent: if the output file already exists it returns
    immediately without recomputing. Delete the file to force a recompute.

    Called from ``scripts/prepare_inputs.py`` after CO2 measurements exist.
    """
    output_file = Path(CONFIG["output_path"]) / ggp.config.BACKGROUND_CO2_FILE_NAME

    if output_file.exists():
        logging.info("Background CO2 file already exists: %s. Skipping.", output_file)
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    bins: list[int] = CONFIG["background"]["height_bins"]

    # ------------------------------------------------------------------
    # 1. Compute the three background methods
    # ------------------------------------------------------------------
    logging.info("Computing dynamic background CO2...")
    dynamic = get_dynamic_background_co2()

    logging.info("Computing minimum background CO2...")
    minimum = get_minimum_background_co2()

    logging.info("Computing height-binned background CO2 (bins=%s)...", bins)
    binned = get_binned_background_co2(bins=bins)

    # ------------------------------------------------------------------
    # 2. Extract CO2 values — drop all non-time coordinates so they do
    #    not collide when assembled into a single Dataset.
    # ------------------------------------------------------------------
    time_coord = {"time": dynamic.time}

    dynamic_co2 = dynamic["co2"].reset_coords(drop=True)
    minimum_co2 = minimum.reset_coords(drop=True)

    # Convert IntervalIndex height_bins → human-readable string labels
    intervals = binned.height_bins.values
    bin_labels = np.array([f"{int(iv.left)}-{int(iv.right)}m" for iv in intervals])
    binned_co2 = (
        binned.drop_vars("background_station")
        .reset_coords(drop=True)
        .assign_coords(height_bins=bin_labels)
    )

    # ------------------------------------------------------------------
    # 3. Extract selected station names as plain string arrays
    # ------------------------------------------------------------------
    dynamic_station_str = _stations_to_str(dynamic.coords["background_station"].values)
    minimum_station_str = _stations_to_str(minimum.coords["background_station"].values)
    binned_station_str = _stations_to_str(
        binned.coords["background_station"].assign_coords(height_bins=bin_labels).values
    )

    # ------------------------------------------------------------------
    # 4. Assemble Dataset
    # ------------------------------------------------------------------
    ds = xr.Dataset(
        {
            "dynamic_background": dynamic_co2,
            "minimum_background": minimum_co2,
            "binned_background": binned_co2,
            "dynamic_background_station": xr.DataArray(
                dynamic_station_str, dims=["time"], coords=time_coord
            ),
            "minimum_background_station": xr.DataArray(
                minimum_station_str, dims=["time"], coords=time_coord
            ),
            "binned_background_station": xr.DataArray(
                binned_station_str,
                dims=["time", "height_bins"],
                coords={"time": dynamic.time, "height_bins": bin_labels},
            ),
        }
    )

    # ------------------------------------------------------------------
    # 5. Attributes
    # ------------------------------------------------------------------
    ds["dynamic_background"].attrs = {
        "long_name": "Dynamic background CO2",
        "units": "ppm",
        "description": (
            "Background CO2 from the upwind Picarro station nearest to the "
            "GRAL domain centroid, selected independently at each timestep."
        ),
    }
    ds["minimum_background"].attrs = {
        "long_name": "Minimum background CO2",
        "units": "ppm",
        "description": (
            "Minimum CO2 across all Picarro stations outside the GRAL domain "
            "at each timestep."
        ),
    }
    ds["binned_background"].attrs = {
        "long_name": "Height-binned background CO2",
        "units": "ppm",
        "description": (
            "Minimum CO2 across Picarro stations grouped by measurement height bin."
        ),
        "height_bin_edges_m_agl": str(bins),
    }
    ds["dynamic_background_station"].attrs = {
        "long_name": "Dynamic background station",
        "description": (
            "Station selected as dynamic background at each timestep. "
            "Empty string when no background station is available."
        ),
    }
    ds["minimum_background_station"].attrs = {
        "long_name": "Minimum background station",
        "description": "Station with the minimum CO2 at each timestep.",
    }
    ds["binned_background_station"].attrs = {
        "long_name": "Height-binned background station",
        "description": (
            "Station with the minimum CO2 for each height bin and timestep."
        ),
    }
    ds["binned_background"].coords["height_bins"].attrs = {
        "long_name": "Measurement height bin",
        "units": "m agl",
        "description": "Height bin label (lower-upper bound above ground level).",
        "bin_edges_m_agl": str(bins),
    }

    ds.attrs = {
        "title": "Background CO2 estimates",
        "description": (
            "Pre-computed background CO2 using dynamic, minimum, and height-binned "
            "methods derived from Picarro tower measurements."
        ),
        "source": "paris_2025.background.create_background_co2()",
        "height_bin_edges_m_agl": str(bins),
    }

    # String variables (station names) may not pass the strict CF checker;
    ggp.io.writers.save_netcdf_with_cf_check(ds, output_file, ignore_tests=True)
