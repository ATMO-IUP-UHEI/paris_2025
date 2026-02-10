import logging

import numpy as np
import xarray as xr

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
