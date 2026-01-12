import numpy as np
import xarray as xr
import logging

import paris_2025 as p


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
    # Get main wind direction
    mean_u_wind, mean_v_wind, mean_wind_speed, mean_wind_direction = (
        p.meteo.get_mean_wind_vars()
    )
    # Normalize mean_u_wind and mean_v_wind
    normalized_mean_u_wind = mean_u_wind / mean_wind_speed
    normalized_mean_v_wind = mean_v_wind / mean_wind_speed
    return normalized_mean_u_wind, normalized_mean_v_wind


def get_background_co2() -> xr.Dataset:
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
    distances = distances.where(background_co2["co2"].notnull())

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
    return dynamic_background
