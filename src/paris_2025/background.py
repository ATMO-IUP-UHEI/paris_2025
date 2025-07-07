import numpy as np

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


def get_background_co2(year: str):
    """
    Get the dynamic background CO2 measurements for a given year.

    Parameters
    ----------
    year : str
        The year for which to get the background CO2 measurements, e.g., '2023'.

    Returns
    -------
    dynamic_background : xarray.Dataset
        The dynamic background CO2 measurements for the specified year.
    """
    if not isinstance(year, str):
        raise TypeError("Year must be a string representing the year, e.g., '2023'.")

    # Get mean wind direction
    normalized_mean_u_wind, normalized_mean_v_wind = get_wind_direction()
    normalized_mean_u_wind = normalized_mean_u_wind.sel(time=year)
    normalized_mean_v_wind = normalized_mean_v_wind.sel(time=year)

    # Get GRAL domain centroid
    centroid_x, centroid_y = p.domain.get_centroid_of_domain("gral")

    # Get background CO2 measurements
    co2 = p.tracers.get_co2_measurements()
    co2 = co2.sel(time=year)

    is_high_cost = co2.instrument == "Picarro"
    background_co2 = co2.sel(station=is_high_cost & ~co2.in_gral_domain)
    # Filter out OVS_20 and GNS_36 due to local contamination
    # (Doc et al., 2024, ‘The Monitoring Network of Greenhouse Gas (CO2, CH4) in the
    # Paris’ Region’.)
    background_co2 = background_co2.sel(
        station=~background_co2.station.isin(["OVS_20", "GNS_36"])
    )

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
    distances = distances.where(background_co2["co2"].notnull())

    dynamic_background_index = distances.idxmin(dim="station")
    dynamic_background = background_co2.sel(station=dynamic_background_index)
    return dynamic_background
