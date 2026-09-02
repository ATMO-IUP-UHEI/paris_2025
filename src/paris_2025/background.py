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
    # Get GRAL domain centroid
    logging.info("Getting GRAL domain centroid...")
    centroid_x, centroid_y = p.domain.get_centroid_of_domain("gral")

    # Get CO2 measurements
    logging.info("Getting background CO2 measurements...")
    co2 = ggp.load("co2_measurements", CONFIG).load()

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

    # Loaded eagerly: reading the string variables lazily crashes the netCDF4
    # library once several lazy datasets of the same file are alive.
    co2 = ggp.load("co2_measurements", CONFIG).load()
    co2 = co2.where(co2.instrument == "Picarro")
    # co2 = co2.sel(station=co2.in_gral_domain)
    background_min = co2.co2.min("station")
    background_stations = co2.co2.idxmin("station")
    background_min["background_station"] = background_stations
    return background_min


def _fill_nearest_height_bin(
    values: xr.DataArray, stations: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Fill empty height bins from the nearest bin that has data.

    ``interpolate_na`` cannot be applied to the string station names, so
    interpolating the values alone would leave ``background_station`` empty for
    every filled bin. Instead the *source bin index* is interpolated (nearest,
    with extrapolation at both ends, exactly as before) and then used to gather
    both the CO2 values and the station names, so the reported station always
    is the one holding the reported value.

    Parameters
    ----------
    values : xarray.DataArray
        Binned CO2 values with a ``height_bins`` dimension.
    stations : xarray.DataArray
        Station names of the selected minimum, same shape as ``values``.

    Returns
    -------
    filled_values, filled_stations : xarray.DataArray
        Both arrays gap-filled along ``height_bins``. Timesteps without any
        data in any bin stay NaN in both.
    """
    positions = xr.DataArray(
        np.arange(values.sizes["height_bins"]),
        dims="height_bins",
        coords={"height_bins": values.height_bins},
    )
    source_index = positions.where(values.notnull()).interpolate_na(
        dim="height_bins",
        method="nearest",
        use_coordinate=False,
        fill_value="extrapolate",
    )
    # Drop height_bins from the indexer: it conflicts with the dimension
    # coordinate of the arrays being indexed. Restored after gathering.
    valid = source_index.notnull().drop_vars("height_bins")
    gather = source_index.fillna(0).astype(int).drop_vars("height_bins")

    filled = tuple(
        da.isel(height_bins=gather)
        .where(valid)
        .assign_coords(height_bins=values.height_bins)
        for da in (values, stations)
    )
    return filled  # type: ignore[return-value]


def get_binned_background_co2(
    bins: int | list[int] = [0, 40, 80, 120, 200]
) -> xr.DataArray:
    """
    Get background CO2 levels binned by height.

    Empty height bins are filled from the nearest bin that has data; the
    ``background_station`` coordinate is filled from the same bin, so it always
    names the station the CO2 value comes from.

    Parameters
    ----------
    bins : int | list[int]
        Number of bins or list of bin edges for height binning.

    Returns
    -------
    background_levels : xarray.DataArray
        Background CO2 levels corresponding to measurement heights, carrying a
        ``background_station`` coordinate with the selected station names.
    """

    # Loaded eagerly: the binning below groups by and interpolates along
    # coordinates that xarray cannot handle as chunked arrays.
    co2 = ggp.load("co2_measurements", CONFIG).load()
    picarro_co2 = co2.co2.where(co2.instrument == "Picarro")
    co2_height_bins = picarro_co2.groupby_bins("height", bins=bins).min()
    selected_stations = picarro_co2.groupby_bins("height", bins=bins).map(
        lambda x: x.idxmin("station")
    )
    co2_height_bins, selected_stations = _fill_nearest_height_bin(
        co2_height_bins, selected_stations
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
    * ``binned_background`` (time, station) — minimum CO2 across Picarro
      stations grouped by measurement height, pre-selected to the bin closest
      to each station's measurement height. Can be added directly to the model
      output without any additional height selection.
    * ``binned_background_by_label`` (time, height_bins) — same data as
      ``binned_background`` but indexed by string labels (e.g. ``"0-40m"``),
      carrying ``height_bin_left``, ``height_bin_right``, and
      ``height_bin_center`` as non-dimension coordinates.

    Each variable has a companion ``*_station`` string variable recording which
    station was selected (empty string when no station is available). The
    companion shares the dimensions of its parent variable, i.e.
    ``binned_background_station`` is (time, station) and
    ``binned_background_by_label_station`` is (time, height_bins).

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
    # Add a coordinate for the lower and upper bin edges and the center of the bin

    binned_co2.coords["height_bin_left"] = (
        "height_bins",
        [iv.left for iv in intervals],
    )
    binned_co2.coords["height_bin_right"] = (
        "height_bins",
        [iv.right for iv in intervals],
    )
    binned_co2.coords["height_bin_center"] = (
        "height_bins",
        [iv.mid for iv in intervals],
    )

    # Center-indexed intermediate: swap dim to height_bin_center for selection.
    binned_co2_by_center = binned_co2.swap_dims(
        {"height_bins": "height_bin_center"}
    ).drop_vars(["height_bins", "height_bin_left", "height_bin_right"])

    # Station-matched version: select bin closest to each station's measurement
    # height → dims (time, station), ready to add to the model output directly.
    # Rename height_bin_center to matched_height_bin_center to avoid a naming
    # conflict when assembling the Dataset (binned_background_by_label also
    # carries a height_bin_center coordinate, but indexed by height_bins).
    co2_stations = ggp.load("co2_measurements", CONFIG)
    binned_co2_by_station = binned_co2_by_center.sel(
        height_bin_center=co2_stations.height, method="nearest"
    ).rename({"height_bin_center": "matched_height_bin_center"})

    # ------------------------------------------------------------------
    # 3. Extract selected station names as plain string arrays
    # ------------------------------------------------------------------
    dynamic_station_str = _stations_to_str(dynamic.coords["background_station"].values)
    minimum_station_str = _stations_to_str(minimum.coords["background_station"].values)

    # Label-indexed station names, dims (time, height_bins) — companion of
    # binned_background_by_label.
    binned_station_raw = binned.coords["background_station"]
    binned_station_by_label = xr.DataArray(
        _stations_to_str(binned_station_raw.values),
        dims=binned_station_raw.dims,
        coords={"time": binned.time, "height_bins": bin_labels},
    ).transpose("time", "height_bins")

    # Station-matched station names, dims (time, station) — companion of
    # binned_background, selected with the same nearest-bin rule.
    binned_station_by_station = (
        binned_station_by_label.assign_coords(
            height_bin_center=("height_bins", [iv.mid for iv in intervals])
        )
        .swap_dims({"height_bins": "height_bin_center"})
        .drop_vars("height_bins")
        .sel(height_bin_center=co2_stations.height, method="nearest")
        .rename({"height_bin_center": "matched_height_bin_center"})
    )

    # ------------------------------------------------------------------
    # 4. Assemble Dataset
    # ------------------------------------------------------------------
    ds = xr.Dataset(
        {
            "dynamic_background": dynamic_co2,
            "minimum_background": minimum_co2,
            "binned_background": binned_co2_by_station,
            "binned_background_by_label": binned_co2,
            "dynamic_background_station": xr.DataArray(
                dynamic_station_str, dims=["time"], coords=time_coord
            ),
            "minimum_background_station": xr.DataArray(
                minimum_station_str, dims=["time"], coords=time_coord
            ),
            "binned_background_station": binned_station_by_station,
            "binned_background_by_label_station": binned_station_by_label,
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
        "long_name": "Height-binned background CO2 (station-matched)",
        "units": "ppm",
        "description": (
            "Minimum CO2 across Picarro stations grouped by measurement height bin, "
            "pre-selected to the bin closest to each station's measurement height. "
            "Dimensions: (time, station). Can be added to model output directly."
        ),
        "height_bin_edges_m_agl": str(bins),
    }
    ds["binned_background_by_label"].attrs = {
        "long_name": "Height-binned background CO2 (label-indexed)",
        "units": "ppm",
        "description": (
            "Minimum CO2 across Picarro stations grouped by measurement height bin. "
            "Indexed by string height_bins labels (e.g. '0-40m'). "
            "Carries height_bin_left, height_bin_right, height_bin_center as "
            "non-dimension coordinates."
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
        "long_name": "Height-binned background station (station-matched)",
        "description": (
            "Station with the minimum CO2 in the height bin closest to each "
            "station's measurement height, at each timestep. "
            "Dimensions: (time, station), matching binned_background."
        ),
    }
    ds["binned_background_by_label_station"].attrs = {
        "long_name": "Height-binned background station (label-indexed)",
        "description": (
            "Station with the minimum CO2 for each height bin and timestep. "
            "Dimensions: (time, height_bins), matching binned_background_by_label."
        ),
    }
    ds["binned_background"].coords["matched_height_bin_center"].attrs = {
        "long_name": "Matched height bin center",
        "units": "m agl",
        "description": (
            "Center of the height bin selected for each station, i.e. the bin "
            "whose center is closest to the station's measurement height."
        ),
    }
    ds["binned_background_by_label"].coords["height_bins"].attrs = {
        "long_name": "Measurement height bin",
        "units": "m agl",
        "description": "Height bin label (lower-upper bound above ground level).",
        "bin_edges_m_agl": str(bins),
    }
    ds["binned_background_by_label"].coords["height_bin_left"].attrs = {
        "long_name": "Height bin lower edge",
        "units": "m agl",
        "description": "Lower edge of the measurement height bin (above ground level).",
    }
    ds["binned_background_by_label"].coords["height_bin_right"].attrs = {
        "long_name": "Height bin upper edge",
        "units": "m agl",
        "description": "Upper edge of the measurement height bin (above ground level).",
    }
    ds["binned_background_by_label"].coords["height_bin_center"].attrs = {
        "long_name": "Height bin center",
        "units": "m agl",
        "description": (
            "Center of the measurement height bin (above ground level). "
            "Use with method='nearest' to select the closest bin for a given height."
        ),
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
