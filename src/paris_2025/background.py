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
            "binned_background": binned_co2_by_station,
            "binned_background_by_label": binned_co2,
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
        "long_name": "Height-binned background station",
        "description": (
            "Station with the minimum CO2 for each height bin and timestep."
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
