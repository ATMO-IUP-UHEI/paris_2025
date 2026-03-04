"""Data loaders for plotting module.

This module contains loaders that call ggp.load() or p.* accessors and perform
preprocessing. Simple one-line ggp.load() calls remain inline in plot functions
to avoid over-abstraction.
"""

from functools import lru_cache
from pathlib import Path

import ggpymanager as ggp
import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

import paris_2025 as p
from paris_2025.config import CONFIG


def load_and_prepare_matching_data(
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = (
        Path(CONFIG["output_path"]) / ggp.config.MATCHING_LOSS_FILE_NAME
    ),
    source_groups_path: str | Path = Path(CONFIG["source_groups_path"]),
    n_sim_ids: int = 10,
):
    """Load and prepare data for matching methods analysis.

    Loads concentration, meteo, matching loss, and source groups data, then
    selects the top N matching simulations and performs unit conversion.

    Parameters
    ----------
    gral_concentration_path : str | Path
        Path to GRAL concentration data
    matching_loss_path : str | Path
        Path to matching loss data
    source_groups_path : str | Path
        Path to source groups data
    n_sim_ids : int, optional
        Number of top simulation IDs to select

    Returns
    -------
    tuple
        (gral_concentration, gral_meteo, matching_loss, source_groups, sim_ids)
    """
    # Load data
    gral_concentration = xr.load_dataset(gral_concentration_path)
    gral_concentration["concentration"] = ggp.utils.ugm3_to_ppm(
        gral_concentration["concentration"], "co2"
    )

    gral_meteo = p.model.get_gral_meteo_data()
    matching_loss = xr.open_dataset(matching_loss_path)
    source_groups = xr.open_mfdataset(source_groups_path)

    # Get top n simulation IDs
    ls = []
    ml = matching_loss.copy()
    for i in range(n_sim_ids):
        sim_ids = ml["matching_loss"].idxmin("sim_id")
        ls.append(sim_ids.astype(int))
        ml["matching_loss"].loc[dict(sim_id=sim_ids)] = np.nan
    sim_ids = xr.concat(ls, dim="ranking")

    return gral_concentration, gral_meteo, matching_loss, source_groups, sim_ids


def load_matching_analysis_data(
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = (
        Path(CONFIG["output_path"]) / ggp.config.MATCHING_LOSS_FILE_NAME
    ),
    loss_type: str = "rmse - filter: True",
    n_best: int = 25,
):
    """Load and prepare data for matching sensitivity analysis.

    Loads concentration and matching loss data, selects best simulations,
    and computes derived meteorological quantities.

    Parameters
    ----------
    gral_concentration_path : str | Path
        Path to GRAL concentration data
    matching_loss_path : str | Path
        Path to matching loss data
    loss_type : str
        Type of loss to use for selecting best simulations
    n_best : int
        Number of best simulations to select

    Returns
    -------
    tuple
        (concentration, loss, speed, direction, stab_class)
        All with best_sim_id dimension for comparing top N simulations
    """
    # Load data
    gral_concentration = xr.load_dataset(gral_concentration_path)
    gral_concentration["concentration"] = ggp.utils.ugm3_to_ppm(
        gral_concentration["concentration"], "co2"
    )

    matching_loss = xr.open_dataset(matching_loss_path)
    gral_meteo = p.model.get_gral_meteo_data()

    # Get top N simulation IDs
    sim_ids = ggp.analysis.get_sim_ids(
        matching_loss.matching_loss.sel(loss_type=loss_type), n_best=n_best
    )

    # Calculate derived quantities
    concentration = (
        gral_concentration.sum("source_group")
        .mean("station")
        .sel(sim_id=sim_ids)
        .concentration
    )
    loss = matching_loss.sel(loss_type=loss_type, sim_id=sim_ids).matching_loss

    avg = gral_meteo.sel(sim_id=sim_ids).mean("station")
    speed = ggp.processing.wind_speed_from_vector(avg.u, avg.v)
    direction = ggp.processing.direction_from_vector(avg.u, avg.v)
    stab_class = avg.stab_class

    return concentration, loss, speed, direction, stab_class


def load_flux_maps_data(
    cadastre_path: str | Path = (
        Path(CONFIG["domain"]["gral"]["conf_path"]) / "cadastre.dat"
    ),
    source_groups_path: str | Path = p.model_input.fluxes.SOURCE_GROUP_NETCDF_PATH,
    point_path: str | Path = (
        Path(CONFIG["domain"]["gral"]["conf_path"]) / "point.dat"
    ),
):
    """Load and prepare data for flux map plots.

    Loads cadastre and point source emissions, merges with source group types,
    and extracts GRAL domain configuration.

    Parameters
    ----------
    cadastre_path : str | Path
        Path to the GRAL cadastre.dat file
    source_groups_path : str | Path
        Path to the source groups NetCDF file
    point_path : str | Path
        Path to the GRAL point.dat file

    Returns
    -------
    tuple
        (cadastre_emissions, source_groups, point_da, GRAL)
    """
    GRAL = CONFIG["domain"]["gral"]["bbox"] | CONFIG["domain"]["gral"]
    source_groups = xr.open_dataset(source_groups_path)
    cadastre_emissions = ggp.io.readers.read_cadastre_file(cadastre_path, GRAL)
    cadastre_emissions["type"] = source_groups.sel(
        source_group=cadastre_emissions["source_group"]
    ).type.load()

    point_da = ggp.io.readers.read_point_file(point_path)["Emission [kg/h]"]
    point_da["type"] = source_groups.sel(
        source_group=point_da["source_group"].reset_coords(drop=True)
    ).type.load()

    return cadastre_emissions, source_groups, point_da, GRAL


@lru_cache(maxsize=1)
def cache_data():
    """Load and cache modeled/measured CO2 data with background.

    Loads concentration timeseries, applies masking for Origins.earth and TNO
    priors, sums across source types, and combines with background CO2 and
    measurements. Results are cached to avoid repeated disk access.

    Returns
    -------
    background : xr.DataArray
        Binned background CO2, dims: (time, height_bins, station)
    co2 : xr.DataArray
        Measured CO2, dims: (time, station)
    co2_model : xr.DataArray
        Modeled CO2 (background + model enhancement), dims: (time, height, station,
        prior)
    """
    loss_type = "rmse - filter: True"

    conc_series = ggp.load("concentration_timeseries", CONFIG)
    t = conc_series.type
    mask = (
        xr.concat(
            [
                t.str.contains("Origins.earth|VPRM"),
                t.str.contains("TNO|VPRM"),
            ],
            dim="prior",
        )
        .assign_coords({"prior": ["Origins.earth", "TNO"]})
        .compute()
    )
    time_series = (
        conc_series.co2_timeseries.sel(loss_type=loss_type).where(mask).sum("type")
    )
    time_series = time_series.where(time_series.loss_diff < 0.1).mean("best_sim_id")
    with ProgressBar():
        time_series = time_series.compute()
    background = (
        ggp.load("background_co2", CONFIG)["binned_background"]
        .sel(station=time_series.station)
        .load()
    )
    co2 = ggp.load("co2_measurements", CONFIG).co2.load()
    co2_model = background.reset_coords(drop=True) + time_series.reset_coords(
        names=["x", "y"], drop=True
    )

    return background, co2, co2_model


@lru_cache()
def load_sector_enhancement_data(
    loss_type: str = "rmse - filter: True",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load model sector enhancement, measured CO2 and background CO2.

    Loads the concentration time series retaining the ``type`` dimension so that
    individual sector contributions remain visible, converts units to ppm, and
    aligns everything to the availability mask of the measurements.

    Parameters
    ----------
    loss_type : str
        Loss-type selector passed to ``sel(loss_type=...)``.

    Returns
    -------
    model_enhancement : xr.DataArray
        Sector-resolved enhancement (dims: time, station, type), masked to
        measurement availability, in ppm.
    co2 : xr.DataArray
        Measured CO2 (dims: time, station), in ppm.
    background : xr.DataArray
        Binned background CO2, broadcast to station heights (dims: time, station).
    """
    co2 = ggp.load("co2_measurements", CONFIG).co2.load()
    measurements_available = co2.notnull()

    conc_series = ggp.load("concentration_timeseries", CONFIG)

    model_enhancement = (
        conc_series.co2_timeseries.sel(loss_type=loss_type)
        .where(measurements_available)
    )
    model_enhancement = model_enhancement.where(model_enhancement.loss_diff < 0.1).mean(
        "best_sim_id"
    )
    with ProgressBar():
        model_enhancement = model_enhancement.compute()

    background = (
        ggp.load("background_co2", CONFIG)["binned_background"]
        .sel(station=model_enhancement.station)
        .load()
    )
    background = background.where(measurements_available)

    return model_enhancement, co2, background
