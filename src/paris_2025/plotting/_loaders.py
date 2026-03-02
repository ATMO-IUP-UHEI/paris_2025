"""Data loaders for plotting module.

This module contains loaders that call ggp.load() or p.* accessors and perform
preprocessing. Simple one-line ggp.load() calls remain inline in plot functions
to avoid over-abstraction.
"""

from pathlib import Path

import ggpymanager as ggp
import numpy as np
import xarray as xr

import paris_2025 as p
from paris_2025.config import CONFIG


def load_and_prepare_matching_data(
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"])
    / "co2.nc",
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
    gral_concentration["concentration"] = _mu_g_m3_to_ppm(
        gral_concentration["concentration"]
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
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"])
    / "co2.nc",
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


def _mu_g_m3_to_ppm(x):
    """Convert concentration from µg/m³ to ppm."""
    return x * 22.71108 / 44.01 / 1000
