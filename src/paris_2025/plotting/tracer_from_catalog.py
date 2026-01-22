from pathlib import Path

import ggpymanager as ggp
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib import colors as mcolors

import paris_2025 as p
from paris_2025.plotting.common import get_metadata


def plot_source_group_contribution_to_stations(
    fig_path_1: str | Path, fig_path_2: str | Path
):
    """Plot contribution of source groups to stations."""
    source_groups = xr.open_dataset(
        "/Users/rmaiwald/Levante/Paris/Input/Fluxes/source_groups.nc"
    )
    area_id = xr.open_dataset("/Users/rmaiwald/Levante/Paris/Input/Fluxes/area_id.nc")
    co2 = p.model.get_co2_data()
    ppm = ggp.utils.ugm3_to_ppm(co2["concentration"], gas="CO2")
    ppm = ppm.where(ppm > 0)  # type: ignore

    ppm.plot(
        col="station", col_wrap=6, cmap="jet", norm=mcolors.LogNorm()
    )  # type: ignore
    plt.savefig(
        fig_path_1,
        metadata=get_metadata("Each pixel shows the contribution of a source group."),
    )
    plt.clf()

    fp = (
        ppm.groupby(source_groups["area_id"])
        .sum("source_group")
        .fillna(0.0)
        .mean("sim_id", skipna=False)
    )
    fp = fp.drop_vars(["x", "y", "z"])
    mindex = pd.MultiIndex.from_arrays(
        [area_id["y_center"].values, area_id["x_center"].values],
        names=["y_center", "x_center"],
    )
    fp["area_id"] = (("area_id"), mindex)
    fp = fp.unstack()
    fp.attrs = {
        "long_name": "Mean contribution over all source groups of that area",
        "units": "ppm",
    }
    fp.plot(col="station", col_wrap=6, cmap="jet")  # type: ignore
    plt.savefig(
        fig_path_2,
        metadata=get_metadata(
            "Mean over all source groups of that area,"
            " each pixel shows the contribution to a station."
        ),
    )
    plt.clf()


def plot_concentration_at_station_per_simulation(fig_path: str | Path):
    """Plot concentration at each station per simulation."""
    co2 = p.model.get_co2_data()
    ppm = ggp.utils.ugm3_to_ppm(co2["concentration"], gas="CO2")
    ppm.mean("source_group").plot()  # type: ignore
    plt.xticks(rotation=90)
    plt.savefig(
        fig_path,
        metadata=get_metadata("Mean concentration at each station per simulation."),
    )
    plt.clf()


def plot_hourly_vprm_concentration(fig_path: str | Path):
    """Plot mean hourly VPRM concentration with temporal factors applied."""
    concentration_timeseries = xr.open_dataset(
        p.CONFIG["output_path"] + "/" + ggp.config.CONCENTRATION_TIMESERIES_FILE_NAME
    ).sel(best_sim_id=0)

    # Calculate hourly mean concentration
    vprm_data = ggp.utils.ugm3_to_ppm(
        concentration_timeseries["co2_timeseries"]
        .sel(
            loss_type="rmse - filter: True",
            type=concentration_timeseries.type.str.contains("VPRM 2023 GEE"),
            time="2023-08",
        )
        .sum("type"),
        "co2",
    )
    # Type checking: ensure vprm_data is xr.DataArray
    assert isinstance(vprm_data, xr.DataArray)
    concentration = vprm_data.groupby("time.hour").mean("time")

    concentration.plot(cbar_kwargs={"label": "Mixing ratio [ppm]"})  # type: ignore
    plt.xticks(rotation=90)
    plt.xlabel("Station")
    plt.ylabel("Hour of day")
    plt.title("Mean hourly contribution from VPRM GEE in August 2023")
    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Mean hourly VPRM concentration with temporal factors applied."
        ),
    )
    plt.clf()
