import ggpymanager as ggp
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import sys

import paris_2025 as p


def get_metadata(description=None):
    """Get metadata for the plots."""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    caller_name = sys._getframe().f_back.f_code.co_name  # type: ignore
    _description = f"Created by function '{caller_name}' on {date_str}."
    if description is not None:
        _description += f"\n{description}"
    return {"Description": _description}


def plot_source_group_contribution_to_stations(fig_path_1, fig_path_2):
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

    fp = ppm.groupby(source_groups["area_id"]).sum("source_group").mean("sim_id")
    fp = fp.drop_vars(["x", "y", "z"])
    # Use xarray's indexing directly with the area_id DataArray
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
        fig_path_2, metadata=get_metadata("Mean over all source groups of that area,")
    )
    plt.clf()
