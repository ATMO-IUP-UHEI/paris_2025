import logging
from pathlib import Path

import ggpymanager as ggp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from paris_2025 import CONFIG


def create_area_partitioning():
    """
    Create a partitioning of the model domain into areas and save it as a NetCDF file.
    """
    AREA_ID_NETCDF_PATH = Path(CONFIG["data_path"]) / "Fluxes/area_id.nc"
    FIGURE_PATH = Path(CONFIG["figures_path"]) / "Input/area_id_overview.png"
    nx = CONFIG["fluxes"]["nx_areas"]
    ny = CONFIG["fluxes"]["ny_areas"]

    gral_grid = ggp.utils.create_domain_grid("gral", CONFIG)
    minx, maxx = gral_grid.x.values[[0, -1]]
    miny, maxy = gral_grid.y.values[[0, -1]]
    xbins = np.linspace(minx, maxx, nx + 1)
    ybins = np.linspace(miny, maxy, ny + 1)
    gral_grid["area_id"] = (
        ("y", "x"),
        np.nan * np.ones(gral_grid.y.shape + gral_grid.x.shape),
    )
    for i in range(nx):
        for j in range(ny):
            gral_grid["area_id"].loc[
                dict(x=slice(xbins[i], xbins[i + 1]), y=slice(ybins[j], ybins[j + 1]))
            ] = (i + j * nx)

    if not AREA_ID_NETCDF_PATH.exists():
        logging.info(f"Writing area_id to {AREA_ID_NETCDF_PATH}")
        gral_grid[["x", "y", "area_id"]].to_netcdf(AREA_ID_NETCDF_PATH)
    else:
        logging.info(f"File {AREA_ID_NETCDF_PATH} exists, not overwriting.")
    gral_grid = xr.open_dataset(AREA_ID_NETCDF_PATH)

    if not FIGURE_PATH.exists():
        plt.hlines(ybins, minx, maxx, label="Bins")
        plt.vlines(xbins, miny, maxy)
        gral_grid.area_id.plot()
        plt.title("Area ID")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend(loc="upper right")
        plt.savefig(FIGURE_PATH)
