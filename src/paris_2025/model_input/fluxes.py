import logging
from datetime import datetime
from pathlib import Path

import ggpymanager as ggp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from paris_2025 import CONFIG

AREA_ID_NETCDF_PATH = Path(CONFIG["data_path"]) / "Fluxes/area_id.nc"
FIGURE_PATH = Path(CONFIG["figures_path"]) / "Input/area_id_overview.png"


def create_area_partitioning():
    """
    Create a partitioning of the model domain into areas and save it as a NetCDF file.
    """

    if not AREA_ID_NETCDF_PATH.exists():
        nx = CONFIG["fluxes"]["nx_areas"]
        ny = CONFIG["fluxes"]["ny_areas"]

        gral_grid = ggp.utils.create_domain_grid("gral", CONFIG)
        minx, maxx = gral_grid.x.values[[0, -1]]
        miny, maxy = gral_grid.y.values[[0, -1]]
        xbins = np.linspace(minx, maxx, nx + 1)
        ybins = np.linspace(miny, maxy, ny + 1)
        gral_grid["xbins"] = (("xbins"), xbins)
        gral_grid["ybins"] = (("ybins"), ybins)
        gral_grid = gral_grid.assign_coords(xbins=xbins, ybins=ybins)

        gral_grid["area_id"] = (
            ("y", "x"),
            np.nan * np.ones(gral_grid.y.shape + gral_grid.x.shape),
        )
        for i in range(nx):
            for j in range(ny):
                gral_grid["area_id"].loc[
                    dict(
                        x=slice(xbins[i], xbins[i + 1]), y=slice(ybins[j], ybins[j + 1])
                    )
                ] = (i + j * nx)

        # Add attrs
        timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        gral_grid.attrs["description"] = f"Dataset compiled {timestamp} EDT"
        gral_grid["area_id"].attrs = {
            "long_name": "Area ID",
            "description": f"Partitioning of the model domain into {nx} x {ny} areas",
            "units": "1",
            "standard_name": "area_id",
        }
        gral_grid["xbins"].attrs = {
            "long_name": "X bin edges",
            "units": "m",
            "standard_name": "projection_x_coordinate",
        }
        gral_grid["ybins"].attrs = {
            "long_name": "Y bin edges",
            "units": "m",
            "standard_name": "projection_y_coordinate",
        }
        gral_grid["x"].attrs = {
            "long_name": "Easting",
            "units": "m",
            "standard_name": "projection_x_coordinate",
        }
        gral_grid["y"].attrs = {
            "long_name": "Northing",
            "units": "m",
            "standard_name": "projection_y_coordinate",
        }

        logging.info(f"Writing area_id to {AREA_ID_NETCDF_PATH}")
        gral_grid[["x", "y", "xbins", "ybins", "area_id"]].to_netcdf(
            AREA_ID_NETCDF_PATH
        )
    else:
        logging.info(f"File {AREA_ID_NETCDF_PATH} exists, not overwriting.")
    gral_grid = xr.open_dataset(AREA_ID_NETCDF_PATH)

    if not FIGURE_PATH.exists():
        minx, maxx = gral_grid.x.values[[0, -1]]
        miny, maxy = gral_grid.y.values[[0, -1]]
        plt.hlines(gral_grid["ybins"], minx, maxx, label="Bins")
        plt.vlines(gral_grid["xbins"], miny, maxy)
        gral_grid.area_id.plot()
        plt.title("Area ID")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend(loc="upper right")
        plt.savefig(FIGURE_PATH)
