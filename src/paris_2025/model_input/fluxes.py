import logging
from datetime import datetime
from pathlib import Path

import ggpymanager as ggp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from rasterio.enums import Resampling

from paris_2025 import CONFIG

AREA_ID_NETCDF_PATH = Path(CONFIG["data_path"]) / "Fluxes/area_id.nc"
FIGURE_PATHS = {
    "area_id": Path(CONFIG["figures_path"]) / "Input/area_id_overview.png",
    "area_fluxes": Path(CONFIG["figures_path"]) / "Input/area_fluxes.png",
}

VPRM_2023_R_NETCDF_PATH = Path(CONFIG["data_path"]) / "Fluxes/area_flux_vprm_2023_R.nc"
VPRM_2023_GEE_NETCDF_PATH = (
    Path(CONFIG["data_path"]) / "Fluxes/area_flux_vprm_2023_GEE.nc"
)
VPRM_2024_R_NETCDF_PATH = Path(CONFIG["data_path"]) / "Fluxes/area_flux_vprm_2024_R.nc"
VPRM_2024_GEE_NETCDF_PATH = (
    Path(CONFIG["data_path"]) / "Fluxes/area_flux_vprm_2024_GEE.nc"
)


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

    if not FIGURE_PATHS["area_id"].exists():
        minx, maxx = gral_grid.x.values[[0, -1]]
        miny, maxy = gral_grid.y.values[[0, -1]]
        plt.hlines(gral_grid["ybins"], minx, maxx, label="Bins")
        plt.vlines(gral_grid["xbins"], miny, maxy)
        gral_grid.area_id.plot()
        plt.title("Area ID")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend(loc="upper right")
        plt.savefig(FIGURE_PATHS)


def convert_vprm_units(vprm):
    """Convert VPRM units from umol CO2 / m2 / s to kg CO2 / m2 / h"""
    molar_mass_co2 = 44.01  # g/mol
    conversion_factor = molar_mass_co2 / 1e6 * 3600 / 1000  # kg / umol * s to h

    for var in vprm.data_vars:
        attrs = vprm[var].attrs
        assert attrs["units"] == "mu mol CO2 m-2 s-1"
        attrs["units"] = "kg CO2 m-2 h-1"
        attrs["long_name"] = "Annual mean CO2 flux"
        vprm[var] = vprm[var] * conversion_factor
        vprm[var].attrs = attrs
    return vprm


def create_vprm_area_fluxes():
    area_id = xr.open_dataset(AREA_ID_NETCDF_PATH)
    vprm_crs = "EPSG:32631"
    vprm = {}
    for year in [2023, 2024]:
        vprm[year] = xr.open_dataset(
            f"/Users/rmaiwald/Levante/Paris/Input/Fluxes/VPRM/vprm_mean_{year}.nc"
        ).rio.write_crs(vprm_crs)
        vprm[year] = vprm[year].rio.reproject_match(
            area_id, resampling=Resampling.average
        )
        vprm[year] = vprm[year].fillna(0)
        vprm[year] = convert_vprm_units(vprm[year])

    for year, r_path, gee_path in [
        (2023, VPRM_2023_R_NETCDF_PATH, VPRM_2023_GEE_NETCDF_PATH),
        (2024, VPRM_2024_R_NETCDF_PATH, VPRM_2024_GEE_NETCDF_PATH),
    ]:
        for flux_type, path in [("R", r_path), ("GEE", gee_path)]:
            if not path.exists():
                logging.info(f"Writing {path}")
                ds = (
                    vprm[year][flux_type]
                    .to_dataset(name="flux")
                    .expand_dims(dim={"type": [f"VPRM {year} {flux_type}"]}, axis=0)
                )
                ds["type"].attrs["long_name"] = "Flux type"
                timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                ds.attrs["description"] = f"Dataset compiled {timestamp} EDT"
                ds.to_netcdf(path)
            else:
                logging.info(f"{path} exists, skipping.")


def plot_area_fluxes():
    if not FIGURE_PATHS["area_fluxes"].exists():
        logging.info(f"Plotting area fluxes to {FIGURE_PATHS['area_fluxes']}")
        xr.open_mfdataset(
            str(Path(CONFIG["data_path"]) / "Fluxes/area_flux_*.nc")
        ).flux.plot(col="type", col_wrap=4)
        plt.savefig(FIGURE_PATHS["area_fluxes"])
    else:
        logging.info(f"File {FIGURE_PATHS['area_fluxes']} exists, skipping plot.")


def process_fluxes():
    create_area_partitioning()
    
    create_vprm_area_fluxes()
    plot_area_fluxes()
