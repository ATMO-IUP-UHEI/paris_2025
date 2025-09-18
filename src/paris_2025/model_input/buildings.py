import logging
from datetime import datetime
from pathlib import Path

import ggpymanager as ggp
import matplotlib.pyplot as plt
import rioxarray  # noqa
import xarray as xr
from rasterio.enums import Resampling

from paris_2025 import CONFIG


def create_buildings_netcdf(gral_grid, GRAL_BUILDINGS_PATH) -> None:
    buildings = xr.open_dataset(
        Path(CONFIG["data_path"]) / "Buildings/FR001_PARIS_UA2012_DHM_V020/Dataset/"
        "FR001_PARIS_UA2012_DHM_V020.tif"
    )
    reprojected_buildings = (
        buildings.band_data.isel(band=0)
        .rio.reproject_match(gral_grid, resampling=Resampling.max)
        .to_dataset(name="building_height")
    )
    assert (
        reprojected_buildings.rio.crs == gral_grid.rio.crs
    ), "CRS of the terrain data does not match the domain CRS"
    reprojected_buildings["x"] = gral_grid["x"]
    reprojected_buildings["y"] = gral_grid["y"]
    # Add attrs
    timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    reprojected_buildings.attrs["description"] = f"Dataset compiled {timestamp} EDT"

    reprojected_buildings["building_height"].attrs = {
        "long_name": "Building Height",
        "units": "m",
        "standard_name": "surface_altitude",
    }
    reprojected_buildings["x"].attrs = {
        "long_name": "Easting",
        "units": "m",
        "standard_name": "projection_x_coordinate",
    }
    reprojected_buildings["y"].attrs = {
        "long_name": "Northing",
        "units": "m",
        "standard_name": "projection_y_coordinate",
    }
    reprojected_buildings.to_netcdf(GRAL_BUILDINGS_PATH)


def create_plot(raster: xr.DataArray, figure_path, title) -> None:
    plt.figure(figsize=(12, 8))
    raster.plot()  # type: ignore
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(figure_path)
    plt.close()


def process_buildings(only_check_status: bool = False) -> Path:
    FIGURE_PATH = Path(CONFIG["figures_path"]) / "Input/buildings.png"
    GRAL_BUILDINGS_PATH = Path(CONFIG["data_path"]) / "Buildings/buildings.nc"

    if only_check_status:
        if GRAL_BUILDINGS_PATH.exists():
            logging.info(f"Building file {GRAL_BUILDINGS_PATH} exists.")
        else:
            logging.warning(f"Building file {GRAL_BUILDINGS_PATH} does not exist.")
        return GRAL_BUILDINGS_PATH

    gral_grid = ggp.utils.create_domain_grid("gral", CONFIG)

    if not GRAL_BUILDINGS_PATH.exists():
        logging.info(f"Creating building file {GRAL_BUILDINGS_PATH}...")
        create_buildings_netcdf(gral_grid, GRAL_BUILDINGS_PATH)
    else:
        logging.info(f"Building file {GRAL_BUILDINGS_PATH} already exists. Skipping.")

    if not FIGURE_PATH.exists():
        logging.info(f"Creating building plot {FIGURE_PATH}...")
        buildings = xr.open_dataset(GRAL_BUILDINGS_PATH)["building_height"]
        create_plot(buildings, FIGURE_PATH, title="Building Height")
    return GRAL_BUILDINGS_PATH
