"""
Terrain data processing module for GRAMM simulations.

This module provides functionality to process terrain data for GRAMM (Graz Mesoscale
Model) simulations, including loading terrain tiles, reprojecting to model grids, and
preparing NetCDF files for simulation input.

The module handles French terrain data from RGEALTI format and converts it to the
appropriate grid resolution and projection for GRAMM simulations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import ggpymanager as ggp
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray  # noqa
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from rasterio.enums import Resampling

from paris_2025 import CONFIG


def create_tmp_netcdf(
    config: Dict[str, Any],
    domain_area: gpd.GeoDataFrame,
    tmp_file: Path,
    figure_path: Path,
) -> None:
    """
    Convert terrain data tiles to a single NetCDF file.

    This function reads terrain data from multiple RGEALTI format files, clips them
    to the domain area, and combines them into a single NetCDF file for efficient
    processing.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing domain CRS information.
    domain_area : gpd.GeoDataFrame
        GeoDataFrame representing the area of interest for clipping terrain tiles.
    tmp_file : Path
        Path where the temporary NetCDF file will be saved.

    Returns
    -------
    None

    Notes
    -----
    This function assumes terrain data is located at a hardcoded path and follows
    the RGEALTI directory structure. It also generates a visualization of terrain
    tiles as a side effect.
    """
    logging.info(f"Converting terrain data to netcdf file {tmp_file}")
    terrain_data_path = Path("/Users/rmaiwald/Levante/Paris/Input/Terrain/RGEALTI/")
    shape_files = sorted(list(terrain_data_path.glob("**/*.shp")))
    tile_gdf = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(f) for f in shape_files], ignore_index=True)
    )

    assert (
        tile_gdf.crs == config["domain"]["crs"]
    ), "CRS of the GeoDataFrame does not match the domain CRS"

    intersection = tile_gdf.clip(domain_area)

    logging.info(f"Total number of tiles in the dataset: {len(tile_gdf)}")
    logging.info(f"Number of tiles intersecting with the area: {len(intersection)}")

    plot_terrain_tiles(domain_area, tile_gdf, intersection, figure_path)

    tile_files = sorted(
        list(terrain_data_path.glob("RGEALTI_2*/RGEALTI/1_DONNEES*/RGEALTI_MNT*/*.asc"))
    )
    tile_files_dict = {f.stem: f for f in tile_files}
    tile_files = [tile_files_dict[name] for name in intersection["NOM_DALLE"].values]

    terrain = xr.open_mfdataset(
        tile_files,
        combine="by_coords",
        parallel=True,
        engine="rasterio",
        chunks={
            "x": 10000,
            "y": 10000,
        },
    )
    terrain = terrain.rio.write_crs(config["domain"]["crs"])
    terrain = terrain.squeeze("band")
    terrain = terrain.rename(band_data="elevation")
    logging.info(
        f"Terrain dataset dimensions: {terrain.dims}\n"
        f"Terrain dataset coords: {terrain.coords}\n"
        f"Size of the terrain dataset: {terrain.nbytes / 1e9:.2f} GB\n"
        f"Number of chunks in the terrain dataset: {terrain.chunks}\n"
    )

    if not tmp_file.exists():
        logging.info(f"Writing terrain data to {tmp_file}")
        delayed = terrain.to_netcdf(tmp_file, compute=False)
        with ProgressBar():
            delayed.compute()


def load_terrain(tmp_file: Path) -> xr.Dataset:
    """
    Load terrain data from a NetCDF file.

    Parameters
    ----------
    tmp_file : Path
        Path to the NetCDF file containing terrain data.

    Returns
    -------
    xr.Dataset
        xarray Dataset containing terrain elevation data with proper coordinate
        decoding and chunking for efficient processing.
    """
    logging.info(f"Loading terrain data at {tmp_file}")
    terrain = xr.open_dataset(tmp_file, decode_coords="all", chunks="auto")
    return terrain


def plot_terrain_tiles(
    domain_area: gpd.GeoDataFrame,
    tile_gdf: gpd.GeoDataFrame,
    intersection: gpd.GeoDataFrame,
    figure_path: Path,
) -> None:
    """
    Create a visualization of terrain tiles and domain area.

    Parameters
    ----------
    domain_area : gpd.GeoDataFrame
        GeoDataFrame representing the domain boundary.
    tile_gdf : gpd.GeoDataFrame
        GeoDataFrame containing all available terrain tiles.
    intersection : gpd.GeoDataFrame
        GeoDataFrame containing terrain tiles that intersect with the domain area.
    """
    import matplotlib.patches as mpatches

    # Create the plots
    tile_gdf.plot(color="lightblue", edgecolor="gray", alpha=0.7)
    domain_area.plot(ax=plt.gca(), color="none", edgecolor="black", lw=2)
    intersection.plot(ax=plt.gca(), color="red", alpha=0.7)

    # Create legend manually using proxy artists
    tiles_patch = mpatches.Patch(color="lightblue", alpha=0.7, label="All tiles")
    domain_patch = mpatches.Patch(
        color="none", edgecolor="black", label="Domain borders"
    )
    intersection_patch = mpatches.Patch(color="red", alpha=0.7, label="Tiles in domain")

    plt.title("Tiles of the domain")
    plt.legend(handles=[tiles_patch, domain_patch, intersection_patch])
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Created figure of terrain tiles {figure_path}")


def create_gramm_terrain_netcdf(
    gramm_grid: xr.Dataset, terrain: xr.Dataset, GRAMM_TERRAIN_PATH: Path
) -> xr.Dataset:
    """
    Create reprojected terrain NetCDF file for GRAMM simulations.

    This function reprojects terrain data to match the GRAMM grid resolution and extent,
    adds proper metadata attributes, and saves the result to a NetCDF file.

    Parameters
    ----------
    gramm_grid : xr.Dataset
        Target GRAMM grid dataset containing the desired spatial resolution and extent.
    terrain : xr.Dataset
        Source terrain dataset to be reprojected.
    GRAMM_TERRAIN_PATH : Path
        Path where the reprojected terrain NetCDF file will be saved.

    Returns
    -------
    xr.Dataset
        Reprojected terrain dataset with elevation data matching the GRAMM grid,
        including proper metadata attributes.

    Raises
    ------
    AssertionError
        If the CRS of terrain data doesn't match the GRAMM grid CRS.

    Notes
    -----
    Uses average resampling method and adds comprehensive metadata attributes
    including creation timestamp and CF-compliant variable descriptions.
    """
    assert (
        terrain.rio.crs == gramm_grid.rio.crs
    ), "CRS of the terrain data does not match the domain CRS"

    with ProgressBar():
        logging.info("Reprojecting terrain data to gramm grid")
        reprojected_elevation = terrain.elevation.rio.reproject_match(
            gramm_grid.grid_placeholder, resampling=Resampling.average
        )

    reprojected_terrain = reprojected_elevation.to_dataset()

    reprojected_terrain["x"] = gramm_grid["x"]
    reprojected_terrain["y"] = gramm_grid["y"]

    # Add attrs
    timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    reprojected_terrain.attrs["description"] = f"Dataset compiled {timestamp} EDT"

    reprojected_terrain["elevation"].attrs = {
        "long_name": "Elevation",
        "units": "m",
        "standard_name": "surface_altitude",
    }
    reprojected_terrain["x"].attrs = {
        "long_name": "Easting",
        "units": "m",
        "standard_name": "projection_x_coordinate",
    }
    reprojected_terrain["y"].attrs = {
        "long_name": "Northing",
        "units": "m",
        "standard_name": "projection_y_coordinate",
    }
    reprojected_terrain.to_netcdf(GRAMM_TERRAIN_PATH)
    return reprojected_terrain


def process_terrain(only_check_status: bool = False) -> Path:
    """
    Main function to process terrain data for GRAMM simulations.

    This function orchestrates the complete terrain processing workflow, including
    loading configuration, creating grids, processing terrain tiles, and generating
    the final reprojected terrain dataset.

    Parameters
    ----------
    only_check_status : bool, optional
        If True, only check and report the existence of terrain data files without
        processing. Default is False.

    Returns
    -------
    Optional[None]
        Always returns None.

    Notes
    -----
    The function creates two main output files:
    - Intermediate terrain data (terrain.nc) - Combined raw terrain tiles
    - Final reprojected terrain data (gramm_terrain.nc) - Grid-matched terrain for GRAMM

    If files already exist, they are reused to avoid unnecessary recomputation.
    Processing includes terrain tile discovery, clipping to domain, reprojection,
    and metadata enhancement.
    """
    TMP_FILE = Path(CONFIG["data_path"]) / "Terrain/terrain.nc"
    FIGURE_PATH = Path(CONFIG["figure_path"]) / "Input/terrain_tile_map.png"
    GRAMM_TERRAIN_PATH = Path(CONFIG["data_path"]) / "Terrain/gramm_terrain.nc"
    # TODO: Implement functions to prepare the terrain input for GRAL

    if only_check_status:
        logging.info("Checking status of terrain data files...")
        if TMP_FILE.exists():
            logging.info(f"Intermediate terrain data exists at {TMP_FILE}")
        else:
            logging.info(f"Intermediate terrain data does not exist at {TMP_FILE}")
        if GRAMM_TERRAIN_PATH.exists():
            logging.info(f"Reprojected terrain data exists at {GRAMM_TERRAIN_PATH}")
        else:
            logging.info(
                f"Reprojected terrain data does not exist at {GRAMM_TERRAIN_PATH}"
            )
        return GRAMM_TERRAIN_PATH

    domain_area = ggp.utils.create_domain_area(CONFIG)
    gramm_grid = ggp.utils.create_gramm_grid(CONFIG)

    if (not TMP_FILE.exists()) or (not FIGURE_PATH.exists()):
        create_tmp_netcdf(CONFIG, domain_area, TMP_FILE, FIGURE_PATH)
    terrain = load_terrain(TMP_FILE)

    if not GRAMM_TERRAIN_PATH.exists():
        create_gramm_terrain_netcdf(gramm_grid, terrain, GRAMM_TERRAIN_PATH)
    else:
        logging.info(f"File {GRAMM_TERRAIN_PATH} is already created")
    return GRAMM_TERRAIN_PATH
