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
from typing import Any, Dict, Literal

import geopandas as gpd
import ggpymanager as ggp
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray  # noqa
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from rasterio.enums import Resampling

from paris_2025.config import CONFIG

TMP_FILE = Path(CONFIG["data_path"]) / "Terrain/terrain.nc"
FIGURE_PATH = Path(CONFIG["figures_path"]) / "Input/terrain_tile_map.png"
GRAMM_TERRAIN_PATH = Path(CONFIG["data_path"]) / "Terrain/gramm_terrain.nc"
GRAL_TERRAIN_PATH = Path(CONFIG["data_path"]) / "Terrain/gral_terrain.nc"


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
    terrain_data_path = Path(CONFIG["data_path"]) / "Terrain/RGEALTI/"
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


def create_terrain_netcdf(
    grid: xr.Dataset, terrain: xr.Dataset, OUTPUT_TERRAIN_NETCDF_PATH: Path
) -> xr.Dataset:
    assert (
        terrain.rio.crs == grid.rio.crs
    ), "CRS of the terrain data does not match the domain CRS"

    with ProgressBar():
        logging.info("Reprojecting terrain data to grid")
        reprojected_terrain = terrain.elevation.rio.reproject_match(
            grid.grid_placeholder, resampling=Resampling.average
        ).to_dataset(name="elevation")

    reprojected_terrain["x"] = grid["x"]
    reprojected_terrain["y"] = grid["y"]

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
    reprojected_terrain.to_netcdf(OUTPUT_TERRAIN_NETCDF_PATH)
    return reprojected_terrain


def process_terrain(
    name: Literal["gramm", "gral"], only_check_status: bool = False
) -> Path:
    """
    Main function to process terrain data for GRAMM simulations.

    This function orchestrates the complete terrain processing workflow, including
    loading configuration, creating grids, processing terrain tiles, and generating
    the final reprojected terrain dataset.

    Parameters
    ----------
    for_gramm : bool, optional
        If True, process terrain data for GRAMM simulations. Default is True.
    for_gral : bool, optional
        If True, process terrain data for GRAL simulations. Default is True.
    only_check_status : bool, optional
        If True, only check and report the existence of terrain data files without
        processing. Default is False.

    Returns
    -------
    Dict[str, Path | None]
        Dictionary with paths to the processed terrain files for GRAMM and GRAL.

    Notes
    -----
    The function creates two main output files:
    - Intermediate terrain data (terrain.nc) - Combined raw terrain tiles
    - Final reprojected terrain data (gramm_terrain.nc) - Grid-matched terrain for GRAMM

    If files already exist, they are reused to avoid unnecessary recomputation.
    Processing includes terrain tile discovery, clipping to domain, reprojection,
    and metadata enhancement.
    """

    if only_check_status:
        logging.info("Checking status of terrain data files...")
        if TMP_FILE.exists():
            logging.info(f"Intermediate terrain data exists at {TMP_FILE}")
        else:
            logging.info(f"Intermediate terrain data does not exist at {TMP_FILE}")
        if name == "gramm":
            if GRAMM_TERRAIN_PATH.exists():
                logging.info(f"Reprojected terrain data exists at {GRAMM_TERRAIN_PATH}")
            else:
                logging.info(
                    f"Reprojected terrain data does not exist at {GRAMM_TERRAIN_PATH}"
                )
        if name == "gral":
            if GRAL_TERRAIN_PATH.exists():
                logging.info(f"GRAL terrain data exists at {GRAL_TERRAIN_PATH}")
            else:
                logging.info(f"GRAL terrain data does not exist at {GRAL_TERRAIN_PATH}")
    else:
        domain_area = ggp.processing.create_domain_geometry("gramm", CONFIG)

        if (not TMP_FILE.exists()) or (not FIGURE_PATH.exists()):
            create_tmp_netcdf(CONFIG, domain_area, TMP_FILE, FIGURE_PATH)
        terrain = load_terrain(TMP_FILE)

        if name == "gramm":
            gramm_grid = ggp.processing.create_domain_grid("gramm", CONFIG)
            if not GRAMM_TERRAIN_PATH.exists():
                create_terrain_netcdf(gramm_grid, terrain, GRAMM_TERRAIN_PATH)
            else:
                logging.info(f"File {GRAMM_TERRAIN_PATH} is already created")
        if name == "gral":
            gral_grid = ggp.processing.create_domain_grid("gral", CONFIG)
            if not GRAL_TERRAIN_PATH.exists():
                create_terrain_netcdf(gral_grid, terrain, GRAL_TERRAIN_PATH)
            else:
                logging.info(f"File {GRAL_TERRAIN_PATH} is already created")

    return GRAMM_TERRAIN_PATH if name == "gramm" else GRAL_TERRAIN_PATH
