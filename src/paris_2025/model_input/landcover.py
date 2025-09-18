import logging
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import ggpymanager as ggp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa
import xarray as xr

from paris_2025 import CONFIG

N_PROCESSES = 4


def load_landcover_file(domain_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Load and preprocess landcover data from Urban Atlas.

    Parameters
    ----------
    domain_area : gpd.GeoDataFrame
        GeoDataFrame defining the domain area to clip landcover data

    Returns
    -------
    gpd.GeoDataFrame
        Clipped and preprocessed landcover GeoDataFrame with simplified codes
    """
    file_path = (
        Path(CONFIG["data_path"]) / "Landcover/UrbanAtlas/Results/"
        "FR001L1_PARIS_UA2018_v013/Data/FR001L1_PARIS_UA2018_v013.gpkg"
    )
    landcover_gdf = gpd.read_file(file_path, layer=0)
    landcover_gdf["code_2018"] = landcover_gdf["code_2018"].str[:3].astype(int)
    landcover_gdf = landcover_gdf.to_crs(CONFIG["domain"]["crs"])
    landcover_gdf = landcover_gdf.clip(domain_area)
    return landcover_gdf


def create_plot(raster: xr.DataArray, figure_paths: Dict[str, Path], key: str) -> None:
    """
    Create and save a plot of a raster DataArray.

    Parameters
    ----------
    raster : xr.DataArray
        Raster data to plot
    figure_paths : Dict[str, Path]
        Dictionary mapping plot keys to output file paths
    key : str
        Key identifying which plot to create and where to save it
    """
    plt.figure(figsize=(12, 8))
    raster.plot()  # type: ignore
    plt.title(key.replace("_", " ").title())
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(figure_paths[key])
    plt.close()


def compute_single_intersection(
    grid_gdf: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame, part_id: Optional[int] = None
) -> gpd.GeoDataFrame:
    """
    Compute intersection between grid cells and landcover polygons.

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Grid cells as GeoDataFrame
    gdf : gpd.GeoDataFrame
        Landcover polygons as GeoDataFrame
    part_id : Optional[int], default=None
        Identifier for this processing part (used in logging)

    Returns
    -------
    gpd.GeoDataFrame
        Intersection result with landcover codes and geometries
    """
    # Compute intersection
    if part_id is not None:
        logging.info(f"Processing part {part_id}")
    intersection = grid_gdf.overlay(gdf[["code_2018", "geometry"]], how="intersection")
    return intersection


def compute_intersection(
    grid_gdf: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame, n_splits: int
) -> pd.DataFrame:
    """
    Compute intersection using multiprocessing for performance.

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Grid cells as GeoDataFrame
    gdf : gpd.GeoDataFrame
        Landcover polygons as GeoDataFrame
    n_splits : int
        Number of parts to split the grid for parallel processing

    Returns
    -------
    pd.DataFrame
        Concatenated intersection results from all processes

    Raises
    ------
    AssertionError
        If CRS of input GeoDataFrames don't match
    """
    assert gdf.crs == grid_gdf.crs, "CRS of both GeoDataFrames must match"
    with Pool(N_PROCESSES) as p:
        results = p.starmap(
            compute_single_intersection,
            [
                (
                    grid_gdf_part,
                    gdf.clip(grid_gdf_part),  # type: ignore
                    i,
                )
                for i, grid_gdf_part in enumerate(np.array_split(grid_gdf, n_splits))
            ],
        )
    return pd.concat(results)


def create_landcover_class_raster(
    domain_area: gpd.GeoDataFrame, gramm_grid: xr.Dataset, TMP_FILE: Path
) -> None:
    """
    Create and save landcover class raster from vector data.

    Parameters
    ----------
    domain_area : gpd.GeoDataFrame
        Domain area for clipping landcover data
    gramm_grid : xr.Dataset
        GRAMM grid defining output raster structure
    TMP_FILE : Path
        Path to save temporary landcover class raster
    """
    logging.info("Loading landcover data...")
    landcover_gdf = load_landcover_file(domain_area)

    logging.info("Rasterizing landcover data...")
    xx, yy = np.meshgrid(gramm_grid.x, gramm_grid.y)
    sampling_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xx.flatten(), yy.flatten(), crs=gramm_grid.rio.crs)
    )
    intersection = compute_intersection(
        sampling_gdf, landcover_gdf, n_splits=len(sampling_gdf) // 10000 + 1
    )

    landcover_class_raster = xr.DataArray(
        intersection["code_2018"].values.reshape(xx.shape),  # type: ignore
        coords={"y": gramm_grid.y, "x": gramm_grid.x},
        dims=["y", "x"],
    )
    if not TMP_FILE.exists():
        landcover_class_raster.to_netcdf(TMP_FILE)


def create_gramm_landcover_netcdf(
    gramm_grid: xr.Dataset,
    landcover_class_raster: xr.DataArray,
    GRAMM_LANDCOVER_PATH: Path,
) -> xr.Dataset:
    """
    Create GRAMM-compatible landcover NetCDF file with physical properties.

    Parameters
    ----------
    gramm_grid : xr.Dataset
        GRAMM grid defining coordinate system
    landcover_class_raster : xr.DataArray
        Raster with landcover class codes
    GRAMM_LANDCOVER_PATH : Path
        Output path for GRAMM landcover NetCDF file

    Returns
    -------
    xr.Dataset
        Dataset with landcover variables converted to GRAMM format
    """
    logging.info("Creating grid with input variables...")
    landcover = ggp.utils.convert_to_gramm_landuse_variables(landcover_class_raster)
    landcover = landcover.rio.write_crs(gramm_grid.rio.crs)

    # Add attrs
    timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    landcover.attrs["description"] = f"Dataset compiled {timestamp} EDT"

    landcover["x"].attrs = {
        "long_name": "Easting",
        "units": "m",
        "standard_name": "projection_x_coordinate",
    }
    landcover["y"].attrs = {
        "long_name": "Northing",
        "units": "m",
        "standard_name": "projection_y_coordinate",
    }
    landcover.to_netcdf(GRAMM_LANDCOVER_PATH)
    return landcover


def create_landcover_plots(
    FIGURE_PATHS: Dict[str, Path], landcover: xr.Dataset
) -> None:
    """
    Create plots for various landcover properties.

    Parameters
    ----------
    FIGURE_PATHS : Dict[str, Path]
        Dictionary mapping plot keys to output file paths
    landcover : xr.Dataset
        Dataset containing landcover variables to plot
    """
    for key in [
        "soil_density",
        "heat_conductivity",
        "surface_roughness",
        "specific_soil_moisture",
    ]:
        logging.info(f"Creating plot of {key}...")
        for data_var in landcover.data_vars:
            if landcover[data_var].attrs.get("standard_name") == key:
                create_plot(landcover[data_var], FIGURE_PATHS, key=key)


def process_landcover(only_check_status: bool = False) -> Path:
    """
    Main function to process landcover data for GRAMM model input.

    Creates landcover class raster from Urban Atlas vector data, converts to
    GRAMM-compatible physical properties, and generates visualization plots.

    Parameters
    ----------
    only_check_status : bool, default=False
        If True, only check if output files exist without processing

    Returns
    -------
    Path
        Path to the final GRAMM landcover NetCDF file
    """
    TMP_FILE = Path(CONFIG["data_path"]) / "Landcover/gramm_landcover_classes.nc"
    FIGURE_PATHS = {
        "landcover_classes": Path(CONFIG["figure_path"])
        / "Input/landcover_classes.png",
        "soil_density": Path(CONFIG["figure_path"])
        / "Input/landcover_soil_density.png",
        "heat_conductivity": Path(CONFIG["figure_path"])
        / "Input/landcover_heat_conductivity.png",
        "surface_roughness": Path(CONFIG["figure_path"])
        / "Input/landcover_landcosurface_roughnessver_map.png",
        "specific_soil_moisture": Path(CONFIG["figure_path"])
        / "Input/landcover_specific_soil_moisture.png",
    }
    GRAMM_LANDCOVER_PATH = Path(CONFIG["data_path"]) / "Landcover/gramm_landcover.nc"

    if only_check_status:
        logging.info("Checking status of landcover data files...")
        if GRAMM_LANDCOVER_PATH.exists():
            logging.info(f"Reprojected landcover data exists at {GRAMM_LANDCOVER_PATH}")
        else:
            logging.info(
                f"Reprojected landcover data does not exist at {GRAMM_LANDCOVER_PATH}"
            )
        return GRAMM_LANDCOVER_PATH

    domain_area = ggp.utils.create_domain_geometry("gramm", CONFIG)
    gramm_grid = ggp.utils.create_domain_grid("gramm", CONFIG)

    if not TMP_FILE.exists():
        logging.info(f"Temporary file {TMP_FILE} does not exist and will be created.")
        create_landcover_class_raster(domain_area, gramm_grid, TMP_FILE)
    landcover_class_raster = xr.open_dataarray(TMP_FILE)
    logging.info("Creating plot of landcover classes...")
    create_plot(landcover_class_raster, FIGURE_PATHS, key="landcover_classes")
    if not GRAMM_LANDCOVER_PATH.exists():
        create_gramm_landcover_netcdf(
            gramm_grid, landcover_class_raster, GRAMM_LANDCOVER_PATH
        )
    else:
        logging.info(f"File {GRAMM_LANDCOVER_PATH} is already created")
    landcover = xr.open_dataset(GRAMM_LANDCOVER_PATH)
    create_landcover_plots(FIGURE_PATHS, landcover)
    return GRAMM_LANDCOVER_PATH
