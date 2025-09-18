import logging
from pathlib import Path

import ggpymanager as ggp
import xarray as xr

import paris_2025 as p
from paris_2025 import CONFIG


def prepare_domain(
    # crs, gral_bbox, gral_dx, gral_dy, gramm_bbox, gramm_dx, gramm_dy, output_path
):
    logging.info("Preparing domain...")


def prepare_terrain():
    # TODO: Load processed files and check if the coordinates are correct
    logging.info("Preparing terrain for GRAMM...")
    gramm_conf = CONFIG["domain"]["gramm"]
    geometry_file = Path(gramm_conf["conf_path"]) / "ggeom.asc"
    if geometry_file.exists():
        logging.info(f"Geometry file {geometry_file} already exists. Skipping terrain.")
    else:
        path = p.model_input.terrain.process_terrain("gramm", only_check_status=False)
        elevation = xr.open_dataset(path)["elevation"]
        elevation = ggp.gramm_geometry.smooth_elevation(elevation)
        geom = ggp.gramm_geometry.create_ggeom_dataset(
            elevation=elevation,
            nz=gramm_conf["nz"],
            z0=gramm_conf["z0"],
            vert_stretching=gramm_conf["vert_stretching"],
        )
        logging.info(f"Writing geometry file to {geometry_file}...")
        ggp.gramm_geometry.write_ggeom_file(geom, file_path=geometry_file)

    logging.info("Preparing terrain for GRAL...")
    gral_conf = CONFIG["domain"]["gral"]
    geometry_file = Path(gral_conf["conf_path"]) / "GRAL_topofile.txt"
    if geometry_file.exists():
        logging.info(f"Geometry file {geometry_file} already exists. Skipping terrain.")
    else:
        path = p.model_input.terrain.process_terrain("gral", only_check_status=False)
        elevation = xr.open_dataset(path)["elevation"]
        logging.info(f"Writing geometry file to {geometry_file}...")
        ggp.utils.write_esri_ascii(geometry_file, elevation)


def prepare_landcover():
    logging.info("Preparing landcover...")
    landuse_file = Path(CONFIG["domain"]["gramm"]["conf_path"]) / "landuse.asc"
    if landuse_file.exists():
        logging.info(f"Landuse file {landuse_file} already exists. Skipping landcover.")
        return
    gramm_landcover_path = p.model_input.landcover.process_landcover()
    landcover = xr.open_dataset(gramm_landcover_path)
    ggp.utils.write_landuse(landuse_file, landcover)


def prepare_buildings():
    logging.info("Preparing buildings...")


def prepare_fluxes():
    logging.info("Preparing fluxes...")


if __name__ == "__main__":
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    prepare_domain()
    prepare_terrain()
    prepare_landcover()
    prepare_fluxes()
