import logging
from pathlib import Path

import ggpymanager as ggp
import xarray as xr

import paris_2025 as p
from paris_2025.config import CONFIG


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
        elevation = ggp.processing.smooth_elevation(elevation)
        geom = ggp.processing.create_ggeom_dataset(
            elevation=elevation,
            nz=gramm_conf["nz"],
            z0=gramm_conf["z0"],
            vert_stretching=gramm_conf["vert_stretching"],
        )
        logging.info(f"Writing geometry file to {geometry_file}...")
        ggp.io.write_ggeom_file(geom, file_path=geometry_file)

    logging.info("Preparing terrain for GRAL...")
    gral_conf = CONFIG["domain"]["gral"]
    geometry_file = Path(gral_conf["conf_path"]) / "GRAL_topofile.txt"
    if geometry_file.exists():
        logging.info(f"Geometry file {geometry_file} already exists. Skipping terrain.")
    else:
        path = p.model_input.terrain.process_terrain("gral", only_check_status=False)
        elevation = xr.open_dataset(path)["elevation"]
        logging.info(f"Writing geometry file to {geometry_file}...")
        ggp.io.write_esri_ascii(geometry_file, elevation)


def prepare_landcover():
    logging.info("Preparing landcover...")
    landuse_file = Path(CONFIG["domain"]["gramm"]["conf_path"]) / "landuse.asc"
    if landuse_file.exists():
        logging.info(f"Landuse file {landuse_file} already exists. Skipping landcover.")
        return
    gramm_landcover_path = p.model_input.landcover.process_landcover()
    landcover = xr.open_dataset(gramm_landcover_path)
    ggp.io.write_landuse(landuse_file, landcover)


def prepare_buildings():
    logging.info("Preparing buildings...")
    buildings_file = Path(CONFIG["domain"]["gral"]["conf_path"]) / "buildings.dat"
    if buildings_file.exists():
        logging.info(
            f"Buildings file {buildings_file} already exists. Skipping buildings."
        )
        return
    gral_buildings_path = p.model_input.buildings.process_buildings()
    buildings = xr.open_dataset(gral_buildings_path)
    logging.info(f"Writing buildings file to {buildings_file}...")
    ggp.io.write_buildings_file(buildings_file, buildings["building_height"])


def prepare_fluxes():
    logging.info("Preparing fluxes...")
    source_group_path = p.model_input.fluxes.process_fluxes()
    source_group_ds = xr.open_dataset(source_group_path)

    logging.info("Preparing point sources for GRAL...")
    point_file = Path(CONFIG["domain"]["gral"]["conf_path"]) / "point.dat"
    points = source_group_ds.sel(source_group=source_group_ds.geometry == "point")
    if not point_file.exists():
        logging.info(f"Writing point file to {point_file}...")
        ggp.io.write_point_dat(
            path=point_file,
            x=points.x_point.values,
            y=points.y_point.values,
            z=points.z_point.values,
            flux=points.source_flux.values,
            exit_velocity=points.exit_velocity.values,
            stack_diameter=points.stack_diameter.values,
            exit_temperature=points.exit_temperature.values,
            source_group=points.source_group.values,
        )

    logging.info("Preparing area sources for GRAL...")
    cadastre_file = Path(CONFIG["domain"]["gral"]["conf_path"]) / "cadastre.dat"
    p.model_input.fluxes.create_cadastre_dat_from_area(path=cadastre_file)


if __name__ == "__main__":
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Model input
    prepare_domain()
    prepare_terrain()
    prepare_landcover()
    prepare_buildings()
    prepare_fluxes()

    # Measurement data
    p.meteo.create_meteo_measurements()
    p.meteo.create_temperature_and_pressure_dataset()
    p.tracers.create_co2_measurements()
    p.google_earth_files.create_files_for_google_earth()
