import logging
from pathlib import Path

import geopandas as gpd
import ggpymanager as ggp

import paris_2025 as p
from paris_2025.config import CONFIG


def create_files_for_google_earth():
    out_path = Path(CONFIG["data_path"])
    logging.info("Loading meteo measurements...")
    meteo = p.meteo.get_meteo_measurements()
    # Create geojson of meteo stations
    gdf = gpd.GeoDataFrame(
        {
            "name": meteo.station.values,
            "latitude": meteo.latitude.values,
            "longitude": meteo.longitude.values,
            "altitude": meteo.altitude.values,
        },
        geometry=gpd.points_from_xy(
            meteo.longitude.values, meteo.latitude.values, meteo.altitude.values
        ),
        crs="EPSG:4326",
    )
    kml_output_path = out_path / "meteo_stations.kml"
    logging.info(f"Writing meteo stations to {kml_output_path}...")
    gdf.to_file(kml_output_path, driver="KML")
    logging.info("Adding altitude mode to meteo stations...")
    ggp.processing.google_earth.add_altitude_mode_to_points(
        kml_file=str(kml_output_path), mode="absolute", add_names=True
    )
    logging.info("Loading CO2 measurements...")
    co2 = p.tracers.get_co2_measurements()
    gdf = gpd.GeoDataFrame(
        {
            "name": co2.station.values,
            "latitude": co2.latitude.values,
            "longitude": co2.longitude.values,
            "height": co2.height.values,
        },
        geometry=gpd.points_from_xy(
            co2.longitude.values, co2.latitude.values, co2.height.values
        ),
        crs="EPSG:4326",
    )
    kml_output_path = out_path / "tracer_stations.kml"
    logging.info(f"Writing tracer stations to {kml_output_path}...")
    gdf.to_file(kml_output_path, driver="KML")
    logging.info("Adding altitude mode to tracer stations...")
    ggp.processing.google_earth.add_altitude_mode_to_points(
        kml_file=str(kml_output_path), mode="relativeToGround", add_names=True
    )
