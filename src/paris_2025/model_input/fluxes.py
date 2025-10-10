import logging
from datetime import datetime
from pathlib import Path

import ggpymanager as ggp
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from rasterio.enums import Resampling

from paris_2025 import CONFIG, model_input

d_path = Path(CONFIG["data_path"])
AREA_ID_NETCDF_PATH = d_path / "Fluxes/area_id.nc"
f_path = Path(CONFIG["figures_path"])
FIGURE_PATHS = {
    "area_id": f_path / "Input/area_id_overview.png",
    "area_fluxes": f_path / "Input/area_fluxes.png",
    "tno_point_fluxes": f_path / "Input/tno_point_fluxes.png",
    "cumulative_fluxes": f_path / "Input/cumulative_fluxes.png",
    "missing_flux": f_path / "Input/missing_flux.png",
    "missing_flux_percentage": f_path / "Input/missing_flux_percentage.png",
}

VPRM_2023_R_NETCDF_PATH = d_path / "Fluxes/area_flux_vprm_2023_R.nc"
VPRM_2023_GEE_NETCDF_PATH = d_path / "Fluxes/area_flux_vprm_2023_GEE.nc"
VPRM_2024_R_NETCDF_PATH = d_path / "Fluxes/area_flux_vprm_2024_R.nc"
VPRM_2024_GEE_NETCDF_PATH = d_path / "Fluxes/area_flux_vprm_2024_GEE.nc"

OE_AREA_NETCDF_PATH = d_path / "Fluxes/area_flux_oe_2023.nc"
OE_POINT_NETCDF_PATH = d_path / "Fluxes/point_flux_oe_2023.nc"

TNO_AREA_NETCDF_PATH = d_path / "Fluxes/area_flux_tno_2018.nc"
TNO_POINT_NETCDF_PATH = d_path / "Fluxes/point_flux_tno_2018.nc"

SOURCE_GROUP_NETCDF_PATH = d_path / "Fluxes/source_groups.nc"


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
    if all(
        path.exists()
        for path in [
            VPRM_2023_R_NETCDF_PATH,
            VPRM_2023_GEE_NETCDF_PATH,
            VPRM_2024_R_NETCDF_PATH,
            VPRM_2024_GEE_NETCDF_PATH,
        ]
    ):
        logging.info("All VPRM area flux files exist, skipping creation.")
        return

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


def convert_oe_area_units(oe):
    """Convert Origins.earth units from kg CO2 / km2 / h to kg CO2 / m2 / h"""
    oe["emissions"] = oe["emissions"] / 1e6
    oe["emissions"].attrs = {
        "long_name": "Annual mean CO2 flux",
        "units": "kg CO2 m-2 h-1",
    }
    return oe


def create_oe_area_fluxes():
    if OE_AREA_NETCDF_PATH.exists():
        logging.info(f"{OE_AREA_NETCDF_PATH} exists, skipping creation.")
        return
    gral_grid = ggp.utils.create_domain_grid("gral", CONFIG)

    logging.info("Loading Origins.earth data...")
    oe = xr.load_dataset(
        Path(CONFIG["data_path"])
        / "Fluxes/Origins.earth/processed/2023_sector_data_without_point_sources.nc"
    )
    mean_emissions = oe.emissions.mean(dim=["x", "y"])
    oe = oe.rio.write_crs(CONFIG["domain"]["crs"])
    oe = oe.rio.reproject_match(gral_grid, resampling=Resampling.average)
    oe = oe.fillna(0)
    new_mean_emissions = oe.emissions.mean(dim=["x", "y"])
    logging.info(
        f"Mean emissions changed from {mean_emissions.values} to "
        f"{new_mean_emissions.values} after regridding."
    )
    logging.info("Rescaling emissions to preserve total emissions.")
    oe["emissions"] = (mean_emissions / new_mean_emissions) * oe.emissions
    assert np.allclose(mean_emissions, oe.emissions.mean(dim=["x", "y"]).values)

    oe = convert_oe_area_units(oe)
    oe = oe.rename({"emissions": "flux", "sector": "type"})
    oe["type"] = [f"Origins.earth 2023 {t}" for t in oe["type"].values]

    oe["type"].attrs["long_name"] = "Flux type"
    timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    oe.attrs["description"] = f"Dataset compiled {timestamp} EDT"

    if not OE_AREA_NETCDF_PATH.exists():
        logging.info(f"Saving Origins.earth data to {OE_AREA_NETCDF_PATH}")
        oe.to_netcdf(OE_AREA_NETCDF_PATH)
    else:
        logging.info(f"File {OE_AREA_NETCDF_PATH} exists, skipping save.")


def convert_oe_point_units(oe_points):
    """Convert from kg CO2 / km2 / h to kg CO2 / h"""
    flux_attrs = oe_points["flux"].attrs
    assert flux_attrs["units"] == "kg CO2 / km^2 / h"
    # Cell size is 10 x 10 m
    oe_points["flux"] = oe_points["flux"] * 1e-6 * 10 * 10
    flux_attrs["units"] = "kg CO2 / h"
    oe_points["flux"].attrs = flux_attrs
    return oe_points


def create_oe_point_fluxes():
    if OE_POINT_NETCDF_PATH.exists():
        logging.info(f"{OE_POINT_NETCDF_PATH} exists, skipping creation.")
        return
    logging.info("Loading Origins.earth point source emissions")
    oe_points = xr.open_dataset(
        Path(CONFIG["data_path"])
        / "Fluxes/Origins.earth/processed/point_sources/"
        / "point_sources_elevations_from_Google_Earth.nc"
    ).load()
    oe_points = oe_points.rename({"emissions": "flux", "sector": "type"})
    oe_points["type"] = (
        "index",
        [f"Origins.earth 2023 {t}" for t in oe_points["type"].values],
    )

    oe_points["type"].attrs["long_name"] = "Flux type"

    logging.info(f"Assiginging CRS to point source emissions: {oe_points.attrs['crs']}")
    assert oe_points.attrs["crs"] == CONFIG["domain"]["crs"]
    oe_points = oe_points.rio.write_crs(CONFIG["domain"]["crs"])

    logging.info("Calculating point source heights above ground level")
    terrain = xr.open_dataset(model_input.terrain.GRAL_TERRAIN_PATH)
    gral_elevation = terrain.elevation.sel(
        x=oe_points.x, y=oe_points.y, method="nearest"
    )
    oe_points["z"] = oe_points["elevation"] - gral_elevation.values
    oe_points = oe_points.set_coords(["z", "type"])

    logging.info("Converting Origins.earth point source emissions to kg CO2 / h")
    oe_points = convert_oe_point_units(oe_points)

    oe_points = oe_points.drop_vars(
        [
            "x",
            "y",
            "lat",
            "lon",
            "new_lat",
            "new_lon",
            "relative_emissions",
            "elevation",
        ]
    )
    oe_points = oe_points.rename({"new_x": "x", "new_y": "y"})

    logging.info("Assign default exit velocity, stack diameter, and exit temperature")
    logging.info("based on Pregger and Friedrich 2009.")
    oe_points["exit_velocity"] = ("index", np.full(oe_points.sizes["index"], 5.0))
    oe_points["stack_diameter"] = ("index", np.full(oe_points.sizes["index"], 1.0))
    oe_points["exit_temperature"] = (
        "index",
        np.full(oe_points.sizes["index"], 150 + 273.15),
    )

    # Add metadata
    timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    oe_points.attrs["description"] = f"Dataset compiled {timestamp} EDT"
    oe_points["x"].attrs = {
        "long_name": "X coordinate of projection",
        "standard_name": "projection_x_coordinate",
        "units": "m",
    }
    oe_points["y"].attrs = {
        "long_name": "Y coordinate of projection",
        "standard_name": "projection_y_coordinate",
        "units": "m",
    }
    oe_points["z"].attrs = {
        "long_name": "Height above ground level",
        "standard_name": "height_above_ground",
        "units": "m",
    }
    oe_points["index"].attrs = {
        "long_name": "Point source index",
        "description": "Unique index for each point source",
        "units": "1",
    }
    oe_points["exit_velocity"].attrs = {
        "long_name": "Exit velocity",
        "description": "Exit velocity of the point source plume",
        "units": "m/s",
    }
    oe_points["stack_diameter"].attrs = {
        "long_name": "Stack diameter",
        "description": "Diameter of the point source stack",
        "units": "m",
    }
    oe_points["exit_temperature"].attrs = {
        "long_name": "Exit temperature",
        "description": "Exit temperature of the point source plume",
        "units": "K",
    }
    if not OE_POINT_NETCDF_PATH.exists():
        logging.info(f"Saving Origins.earth data to {OE_POINT_NETCDF_PATH}")
        oe_points.to_netcdf(OE_POINT_NETCDF_PATH)
    else:
        logging.info(f"File {OE_POINT_NETCDF_PATH} exists, skipping save.")


def convert_tno_area_units(tno):
    """Convert kg CO2 / year to kq CO2 / m2 / h"""
    days_in_2018 = 365
    area = tno["area"].isel(
        latitude=tno.latitude_index - 1, longitude=tno.longitude_index - 1
    )  # m2
    for var in tno.data_vars:
        if str(var).find("co2") != -1:
            logging.info(f"Converting units for {var}.")
            attrs = tno[var].attrs
            assert attrs["units"] == "kg/year"
            tno[var] = tno[var] / days_in_2018 / 24 / area
            tno[var].attrs = {
                "long_name": attrs["long_name"],
                "units": "kg CO2 m-2 h-1",
            }
    return tno


def create_tno_area_fluxes():
    if TNO_AREA_NETCDF_PATH.exists():
        logging.info(f"{TNO_AREA_NETCDF_PATH} exists, skipping creation.")
        return
    gral_grid = ggp.utils.create_domain_grid("gral", CONFIG)

    logging.info("Loading TNO data...")
    tno = xr.open_mfdataset(
        Path(CONFIG["data_path"]) / "Fluxes/TNO/TNO_GHGco_v4_1_highres_year2018.nc"
    )
    logging.info(f"Number of sources in TNO: {tno.source.size}")

    france_index = np.where(tno.country_name == b"France")[0] + 1
    france_mask = (tno.country_index == france_index).compute()
    logging.info(f"Number of sources in France: {france_mask.sum().item()}")

    tno = tno.sel(source=france_mask).load()
    area_index = np.where(tno.source_type_name == b"area")[0] + 1
    area_mask = (tno.source_type_index == area_index).compute()
    tno = tno.sel(source=area_mask)
    logging.info(f"Number of area sources in France: {area_mask.sum().item()}")

    domain_area = ggp.utils.create_domain_geometry("gral", CONFIG)
    min_lon, min_lat, max_lon, max_lat = (
        domain_area.buffer(1e4).to_crs("EPSG:4326").total_bounds
    )
    tno = tno.sel(
        source=(
            (tno.longitude_source >= min_lon)
            & (tno.longitude_source <= max_lon)
            & (tno.latitude_source >= min_lat)
            & (tno.latitude_source <= max_lat)
        )
    )
    logging.info(f"Number of area sources in domain: {tno.source.size}")

    tno = tno.set_coords(
        [
            "country_id",
            "country_name",
            "emis_cat_code",
            "emis_cat_name",
            "source_type_code",
            "source_type_name",
            "longitude_bounds",
            "latitude_bounds",
            "area",
            "longitude_source",
            "latitude_source",
            "longitude_index",
            "latitude_index",
            "country_index",
            "emission_category_index",
            "source_type_index",
        ]
    )

    tno = convert_tno_area_units(tno)

    logging.info("Converting TNO list of sources to gridded fluxes...")
    sector_grids = []
    for sector in tno.emis_cat_name.values:
        sector_index = np.where(tno.emis_cat_name == sector)[0] + 1
        sector_mask = tno.emission_category_index == sector_index
        logging.info(
            f"Number of sources in sector {sector.decode()}: {sector_mask.sum().item()}"
        )
        tno_sector = tno.sel(source=sector_mask)
        tno_sector = tno_sector.expand_dims({"sector": [sector]})
        logging.info(
            f"Total emissions in sector {sector.decode()}: "
            f"{(tno_sector["co2_ff"]+tno_sector["co2_bf"]).sum().item()/1e3:.2f} kt/yr"
        )
        if tno_sector.source.size == 0:
            logging.info(f"No sources for sector {sector} in domain.")
        else:
            sector_grids.append(
                tno_sector.groupby(["longitude_index", "latitude_index"]).sum()
            )

    tno_for_domain = xr.concat(sector_grids, dim="sector")
    tno_for_domain["co2"] = tno_for_domain.co2_ff + tno_for_domain.co2_bf
    tno_for_domain["sector"] = tno_for_domain["sector"].astype(str)
    tno_for_domain["longitude_index"] = tno.longitude[
        tno_for_domain.longitude_index - 1
    ]
    tno_for_domain["latitude_index"] = tno.latitude[tno_for_domain.latitude_index - 1]
    tno_for_domain = tno_for_domain.rename(
        {"longitude_index": "lon", "latitude_index": "lat"}
    ).rio.write_crs("EPSG:4326")
    tno_for_domain = tno_for_domain.transpose("sector", "lat", "lon", ...)

    SECTOR_GROUPS = {
        "Public Power": "Power",
        "Industry": "Industry",
        "Other Stationary Combustion": "Combustion",
        "Fugitives": "Industry",
        "Solvents": "Solvents",
        "RoadTransport exhaust gasoline": "Traffic",
        "RoadTransport exhaust diesel": "Traffic",
        "RoadTransport exhaust LPG gas": "Traffic",
        "RoadTransport non-exhaust": "Traffic",
        "Shipping": "Traffic",
        "OffRoad": "Traffic",
        "Waste": None,
        "Agricultural Livestock": None,
        "Agricultural Other": None,
    }

    logging.info(
        "Contribution of each TNO sector in percent to the total CO2 emissions"
    )
    co2_contribution_series = (
        ((tno_for_domain.co2.sum(("lat", "lon")) / tno_for_domain.co2.sum()) * 100)
        .drop_vars("spatial_ref")
        .to_dataframe(name="Relative contribution of CO2 [%]")
    )

    co2_contribution_series["Group"] = SECTOR_GROUPS
    logging.info(co2_contribution_series)

    logging.info("Grouping the sectors into larger groups.")
    tno_for_domain["sector_group"] = xr.DataArray(
        [v for v in SECTOR_GROUPS.values()],
        dims=("sector",),
        coords={"sector": [k for k in SECTOR_GROUPS.keys()]},
    )
    tno_for_domain = tno_for_domain.groupby("sector_group").sum()
    tno_gral = tno_for_domain.co2.rio.reproject_match(
        gral_grid, resampling=Resampling.average
    )
    tno_gral = tno_gral.to_dataset(name="flux").rename({"sector_group": "type"})
    tno_gral["type"] = [f"TNO 2018 {type_}" for type_ in tno_gral["type"].values]
    tno_gral["flux"].attrs = {
        "long_name": "Annual mean CO2 flux",
        "units": "kg CO2 m-2 h-1",
    }
    tno_gral["type"].attrs = {"long_name": "Flux type"}
    tno_gral = tno_gral.set_coords(["spatial_ref"])
    timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    tno_gral.attrs["description"] = f"Dataset compiled {timestamp} EDT"

    if not TNO_AREA_NETCDF_PATH.exists():
        logging.info(f"Saving TNO data to {TNO_AREA_NETCDF_PATH}")
        logging.info(tno_gral)
        tno_gral.to_netcdf(TNO_AREA_NETCDF_PATH)
    else:
        logging.info(f"File {TNO_AREA_NETCDF_PATH} exists, skipping save.")


def convert_tno_point_units(tno):
    flux_attrs = tno["flux"].attrs
    assert flux_attrs["units"] == "kg CO2 / year"
    days_in_2018 = 365
    tno["flux"] = tno["flux"] / (days_in_2018 * 24)
    flux_attrs["units"] = "kg CO2 h-1"
    tno["flux"].attrs = flux_attrs
    return tno


def create_tno_point_fluxes():
    if TNO_POINT_NETCDF_PATH.exists():
        logging.info(f"{TNO_POINT_NETCDF_PATH} exists, skipping creation.")
        return
    logging.info("Loading TNO data...")
    tno = xr.open_mfdataset(
        Path(CONFIG["data_path"]) / "Fluxes/TNO/TNO_GHGco_v4_1_highres_year2018.nc"
    )
    logging.info(f"Number of sources in TNO: {tno.source.size}")

    france_index = np.where(tno.country_name == b"France")[0] + 1
    france_mask = (tno.country_index == france_index).compute()
    logging.info(f"Number of sources in France: {france_mask.sum().item()}")

    tno = tno.sel(source=france_mask).load()
    point_index = np.where(tno.source_type_name == b"point")[0] + 1
    point_mask = (tno.source_type_index == point_index).compute()
    tno = tno.sel(source=point_mask)
    logging.info(f"Number of point sources in France: {tno.source.size}")
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", CONFIG["domain"]["crs"], always_xy=True
    )
    x, y = transformer.transform(tno.longitude_source, tno.latitude_source)

    tno["x"] = ("source", x)
    tno["y"] = ("source", y)

    tno = tno.sel(
        source=(
            (tno.x >= CONFIG["domain"]["gral"]["bbox"]["x0"])
            & (tno.x <= CONFIG["domain"]["gral"]["bbox"]["x1"])
            & (tno.y >= CONFIG["domain"]["gral"]["bbox"]["y0"])
            & (tno.y <= CONFIG["domain"]["gral"]["bbox"]["y1"])
        )
    )
    logging.info(f"Number of point sources in domain: {tno.source.size}")

    logging.info("Converting TNO units from kg CO2 / year to kg CO2 / h...")
    tno["flux"] = tno["co2_ff"] + tno["co2_bf"]
    tno["flux"].attrs = {
        "long_name": "CO2 emissions from fossil fuel and biofuel combustion",
        "units": "kg CO2 / year",
    }
    tno = convert_tno_point_units(tno)
    tno["type"] = (
        "source",
        tno["emis_cat_name"]
        .isel(emis_cat=tno["emission_category_index"] - 1)
        .astype(str)
        .values,
    )

    tno["type"] = ("source", [f"TNO 2018 {type_}" for type_ in tno["type"].values])
    tno = tno.rename({"source": "index"})
    tno["index"] = np.arange(tno.sizes["index"])
    tno = tno[["x", "y", "flux", "type"]]
    tno = tno.rio.write_crs(CONFIG["domain"]["crs"])
    # Heights and new positions from Google Earth
    # For some sources, there are two chimneys, so we split the flux
    new_coords = {
        0: (48.832178, 2.263969),
        1: ((48.822014, 2.388833), (48.822888, 2.388003)),
        2: ((48.791998, 2.416504), (48.791153, 2.416794)),
        3: (48.903622, 2.452672),
        4: None,  # Correct
        5: None,  # Correct
        6: (48.897583, 2.232488),
        7: None,  # Correct
        8: ((48.842797, 2.373920), (48.842392, 2.373184)),
        9: (48.816014, 2.407023),
        10: (48.866384, 2.420180),
        11: None,  # Correct
        12: None,  # Correct
        13: (48.834659, 2.310637),
    }

    # m asl
    heights = {
        0: 60,
        1: (114, 114),
        2: (195, 195),
        3: 90,
        4: 130,
        5: 75,
        6: 101,
        7: 70,
        8: (113, 113),
        9: 150,
        10: 137,
        11: 165,
        12: 90,
        13: 150,
    }

    logging.info("Creating figure to check new coordinates...")
    create_tno_point_figure(tno, transformer, new_coords)

    logging.info("Updating TNO coordinates and heights based on Google Earth data...")
    tno["elevation"] = ("index", np.full(tno.sizes["index"], np.nan))
    for idx, new_coord in new_coords.items():
        if new_coord is None:
            pass
        elif isinstance(new_coord[0], float):
            lat = new_coord[0]
            lon = new_coord[1]
            x, y = transformer.transform(lon, lat)
            tno["x"].loc[idx] = x
            tno["y"].loc[idx] = y
            tno["elevation"].loc[idx] = heights[idx]  # in m asl
        elif isinstance(new_coord[0], tuple):
            tno["flux"].loc[idx] /= 2
            lat1 = new_coord[0][0]
            lon1 = new_coord[0][1]
            x1, y1 = transformer.transform(lon1, lat1)
            lat2 = new_coord[1][0]
            lon2 = new_coord[1][1]
            x2, y2 = transformer.transform(lon2, lat2)
            tno["x"].loc[idx] = x1
            tno["y"].loc[idx] = y1
            tno["elevation"].loc[idx] = heights[idx][0]  # in m asl

            new_point_source = tno.isel(index=idx).copy()
            new_point_source["x"] = x2
            new_point_source["y"] = y2
            new_point_source["elevation"] = heights[idx][1]
            new_point_source["index"] = tno.index.max().item() + 1
            tno = xr.concat([tno, new_point_source], dim="index")
        else:
            raise ValueError("Unexpected type in new_coords")
    tno = tno.sel(
        index=(
            (tno.x >= CONFIG["domain"]["gral"]["bbox"]["x0"])
            & (tno.x <= CONFIG["domain"]["gral"]["bbox"]["x1"])
            & (tno.y >= CONFIG["domain"]["gral"]["bbox"]["y0"])
            & (tno.y <= CONFIG["domain"]["gral"]["bbox"]["y1"])
        )
    )
    logging.info(f"Number of point sources in domain: {tno.index.size}")

    terrain = xr.open_dataset(model_input.terrain.GRAL_TERRAIN_PATH)
    gral_elevation = terrain.elevation.sel(x=tno.x, y=tno.y, method="nearest")
    tno["z"] = tno["elevation"] - gral_elevation.values
    tno = tno.set_coords(["x", "y", "z", "type"]).drop_vars("elevation")
    logging.info("Assign default exit velocity, stack diameter, and exit temperature")
    logging.info("based on Pregger and Friedrich 2009.")
    tno["exit_velocity"] = ("index", np.full(tno.sizes["index"], 5.0))
    tno["stack_diameter"] = ("index", np.full(tno.sizes["index"], 1.0))
    tno["exit_temperature"] = (
        "index",
        np.full(tno.sizes["index"], 150 + 273.15),
    )

    # Add metadata
    timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    tno.attrs = {"description": f"Dataset compiled {timestamp} EDT"}
    tno["x"].attrs = {
        "long_name": "X coordinate of projection",
        "standard_name": "projection_x_coordinate",
        "units": "m",
    }
    tno["y"].attrs = {
        "long_name": "Y coordinate of projection",
        "standard_name": "projection_y_coordinate",
        "units": "m",
    }
    tno["z"].attrs = {
        "long_name": "Height above ground level",
        "standard_name": "height_above_ground",
        "units": "m",
    }
    tno["index"].attrs = {
        "long_name": "Point source index",
        "description": "Unique index for each point source",
        "units": "1",
    }
    tno["exit_velocity"].attrs = {
        "long_name": "Exit velocity",
        "description": "Exit velocity of the point source plume",
        "units": "m/s",
    }
    tno["stack_diameter"].attrs = {
        "long_name": "Stack diameter",
        "description": "Diameter of the point source stack",
        "units": "m",
    }
    tno["exit_temperature"].attrs = {
        "long_name": "Exit temperature",
        "description": "Exit temperature of the point source plume",
        "units": "K",
    }
    if not TNO_POINT_NETCDF_PATH.exists():
        logging.info(f"Saving Origins.earth data to {TNO_POINT_NETCDF_PATH}")
        tno.to_netcdf(TNO_POINT_NETCDF_PATH)
    else:
        logging.info(f"File {TNO_POINT_NETCDF_PATH} exists, skipping save.")


def create_tno_point_figure(tno, transformer, new_coords):
    for idx, new_coord in new_coords.items():
        plt.plot(tno.x[idx].values, tno.y[idx].values, "go")
        if new_coord is None:
            continue
        elif isinstance(new_coord[0], tuple):
            lat = [new_coord[0][0]]
            lon = [new_coord[0][1]]
            x, y = transformer.transform(lon, lat)
            plt.plot(
                [tno.x[idx].values] + x,
                [tno.y[idx].values] + y,
                "ro-",
            )
            lat = [new_coord[1][0]]
            lon = [new_coord[1][1]]
        else:
            lat = [new_coord[0]]
            lon = [new_coord[1]]

        x, y = transformer.transform(lon, lat)
        plt.plot(
            [tno.x[idx].values] + x,
            [tno.y[idx].values] + y,
            "ro-",
        )
        plt.plot(tno.x[idx].values, tno.y[idx].values, "go")
    plt.legend(["Old position", "New position"])
    plt.title("TNO point sources showing old and new positions")
    if not FIGURE_PATHS["tno_point_fluxes"].exists():
        logging.info(
            f"Saving TNO point source figure to {FIGURE_PATHS['tno_point_fluxes']}"
        )
        plt.savefig(FIGURE_PATHS["tno_point_fluxes"])
    plt.close()


def plot_area_fluxes():
    if not FIGURE_PATHS["area_fluxes"].exists():
        logging.info(f"Plotting area fluxes to {FIGURE_PATHS['area_fluxes']}")
        xr.open_mfdataset(
            str(Path(CONFIG["data_path"]) / "Fluxes/area_flux_*.nc"),
            decode_coords="all",
        ).flux.plot(col="type", col_wrap=4, vmax=0.001)
        plt.savefig(FIGURE_PATHS["area_fluxes"])
    else:
        logging.info(f"File {FIGURE_PATHS['area_fluxes']} exists, skipping plot.")


def create_source_group_dataset():
    if SOURCE_GROUP_NETCDF_PATH.exists():
        logging.info(f"{SOURCE_GROUP_NETCDF_PATH} exists, skipping creation.")
        return
    areas = (
        xr.open_mfdataset(
            [
                VPRM_2023_R_NETCDF_PATH,
                VPRM_2023_GEE_NETCDF_PATH,
                VPRM_2024_R_NETCDF_PATH,
                VPRM_2024_GEE_NETCDF_PATH,
                OE_AREA_NETCDF_PATH,
                TNO_AREA_NETCDF_PATH,
            ],
            decode_coords="all",
        )
        .astype(np.float32)
        .load()
        .chunk({"type": 15})
    )
    points = xr.open_mfdataset(
        [
            OE_POINT_NETCDF_PATH,
            TNO_POINT_NETCDF_PATH,
        ],
        concat_dim="index",
        combine="nested",
    )
    flux_grid = xr.open_dataset(AREA_ID_NETCDF_PATH)

    logging.info("Combining point and area sources into one dataset...")
    cadastre = (
        xr.concat(
            [areas.where(flux_grid.area_id == i) for i in np.unique(flux_grid.area_id)],
            dim="area_id",
        )
        .assign_coords(area_id=np.unique(flux_grid.area_id))
        .compute()
    )
    cadastre = cadastre.stack(
        source_group=("area_id", "type"),
    )
    cadastre = cadastre.reset_index("source_group")
    cadastre["source_group"] = (
        np.arange(len(cadastre["source_group"])) + 1 + len(points.index)
    )  # continue index from points and start with 1
    cadastre["source_flux"] = cadastre["flux"].sum(["x", "y"]) * 100
    cadastre = cadastre.rename({"flux": "area_flux"})
    cadastre["geometry"] = ("source_group", ["area"] * len(cadastre.source_group))

    points = points.rename(
        {
            "flux": "source_flux",
            "x": "x_point",
            "y": "y_point",
            "z": "z_point",
            "index": "source_group",
        }
    )
    points["source_group"] = np.arange(len(points["source_group"])) + 1  # start with 1
    points["geometry"] = ("source_group", ["point"] * len(points.source_group))

    source_group_ds = xr.merge([points, cadastre])
    source_group_ds["source_flux"].attrs = {
        "long_name": "Source flux",
        "units": "kg CO2 h-1",
        "description": "Flux from point and area sources combined",
    }
    source_group_ds["source_group"].attrs = {
        "long_name": "Source group index",
        "description": "Index for point and area sources combined, starting with 1",
    }
    source_group_ds["geometry"].attrs = {
        "long_name": "Geometry type",
        "description": "Geometry type of the source, either point or area",
    }
    timestamp = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    source_group_ds.attrs = {
        "title": "Combined point and area sources from TNO, OE, and VPRM",
        "description": f"Dataset compiled {timestamp} EDT",
        "source": "TNO 2018, Origins.earth 2021, VPRM 2023/2024",
    }
    if not Path(SOURCE_GROUP_NETCDF_PATH).exists():
        logging.info(f"Saving combined source groups to {SOURCE_GROUP_NETCDF_PATH}...")
        with ProgressBar():
            source_group_ds.to_netcdf(SOURCE_GROUP_NETCDF_PATH)
    else:
        logging.info(f"File {SOURCE_GROUP_NETCDF_PATH} exists, skipping save.")


def convert_area_flux_to_absolute_units(ds):
    attrs = ds["area_flux"].attrs
    assert attrs["units"] == "kg CO2 m-2 h-1", f"Unexpected units: {attrs['units']}"
    ds["area_flux"] = ds["area_flux"] * 10 * 10
    attrs["units"] = "kg CO2 h-1"
    ds["area_flux"].attrs = attrs
    return ds


def create_cadastre_dat_from_area(path):
    if Path(path).exists():
        logging.info(f"{path} exists, skipping creation.")
        return
    logging.info("Loading source group dataset...")
    source_group_ds = xr.open_mfdataset(
        SOURCE_GROUP_NETCDF_PATH, chunks={"source_group": 200}
    )
    flux_grid = xr.open_dataset(AREA_ID_NETCDF_PATH)

    logging.info("Combining area fluxes...")
    ds_list = []
    for area_id in np.unique(flux_grid["area_id"]):
        ds_list.append(
            source_group_ds.drop_vars(
                ["type", "x_point", "y_point", "z_point", "spatial_ref"]
            )
            .where((source_group_ds["area_id"] == area_id).compute(), drop=True)
            .where(flux_grid["area_id"] == area_id, drop=True)[
                ["x", "y", "source_group", "area_flux"]
            ]
            .stack(index=["y", "x", "source_group"], create_index=False)
        )
    with ProgressBar():
        area_fluxes = xr.concat(ds_list, dim="index").compute()

    logging.info("Converting area fluxes to absolute fluxes...")
    area_fluxes = convert_area_flux_to_absolute_units(area_fluxes)
    total_area_flux = (
        source_group_ds["source_flux"]
        .sel(source_group=source_group_ds["geometry"] == "area")
        .sum()
        .compute()
    )
    processed_total_area_flux = area_fluxes["area_flux"].sum().compute()
    logging.info(
        "Total area flux in processed area_fluxes: "
        f"{processed_total_area_flux.values:.2f} "
        f"{area_fluxes["area_flux"].attrs['units']}"
    )
    assert np.isclose(
        total_area_flux.values,
        processed_total_area_flux.values,
    ), "Total area flux does not match!"

    logging.info("Filter out small area fluxes...")
    logging.info("Drop zero area fluxes...")
    area_fluxes_filtered = area_fluxes.where(area_fluxes["area_flux"] > 0, drop=True)
    logging.info("Drop small area fluxes...")
    ascending = area_fluxes_filtered.sortby("area_flux", ascending=True)["area_flux"]
    cumsum = np.cumsum(ascending)
    cumsum_ratio = 0.02
    cut_off = ascending[cumsum < cumsum_ratio * cumsum[-1]].max()
    logging.info(
        "Cut-off for area fluxes: "
        f"{cut_off.values:.2f} {area_fluxes['area_flux'].attrs['units']}"
    )
    logging.info(
        f"Number of area fluxes before filtering: {len(area_fluxes['area_flux'])}"
    )
    cut_off_mask = area_fluxes_filtered["area_flux"] >= cut_off
    area_fluxes_filtered_cut_off = area_fluxes_filtered.where(cut_off_mask, drop=True)
    logging.info(
        "Number of area fluxes after filtering: "
        f"{len(area_fluxes_filtered_cut_off['area_flux'])}"
    )

    missing_fluxes = (
        area_fluxes_filtered.sel(index=~cut_off_mask).groupby("source_group").sum()
    )
    missing_fluxes["type"] = (
        source_group_ds["type"].sel(source_group=missing_fluxes["source_group"]).load()
    )
    missing_fluxes = missing_fluxes.groupby("type").sum()

    logging.info("Creating plots...")
    plot_cumulative_fluxes(cumsum, cumsum_ratio)
    plot_missing_flux_percentage(source_group_ds, cumsum_ratio, missing_fluxes)
    plot_missing_flux(cumsum_ratio, missing_fluxes)

    df = area_fluxes_filtered_cut_off.to_dataframe()
    df["z"] = 0
    df["dx"] = 10
    df["dy"] = 10
    df["dz"] = 0
    ggp.utils.write_cadastre_dat(
        path=path,
        x=df["x"],
        y=df["y"],
        z=df["z"],
        dx=df["dx"],
        dy=df["dy"],
        dz=df["dz"],
        flux=df["area_flux"].round(4),
        source_group=df["source_group"],
    )


def plot_missing_flux(cumsum_ratio, missing_fluxes):
    plt.figure()
    missing_fluxes.load()["area_flux"].to_dataframe()["area_flux"].plot.bar()
    plt.ylabel("Missing flux [kg CO2 h-1]")
    plt.title(
        f"Missing fluxes by type after cut-off of small area fluxes by "
        f"{cumsum_ratio*100:.1f}%"
    )
    if not FIGURE_PATHS["missing_flux"].exists():
        logging.info(f"Saving missing flux figure to {FIGURE_PATHS['missing_flux']}")
        plt.savefig(FIGURE_PATHS["missing_flux"])
    else:
        logging.info(f"File {FIGURE_PATHS['missing_flux']} exists, skipping save.")
    plt.close()


def plot_missing_flux_percentage(source_group_ds, cumsum_ratio, missing_fluxes):
    plt.figure()
    (
        (missing_fluxes / source_group_ds["source_flux"].load().groupby("type").sum())
        * 100
    ).load()["area_flux"].to_dataframe()["area_flux"].plot.bar()
    plt.ylabel("% of total flux missing")
    plt.title(
        "Missing fluxes by type after cut-off of small area fluxes by "
        f"{cumsum_ratio*100:.1f}%"
    )
    if not FIGURE_PATHS["missing_flux_percentage"].exists():
        logging.info(
            f"Saving missing flux percentage figure to "
            f"{FIGURE_PATHS['missing_flux_percentage']}"
        )
        plt.savefig(FIGURE_PATHS["missing_flux_percentage"])
    else:
        logging.info(
            f"File {FIGURE_PATHS['missing_flux_percentage']} exists, skipping save."
        )
    plt.close()


def plot_cumulative_fluxes(cumsum, cumsum_ratio):
    plt.figure()
    plt.plot(cumsum / cumsum[-1], label="Cumulative sum")
    plt.hlines(
        cumsum_ratio,
        0,
        len(cumsum),
        colors="r",
        linestyles="dashed",
        label=f"Cut-off ratio at {cumsum_ratio*100:.0f}%",
    )
    plt.xlabel("Number of area fluxes (sorted ascending)")
    plt.ylabel("Cumulative sum of area fluxes (normalized)")
    plt.yscale("log")
    plt.grid()
    plt.legend(loc="lower right")
    plt.title("Cumulative sum of area fluxes")
    if not FIGURE_PATHS["cumulative_fluxes"].exists():
        logging.info(
            f"Saving cumulative area fluxes figure to "
            f"{FIGURE_PATHS['cumulative_fluxes']}"
        )
        plt.savefig(FIGURE_PATHS["cumulative_fluxes"])
    else:
        logging.info(f"File {FIGURE_PATHS['cumulative_fluxes']} exists, skipping save.")
    plt.close()


def process_fluxes() -> Path:
    create_area_partitioning()

    create_vprm_area_fluxes()
    create_oe_area_fluxes()
    create_oe_point_fluxes()
    create_tno_area_fluxes()
    create_tno_point_fluxes()

    plot_area_fluxes()

    create_source_group_dataset()
    return SOURCE_GROUP_NETCDF_PATH
