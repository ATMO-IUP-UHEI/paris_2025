"""Miscellaneous plotting functions."""

from pathlib import Path

import colorcet  # noqa: F401
import ggpymanager as ggp
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import paris_2025 as p
from paris_2025.config import CONFIG
from paris_2025.plotting.common import get_metadata


def plot_temperature_anomaly_with_co2(
    fig_path: str | Path,
    time_period: str | slice = slice("2023-09-15", "2023-10"),
    co2_station: str = "JUS_30",
    temp_station: str = "TOUR EIFFEL",
):
    """Plot temperature anomaly heatmap with CO2 and temperature comparison.

    Creates a two-panel figure showing:
    - Temperature anomaly across all meteorological stations
    - CO2 concentration at a reference station with temperature anomaly
      at a specific station on a twin axis

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    time_period : str | slice, optional
        Time period for the analysis. Default is slice("2023-09-15", "2023-10")
    co2_station : str, optional
        Station code for CO2 measurements. Default is "JUS_30"
    temp_station : str, optional
        Station name for temperature comparison. Default is "TOUR EIFFEL"
    """
    # Load data
    co2_data = ggp.load("co2_measurements", CONFIG)
    co2_station_data = co2_data.sel(station=co2_station, time=time_period).co2

    temperature = p.meteo.get_meteo_measurements().sel(time=time_period).temperature
    # Drop stations with missing temperature data
    temperature = temperature.dropna("station")

    # Calculate temperature anomaly
    temp_anomaly = temperature - temperature.mean("station")

    # Create figure
    gridspec = {"width_ratios": [1, 0.02]}
    fig, axs = plt.subplots(2, 2, figsize=(18, 6), gridspec_kw=gridspec, sharex="col")

    # Top panel: Temperature anomaly heatmap
    temp_anomaly.plot(
        x="time",
        ax=axs[0, 0],
        cbar_kwargs={"label": "Temperature anomaly [°C]", "cax": axs[0, 1]},
    )

    # Bottom panel: CO2 with temperature on twin axis
    co2_station_data.plot(x="time", ax=axs[1, 0])
    twin_ax = axs[1, 0].twinx()
    temp_station_anomaly = temp_anomaly.sel(station=temp_station)
    temp_station_anomaly.plot(ax=twin_ax, c="red", label="Temperature anomaly")
    twin_ax.set_ylim(-5, 15)
    twin_ax.set_ylabel("Temperature anomaly [°C]", color="red")
    twin_ax.tick_params(axis="y", labelcolor="red")
    twin_ax.set_title("")

    axs[1, 0].set_title(
        f"CO2 at {co2_station} and temperature anomaly at {temp_station}"
    )
    axs[1, 0].set_ylabel("CO2 [ppm]")
    axs[1, 0].grid(axis="x")

    # Remove empty subplot
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Temperature anomaly heatmap with CO2 at {co2_station} "
            f"and temperature at {temp_station}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def _load_terrain_elevation_data():
    """Load terrain and building height data from GGpyManager.

    Returns
    -------
    tuple
        (building_height, gramm_terrain, gral_terrain, gramm_albedo)
    """
    building_height = ggp.load("gral_buildings", CONFIG).building_height
    gramm_terrain = ggp.load("gramm_terrain", CONFIG).elevation
    gral_terrain = ggp.load("gral_terrain", CONFIG).elevation
    gramm_albedo = ggp.load("gramm_landcover", CONFIG).ALBEDO.T
    return building_height, gramm_terrain, gral_terrain, gramm_albedo


def plot_terrain_elevation_maps(
    fig_path: str | Path,
    vert_exag: float = 10.0,
):
    """Plot terrain elevation maps with hillshading for GRAMM and GRAL models.

    Creates a 2x2 figure with hillshaded terrain maps showing:
    - GRAMM terrain elevation
    - GRAMM albedo
    - GRAL terrain elevation
    - GRAL building height

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    vert_exag : float, optional
        Vertical exaggeration factor for hillshading. Default is 10.0
    """
    # Load data
    building_height, gramm_terrain, gral_terrain, gramm_albedo = (
        _load_terrain_elevation_data()
    )

    # Helper function to add custom colorbars
    def add_custom_cbar(ax, data, cmap, label="Elevation (m)"):
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=data.min(), vmax=data.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(label)

    # Create light source for hillshading
    light = mcolors.LightSource(azdeg=315, altdeg=45)

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

    # GRAMM terrain
    axs[0, 0].set_title("GRAMM Terrain", loc="center")
    gramm_shaded = light.shade(
        gramm_terrain.values,
        cmap=plt.get_cmap("cet_CET_L10"),
        blend_mode="soft",
        dx=100,
        dy=100,
        vert_exag=vert_exag,
    )
    axs[0, 0].imshow(gramm_shaded, origin="lower")
    add_custom_cbar(axs[0, 0], gramm_terrain.values, cmap=plt.get_cmap("cet_CET_L10"))

    # GRAMM albedo
    gramm_albedo.plot(cmap="magma", ax=axs[0, 1])
    axs[0, 1].set_title("GRAMM Albedo", loc="center")

    # GRAL terrain
    axs[1, 0].set_title("GRAL Terrain", loc="center")
    gral_shaded = light.shade(
        gral_terrain.values,
        cmap=plt.get_cmap("cet_CET_L10"),
        blend_mode="soft",
        dx=10,
        dy=10,
        vert_exag=vert_exag,
    )
    axs[1, 0].imshow(gral_shaded, origin="lower")
    add_custom_cbar(axs[1, 0], gral_terrain.values, cmap=plt.get_cmap("cet_CET_L10"))

    # GRAL building height
    axs[1, 1].set_title("GRAL Building Height", loc="center")
    building_shaded = light.shade(
        building_height.fillna(0.0).values,
        cmap=plt.get_cmap("inferno"),
        blend_mode="hsv",
        dx=10,
        dy=10,
        vert_exag=vert_exag,
    )
    axs[1, 1].imshow(building_shaded, origin="lower")
    add_custom_cbar(
        axs[1, 1],
        building_height.fillna(0.0).values,
        cmap=plt.get_cmap("inferno"),
        label="Building Height (m)",
    )

    # Format subplots
    for ax, index in zip(axs.flatten(), "abcd"):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"({index})", loc="left")

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Terrain elevation maps with hillshading for GRAMM and GRAL models, "
            "including GRAMM albedo and GRAL building height."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
