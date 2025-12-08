import logging
from pathlib import Path

import matplotlib.pyplot as plt

import paris_2025 as p
import paris_2025.plotting as plotting

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

YEAR = 2023
FIGURE_PATH = Path(p.CONFIG["figures_path"])

# Set matplotlib style like size of the figures, text size, etc. by hand
# sns.set_context("paper", font_scale=1)
plt.rcParams.update(
    {
        "figure.figsize": (7, 5),
        # "axes.titlesize": 16,
        # "axes.labelsize": 14,
        # "xtick.labelsize": 12,
        # "ytick.labelsize": 12,
        # "legend.fontsize": 12,
        # "lines.linewidth": 1.5,
        # "grid.linewidth": 0.5,
        # "grid.alpha": 0.5,
        # "font.family": "sans-serif",
        # "font.sans-serif": "Arial",
        # "axes.grid": True,
        # "axes.spines.right": False,
        # "axes.spines.top": False,
        "savefig.dpi": 300,
    }
)


def create_figures_if_missing(fig_paths, plot_function):
    """Create figures only if any of them don't exist."""
    if isinstance(fig_paths, (list, tuple)):
        if any(not path.exists() for path in fig_paths):
            logging.info(f"Creating figures:\n{[str(p) for p in fig_paths]}")
            plot_function(*fig_paths)
        else:
            logging.info(
                "Figures already exist:\n"
                + ("\n".join([str(p) for p in fig_paths if p.exists()]))
            )
    else:
        if not fig_paths.exists():
            logging.info(f"Creating figure:\n{fig_paths}")
            plot_function(fig_paths)
        else:
            logging.info(f"Figure already exists:\n{fig_paths}")


if __name__ == "__main__":
    # Concentration from catalog
    DIR = "concentration_from_catalog"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        [
            FIGURE_PATH / DIR / "source_group_contribution_stations.png",
            FIGURE_PATH / DIR / "source_group_area_contribution_stations.png",
        ],
        plotting.tracer_from_catalog.plot_source_group_contribution_to_stations,
    )

    # Wind measurements
    DIR = "wind_measurements"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        FIGURE_PATH / DIR / "wind_roses_measurements.png",
        p.plotting.meteo_measurements.plot_wind_roses_of_meteo_measurements,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "meteo_overview.png",
        p.plotting.meteo_measurements.plot_meteo_overview,
    )
    create_figures_if_missing(
        [
            FIGURE_PATH / DIR / "wind_data_availability_2023.png",
            FIGURE_PATH / DIR / "wind_data_availability_2024.png",
        ],
        p.plotting.meteo_measurements.plot_wind_data_availability,
    )

    # CO2 measurements
    DIR = "co2_measurements"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        FIGURE_PATH / DIR / "co2_concentration_heatmap.png",
        p.plotting.tracer_measurements.plot_co2_concentration_heatmap,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "co2_by_height_and_instrument.png",
        p.plotting.tracer_measurements.plot_co2_by_height_and_instrument,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "average_co2_spatial.png",
        p.plotting.tracer_measurements.plot_average_co2_spatial,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "co2_instruments_map.png",
        p.plotting.tracer_measurements.plot_co2_instruments_map,
    )
    create_figures_if_missing(
        [
            FIGURE_PATH / DIR / "co2_data_availability_2023.png",
            FIGURE_PATH / DIR / "co2_data_availability_2024.png",
        ],
        p.plotting.tracer_measurements.plot_co2_data_availability,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "picarro_co2_violin_2023.png",
        p.plotting.tracer_measurements.plot_picarro_co2_violin,
    )

    # CO2 background analysis
    DIR = "co2_background"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        FIGURE_PATH / DIR / "mean_windrose_2023.png",
        p.plotting.tracer_background.plot_mean_windrose,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "co2_stations_with_windrose_2023.png",
        p.plotting.tracer_background.plot_co2_stations_with_windrose,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "background_co2_stations_2023.png",
        p.plotting.tracer_background.plot_background_co2_stations,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "background_station_counts_2023.png",
        p.plotting.tracer_background.plot_background_station_counts,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "background_station_co2_violin_2023.png",
        p.plotting.tracer_background.plot_background_station_co2_violin,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "background_station_hourly_contribution_2023.png",
        p.plotting.tracer_background.plot_background_station_hourly_contribution,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "co2_diff_vs_wind_speed_2023.png",
        p.plotting.tracer_background.plot_co2_diff_vs_wind_speed,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "co2_diff_vs_wind_direction_2023.png",
        p.plotting.tracer_background.plot_co2_diff_vs_wind_direction,
    )

    # Fluxes
    DIR = "fluxes"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        FIGURE_PATH / DIR / "flux_by_type.png",
        p.plotting.fluxes.plot_flux_by_type,
    )
