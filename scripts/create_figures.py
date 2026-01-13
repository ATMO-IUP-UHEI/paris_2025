import logging
from pathlib import Path

import matplotlib.pyplot as plt

import paris_2025 as p
from paris_2025.plotting import (
    matching_methods,
    meteo_from_catalog,
    meteo_measurements,
    tracer_background,
    tracer_from_catalog,
    tracer_measurements,
)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

FIGURE_PATH = Path(p.CONFIG["figures_path"])
FORCE = False  # Overwrite existing figures

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


def create_figures_if_missing(fig_paths, plot_function, force=FORCE, *args, **kwargs):
    """Create figures only if any of them don't exist."""
    if isinstance(fig_paths, (list, tuple)):
        if any(not path.exists() for path in fig_paths) or force:
            logging.info(f"Creating figures:\n{[str(p) for p in fig_paths]}")
            plot_function(*fig_paths, *args, **kwargs)
        else:
            logging.info(
                "Figures already exist:\n"
                + ("\n".join([str(p) for p in fig_paths if p.exists()]))
            )
    else:
        if not fig_paths.exists() or force:
            logging.info(f"Creating figure:\n{fig_paths}")
            plot_function(fig_paths, *args, **kwargs)
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
        tracer_from_catalog.plot_source_group_contribution_to_stations,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "concentration_at_station_per_simulation.png",
        tracer_from_catalog.plot_concentration_at_station_per_simulation,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "hourly_vprm_concentration.png",
        tracer_from_catalog.plot_hourly_vprm_concentration,
    )

    # Wind measurements
    DIR = "meteo_measurements"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        FIGURE_PATH / DIR / "meteo_stations_map.png",
        meteo_measurements.plot_meteo_measurements_heatmap,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "wind_roses_measurements.png",
        meteo_measurements.plot_wind_roses_of_meteo_measurements,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "meteo_overview.png",
        meteo_measurements.plot_meteo_overview,
    )
    create_figures_if_missing(
        [
            FIGURE_PATH / DIR / "wind_data_availability_2023.png",
            FIGURE_PATH / DIR / "wind_data_availability_2024.png",
        ],
        meteo_measurements.plot_wind_data_availability,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "wind_speed_by_altitude_lidar_2023.png",
        meteo_measurements.plot_wind_speed_by_altitude_lidar,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "wind_speed_by_altitude_non_lidar_2023.png",
        meteo_measurements.plot_wind_speed_by_altitude_non_lidar,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "wind_components_scatter_2023.png",
        meteo_measurements.plot_wind_components,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "hodographs_PACHEM.png",
        meteo_measurements.plot_hodographs,
        station_identifier="PACHEM",
    )

    # Meteo model comparison
    DIR = "meteo_model_comparison"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        FIGURE_PATH / DIR / "wind_components_by_stability_class_2023.png",
        meteo_from_catalog.plot_gral_wind_components_by_stability_class,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "meteo_gramm_gral_comparison.png",
        meteo_from_catalog.plot_meteo_model_comparison,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "matching_methods_comparison.pdf",
        meteo_from_catalog.plot_comparison_of_different_matching_methods,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "model_wind_speed_vs_synoptic.png",
        meteo_from_catalog.plot_model_wind_speed_vs_synoptic,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "hodographs_PACHEM.png",
        meteo_from_catalog.plot_hodographs,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "stability_class_wind_speed_by_season.png",
        meteo_from_catalog.plot_stability_class_and_wind_speed_by_season,
    )

    # Matching methods
    DIR = "matching_methods"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        FIGURE_PATH / DIR / "matching_loss_colormesh.png",
        matching_methods.plot_colormesh_of_loss,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "matching_loss_cumulative_distribution.png",
        matching_methods.plot_matching_loss_distribution,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "n_stations_per_time.png",
        matching_methods.plot_n_stations_per_time,
    )

    # CO2 measurements
    DIR = "co2_measurements"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        FIGURE_PATH / DIR / "co2_concentration_heatmap.png",
        tracer_measurements.plot_co2_concentration_heatmap,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "co2_by_height_and_instrument.png",
        tracer_measurements.plot_co2_by_height_and_instrument,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "average_co2_spatial.png",
        tracer_measurements.plot_average_co2_spatial,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "co2_instruments_map.png",
        tracer_measurements.plot_co2_instruments_map,
    )
    create_figures_if_missing(
        [
            FIGURE_PATH / DIR / "co2_data_availability_2023.png",
            FIGURE_PATH / DIR / "co2_data_availability_2024.png",
        ],
        tracer_measurements.plot_co2_data_availability,
    )
    for year in ["2023", "2024"]:
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"picarro_co2_violin_{year}.png",
            tracer_measurements.plot_picarro_co2_violin,
            year=year,
        )

    # CO2 background analysis
    DIR = "co2_background"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    for year in ["2023", "2024"]:
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"mean_windrose_{year}.png",
            tracer_background.plot_mean_windrose,
            year=year,
        )
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"co2_stations_with_windrose_{year}.png",
            tracer_background.plot_co2_stations_with_windrose,
            year=year,
        )
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"background_co2_stations_{year}.png",
            tracer_background.plot_background_co2_stations,
            year=year,
        )
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"background_station_counts_{year}.png",
            tracer_background.plot_background_station_counts,
            year=year,
        )
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"background_station_co2_violin_{year}.png",
            tracer_background.plot_background_station_co2_violin,
            year=year,
        )
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"background_station_hourly_contribution_{year}.png",
            tracer_background.plot_background_station_hourly_contribution,
            year=year,
        )
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"co2_diff_vs_wind_speed_{year}.png",
            tracer_background.plot_co2_diff_vs_wind_speed,
            year=year,
        )
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"co2_diff_vs_wind_direction_{year}.png",
            tracer_background.plot_co2_diff_vs_wind_direction,
            year=year,
        )

    # Fluxes
    DIR = "fluxes"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        FIGURE_PATH / DIR / "flux_by_type.png",
        p.plotting.fluxes.plot_flux_by_type,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "temporal_scaling_factors.png",
        p.plotting.fluxes.plot_temporal_scaling_factors,
    )
