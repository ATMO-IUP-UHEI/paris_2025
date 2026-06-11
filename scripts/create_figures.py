import logging
from pathlib import Path

import matplotlib.pyplot as plt

from paris_2025.config import CONFIG
from paris_2025.plotting import (
    matching_methods,
    meteo_from_catalog,
    meteo_measurements,
    misc,
    tracer_background,
    tracer_comparison,
    tracer_from_catalog,
    fluxes,
    gradient_for_matching,
    tracer_measurements,
    RC_PARAMS,
)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

FIGURE_PATH = Path(CONFIG["figures_path"])
FORCE = False  # Overwrite existing figures


# Set matplotlib style like size of the figures, text size, etc. by hand
# sns.set_context("paper", font_scale=1)

plt.rcParams.update(RC_PARAMS)


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
    elif isinstance(fig_paths, dict):
        search_str = str(fig_paths["template"])
        # Replace placeholders with wildcards to search for existing files
        bracket_opens = [i for i, c in enumerate(search_str) if c == "{"]
        bracket_closes = [i for i, c in enumerate(search_str) if c == "}"]
        for open_idx, close_idx in zip(
            reversed(bracket_opens), reversed(bracket_closes)
        ):
            search_str = search_str[:open_idx] + "*" + search_str[close_idx + 1 :]
        existing_files = list(Path(search_str).parent.glob(Path(search_str).name))
        if len(existing_files) < fig_paths["n"] or force:
            logging.info(f"Creating figures matching:\n{search_str}")
            plot_function(fig_paths["template"], *args, **kwargs)
        else:
            logging.info(
                f"Figures already exist matching:\n{search_str} "
                f"({len(existing_files)} files found)"
            )
    elif isinstance(fig_paths, Path):
        if not fig_paths.exists() or force:
            logging.info(f"Creating figure:\n{fig_paths}")
            plot_function(fig_paths, *args, **kwargs)
        else:
            logging.info(f"Figure already exists:\n{fig_paths}")
    else:
        raise ValueError("fig_paths must be a Path or a list/tuple of Paths.")


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
    create_figures_if_missing(
        FIGURE_PATH / DIR / "ensemble_spread_night_and_day_Origins_earth.png",
        tracer_from_catalog.plot_ensemble_spread_night_and_day,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "ensemble_spread_cycles.png",
        tracer_from_catalog.plot_ensemble_spread_cycles,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "ensemble_spread_vs_mismatch.png",
        tracer_comparison.plot_ensemble_spread_vs_mismatch,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "ensemble_spread_vs_mismatch_absolute.png",
        tracer_comparison.plot_ensemble_spread_vs_mismatch,
        absolute=True,
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
    create_figures_if_missing(
        FIGURE_PATH / DIR / "example_time_series_comparison.png",
        meteo_from_catalog.plot_meteo_timeseries_comparison,
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

    # Matching methods - meteo condition analysis
    for var in ["synoptic_wind_speed", "synoptic_wind_direction", "stab_class"]:
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"selected_meteo_conditions_{var}.png",
            matching_methods.plot_selected_meteo_conditions_by_variable,
            variable=var,
        )
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"meteo_selection_frequency_{var}.png",
            matching_methods.plot_meteo_selection_frequency_by_variable,
            variable=var,
        )

    # Matching methods - CO2 analysis
    create_figures_if_missing(
        FIGURE_PATH / DIR / "co2_concentration_violin_by_loss_type.png",
        matching_methods.plot_co2_concentration_violin_by_loss_type,
    )

    for var in ["synoptic_wind_speed", "synoptic_wind_direction", "stab_class"]:
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"co2_distribution_by_{var}.png",
            matching_methods.plot_co2_distribution_by_meteo_variable,
            variable=var,
        )

    # Matching methods - loss analysis
    for var in ["synoptic_wind_speed", "stab_class"]:
        create_figures_if_missing(
            FIGURE_PATH / DIR / f"matching_loss_by_{var}.png",
            matching_methods.plot_matching_loss_by_meteo_variable,
            variable=var,
        )

    # Matching methods - gradient for matching
    for method in ["stations", "all"]:
        fig_path = FIGURE_PATH / DIR / f"gradient_for_matching_overview_{method}.png"
        create_figures_if_missing(
            fig_path,
            gradient_for_matching.create_figure,
            method=method,
        )

    # Matching methods - sensitivity analysis
    create_figures_if_missing(
        FIGURE_PATH / DIR / "concentration_vs_meteo_differences.png",
        matching_methods.plot_concentration_vs_meteo_differences,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "loss_vs_max_loss_difference.png",
        matching_methods.plot_loss_vs_max_loss_difference,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "loss_difference_vs_concentration_difference.png",
        matching_methods.plot_loss_difference_vs_concentration_difference,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "loss_vs_max_concentration_difference.png",
        matching_methods.plot_loss_vs_max_concentration_difference,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "concentration_vs_max_concentration_difference.png",
        matching_methods.plot_concentration_vs_max_concentration_difference,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "wind_speed_vs_max_concentration_difference.png",
        matching_methods.plot_wind_speed_vs_max_concentration_difference,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "wind_direction_vs_max_concentration_difference.png",
        matching_methods.plot_wind_direction_vs_max_concentration_difference,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "stability_class_vs_max_concentration_difference.png",
        matching_methods.plot_stability_class_vs_max_concentration_difference,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "loss_vs_co2_spread_distribution.png",
        matching_methods.plot_loss_vs_co2_spread_distribution,
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
        FIGURE_PATH / DIR / "co2_and_meteo_stations_map.png",
        tracer_measurements.plot_co2_and_meteo_stations_map,
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
    for background_type in ["dynamic", "minimum", "binned"]:
        for grouper_type in ["hour", "month", "wind direction"]:
            create_figures_if_missing(
                FIGURE_PATH / DIR / f"background_co2_distribution_"
                f"{background_type}_by_{grouper_type}.png",
                tracer_background.plot_background_station_count,
                background_type=background_type,
                grouper_type=grouper_type,
            )

    # Fluxes
    DIR = "fluxes"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        FIGURE_PATH / DIR / "flux_by_type.png",
        fluxes.plot_flux_by_type,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "temporal_scaling_factors.png",
        fluxes.plot_temporal_scaling_factors,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "temporal_scaling_factor_cycles.png",
        fluxes.plot_temporal_scaling_factor_cycles,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "temporal_scaling_vprm.png",
        fluxes.plot_temporal_scaling_vprm,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "flux_maps.png",
        fluxes.plot_flux_maps,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "total_flux_by_inventory.png",
        fluxes.plot_total_flux_by_inventory,
    )

    # Miscellaneous
    DIR = "misc"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    create_figures_if_missing(
        FIGURE_PATH / DIR / "temperature_anomaly_with_co2.png",
        misc.plot_temperature_anomaly_with_co2,
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "terrain_elevation_maps.png",
        misc.plot_terrain_elevation_maps,
    )

    # Tracer model comparison
    DIR = "tracer_model_comparison"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    # Scatter plots for different variable combinations
    create_figures_if_missing(
        {
            "n": 16,
            "template": FIGURE_PATH
            / DIR
            / (
                "tracer_model_density_plots"
                "{x_title}_{y_title}{afternoon_label}{wind_label}.png"
            ),
        },
        tracer_comparison.plot_tracer_model_scatter_plots,
    )
    # Scatter plots of bias and RMSE by location
    create_figures_if_missing(
        {
            "n": 6,
            "template": FIGURE_PATH
            / DIR
            / ("bias_and_rmse_tracer_model_scatter_plots_{y_title}_{x_title}.png"),
        },
        tracer_comparison.plot_bias_rmse_by_location,
    )
    # Custom grid plots
    create_figures_if_missing(
        FIGURE_PATH / DIR / "diurnal_cycle_Origins_earth_weekday_and_sunday.png",
        tracer_comparison.plot_tracer_custom_grid,
        station_list=[
            ["JUS_30", "ROV_103", "Mean"],
            ["JUS_30", "ROV_103", "Mean"],
        ],
        plot_info_list=[
            [
                "groupby hour Origins.earth filter weekday",
                "groupby hour Origins.earth filter weekday",
                "groupby hour Origins.earth filter weekday",
            ],
            [
                "groupby hour Origins.earth filter Sunday",
                "groupby hour Origins.earth filter Sunday",
                "groupby hour Origins.earth filter Sunday",
            ],
        ],
    )
    # Custom grid plots
    create_figures_if_missing(
        FIGURE_PATH / DIR / "diurnal_cycle_TNO_weekday_and_sunday.png",
        tracer_comparison.plot_tracer_custom_grid,
        station_list=[
            ["JUS_30", "ROV_103", "Mean"],
            ["JUS_30", "ROV_103", "Mean"],
        ],
        plot_info_list=[
            [
                "groupby hour TNO filter weekday",
                "groupby hour TNO filter weekday",
                "groupby hour TNO filter weekday",
            ],
            [
                "groupby hour TNO filter Sunday",
                "groupby hour TNO filter Sunday",
                "groupby hour TNO filter Sunday",
            ],
        ],
    )
    # Custom grid plots with sector legends
    create_figures_if_missing(
        FIGURE_PATH / DIR / "diurnal_cycle_sector_Origins_earth_TNO.png",
        tracer_comparison.plot_tracer_custom_grid_with_sector_legends,
        station_list=[
            ["JUS_30", "ROV_103", "Mean"],
            ["JUS_30", "ROV_103", "Mean"],
        ],
        plot_info_list=[
            [
                "groupby_sector hour Origins.earth",
                "groupby_sector hour Origins.earth",
                "groupby_sector hour Origins.earth",
            ],
            [
                "groupby_sector hour TNO",
                "groupby_sector hour TNO",
                "groupby_sector hour TNO",
            ],
        ],
    )
    # Time series comparison
    for season in ["summer", "fall", "winter"]:
        if season == "summer":
            start_time = "2023-05-16"
        elif season == "fall":
            start_time = "2023-10-01"
        elif season == "winter":
            start_time = "2023-01-01"
        else:
            raise ValueError(f"Unknown season: {season}")
        for inventory in ["Origins.earth", "TNO"]:
            for n_best in [3, 5, 10, 20]:
                create_figures_if_missing(
                    (
                        FIGURE_PATH / DIR / f"time_series_"
                        f"{season}_{inventory}_{n_best}.png"
                    ),
                    tracer_comparison.plot_timeseries_comparison,
                    inventory=inventory,
                    n_best=n_best,
                    stations=["CDS_34", "JUS_30", "ROV_103"],
                    start_time=start_time,
                )

    # Diurnal, weekly, and annual cycles
    for lt in ["rmse - filter: True"]:
        for prior in ["Origins.earth", "TNO"]:
            for groupby_iterator in [
                "hour",
                "hour-spring",
                "hour-summer",
                "hour-fall",
                "hour-winter",
                "day",
                "week",
            ]:
                for time_slice in ["2023", "2024", slice("2023", "2024")]:
                    time_str = (
                        time_slice
                        if isinstance(time_slice, str)
                        else f"{time_slice.start}-{time_slice.stop}"
                    )
                    loss_type_string = (
                        lt.replace(" - ", "_").replace(": ", "_").replace(" ", "_")
                    )

                    # stations = ["CDS_34", "JUS_30", "JUS_40", "MEU_45", "ROV_103"]
                    if len(groupby_iterator.split("-")) == 2:
                        groupby, season = groupby_iterator.split("-")
                        season_str = "_" + season
                    else:
                        groupby = groupby_iterator
                        season = ""
                        season_str = ""

                    fig_path = (
                        FIGURE_PATH / DIR / f"cycle_{groupby}_"
                        f"{prior.replace('.', '_')}_"
                        f"{loss_type_string}_"
                        f"{time_str}{season_str}.png"
                    )
                    create_figures_if_missing(
                        fig_path,
                        tracer_comparison.plot_cycles_per_station,
                        loss_type=lt,
                        prior=prior,
                        time_slice=time_slice,
                        time_str=time_str,
                        groupby=groupby,
                        season=season,
                    )

    # Sector contribution cycles per station
    for inventory in ["TNO", "Origins.earth"]:
        for groupby in ["hour", "day", "week", "month"]:
            create_figures_if_missing(
                FIGURE_PATH
                / DIR
                / f"sector_cycles_{groupby}_{inventory.replace('.', '_')}.png",
                tracer_comparison.plot_sector_cycles_per_station,
                inventory=inventory,
                groupby=groupby,
            )

    # Diurnal cycle by weekday
    for inventory in ["TNO", "Origins.earth"]:
        create_figures_if_missing(
            FIGURE_PATH
            / DIR
            / f"diurnal_cycle_weekday_{inventory.replace('.', '_')}.png",
            tracer_comparison.plot_diurnal_cycle_by_weekday,
            inventory=inventory,
        )

    # Full time series plots
    for prior in ["Origins.earth", "TNO"]:
        for afternoon_only in [True, False]:
            afternoon_str = "_afternoon" if afternoon_only else ""
            create_figures_if_missing(
                {
                    "n": 26,
                    "template": FIGURE_PATH
                    / DIR
                    / "plot_full_timeseries_mean_{station}"
                    f"_{prior.replace(".", "_")}{afternoon_str}.png",
                },
                tracer_comparison.plot_full_timeseries_daily_mean,
                loss_type="rmse - filter: True",
                prior=prior,
                afternoon_only=afternoon_only,
            )

    # Interactive time series with quantile bands
    create_figures_if_missing(
        FIGURE_PATH / DIR / "timeseries_quantile_bands_jus30_winter.png",
        tracer_comparison.plot_timeseries_with_quantile_bands,
        start_date="2024-01-09",
        end_date="2024-01-22",
        station="JUS_30",
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "timeseries_quantile_bands_jus30_winter_2.png",
        tracer_comparison.plot_timeseries_with_quantile_bands,
        start_date="2023-02-01",
        end_date="2023-02-14",
        station="JUS_30",
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "timeseries_quantile_bands_jus30_winter_2_short.png",
        tracer_comparison.plot_timeseries_with_quantile_bands,
        start_date="2023-02-07",
        end_date="2023-02-14",
        station="JUS_30",
    )
    create_figures_if_missing(
        FIGURE_PATH / DIR / "timeseries_quantile_bands_jus30_spring.png",
        tracer_comparison.plot_timeseries_with_quantile_bands,
        start_date="2023-05-16",
        end_date="2023-05-29",
        station="JUS_30",
    )

    #
    station_list = [
        ["JUS_30", "JUS_30", "Mean Picarro|HPP", "Mean Picarro|HPP"],
        ["JUS_30", "JUS_30", "Mean Picarro|HPP", "Mean Picarro|HPP"],
        ["JUS_30", "JUS_30", "Mean Picarro|HPP", "Mean Picarro|HPP"],
    ]
    plot_info_list = [
        [
            "hist2d Origins.earth vs CO2",
            "groupby hour Origins.earth",
            "groupby hour Origins.earth",
            "groupby_sector hour Origins.earth",
        ],
        [
            "hist2d Origins.earth vs CO2 filter Sunday",
            "groupby hour Origins.earth filter Sunday",
            "groupby hour Origins.earth filter Sunday",
            "groupby_sector hour Origins.earth filter Sunday",
        ],
        [
            "hist2d TNO vs CO2",
            "groupby hour TNO",
            "groupby hour TNO",
            "groupby_sector hour TNO",
        ],
    ]
    create_figures_if_missing(
        FIGURE_PATH / DIR / "diurnal_cycle_selection.png",
        tracer_comparison.tracer_by_axes_plot,
        station_list=station_list,
        plot_info_list=plot_info_list,
    )
