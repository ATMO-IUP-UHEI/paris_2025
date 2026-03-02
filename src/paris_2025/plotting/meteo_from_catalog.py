from pathlib import Path

import ggpymanager as ggp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches

import paris_2025 as p
from paris_2025.config import CONFIG
from paris_2025.plotting.common import get_metadata


def calculate_wind_rmse(sm, sm_model, dim):
    """Calculate root mean square error for wind components."""
    return np.sqrt(
        np.mean(
            (
                (sm_model["u"] - sm["u_wind"]) ** 2
                + (sm_model["v"] - sm["v_wind"]) ** 2
            ).min(dim)
        )
    )


def calculate_wind_mae(sm, sm_model, dim):
    """Calculate mean absolute error for wind components."""
    return np.mean(
        np.abs(
            (sm_model["u"] - sm["u_wind"]) ** 2 + (sm_model["v"] - sm["v_wind"]) ** 2
        ).min(dim)
    )


def plot_gral_wind_components_by_stability_class(
    fig_path: str | Path, year: str = "2023"
):
    """Plot wind components (u vs v) by stability class for non-Lidar stations."""
    gral_meteo = p.model.get_gral_meteo_data()

    # Filter out Lidar stations
    non_lidar_meteo = gral_meteo.where(gral_meteo.operator.load() != "lidar", drop=True)

    # Create facet plot
    n_rows = int(np.ceil(len(non_lidar_meteo.station) / 3))
    g = non_lidar_meteo.plot.scatter(
        x="u",
        y="v",
        hue="stab_class",
        col="station",
        col_wrap=3,
        figsize=(10, 4 * n_rows),
    )
    g.fig.suptitle(f"Wind Components by Stability Class (Non-Lidar) for {year}", y=1.02)

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Wind components by stability class for non-Lidar stations in {year}."
        ),
        bbox_inches="tight",
    )
    plt.close(g.fig)


def plot_meteo_model_comparison(
    fig_path: str | Path,
    buildings_path: (
        str | Path
    ) = Path(CONFIG["buildings_path"]),
    selected_operators: list[str] | None = None,
    max_wind_speed: float = 20.0,
):
    """
    Plot comparison between meteorological measurements and GRAMM/GRAL model outputs.

    Creates a multi-panel figure showing:
    - Measured wind components (u, v)
    - GRAMM model wind components with RMSE/MAE statistics
    - GRAL model wind components with RMSE/MAE statistics
    - Building height differences at GRAL station locations

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure.
    buildings_path : str | Path, optional
        Path to buildings NetCDF file. Default is Paris buildings dataset.
    selected_operators : list[str] | None, optional
        List of operator names to include. Default is ["MeteoFrance", "NCAR",
        "high-cost"].
    max_wind_speed : float, optional
        Maximum wind speed for plot axis limits. Default is 20.0 m/s.
    """
    if selected_operators is None:
        selected_operators = ["MeteoFrance", "NCAR", "high-cost"]

    meteo = p.meteo.get_meteo_measurements()
    # Select time period
    meteo = meteo.sel(
        time=slice(CONFIG["matching"]["time_start"], CONFIG["matching"]["time_end"])
    )
    meteo_gramm = p.model.get_gramm_meteo_data()
    meteo_gral = p.model.get_gral_meteo_data()
    buildings = xr.open_dataset(buildings_path)

    # Filter stations
    non_lidar_mask = meteo["operator"] != "lidar"
    not_all_nan = ~meteo["u_wind"].isnull().all("time")
    operator_mask = meteo["operator"].isin(selected_operators)

    selected_stations = meteo.where(
        non_lidar_mask & not_all_nan & meteo.in_gramm_domain & operator_mask, drop=True
    )["station"].values

    n_rows = len(selected_stations)
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    # Ensure axs is 2D even with single row
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    for i, s in enumerate(selected_stations):
        # Meteo measurements
        sm = meteo.sel(station=s)
        sm.plot.scatter(x="u_wind", y="v_wind", ax=axs[i, 0], alpha=0.5)

        # GRAMM model
        if s in meteo_gramm["station"]:
            sm_gramm = meteo_gramm.sel(station=s)
            sm_gramm.plot.scatter(x="u", y="v", ax=axs[i, 1], alpha=0.5)
            rmse = {
                dim: calculate_wind_rmse(sm, sm_gramm, dim).values
                for dim in ["time", "sim_id"]
            }
            mae = {
                dim: calculate_wind_mae(sm, sm_gramm, dim).values
                for dim in ["time", "sim_id"]
            }
            axs[i, 1].text(
                0.05,
                0.95,
                f"RMSE: {rmse['time']:.2f} ({rmse['sim_id']:.2f}) m/s\n"
                f"MAE: {mae['time']:.2f} ({mae['sim_id']:.2f}) m/s",
                transform=axs[i, 1].transAxes,
                verticalalignment="top",
            )

        # GRAL model
        if s in meteo_gral["station"]:
            sm_gral = meteo_gral.sel(station=s)
            sm_gral.plot.scatter(x="u", y="v", ax=axs[i, 2], alpha=0.5)
            rmse = {
                dim: calculate_wind_rmse(sm, sm_gral, dim).values
                for dim in ["time", "sim_id"]
            }
            mae = {
                dim: calculate_wind_mae(sm, sm_gral, dim).values
                for dim in ["time", "sim_id"]
            }
            axs[i, 2].text(
                0.05,
                0.95,
                f"RMSE: {rmse['time']:.2f} ({rmse['sim_id']:.2f}) m/s\n"
                f"MAE: {mae['time']:.2f} ({mae['sim_id']:.2f}) m/s",
                transform=axs[i, 2].transAxes,
                verticalalignment="top",
            )

            # Plot building height difference
            window = 100  # m
            x = sm_gral["x"]
            y = sm_gral["y"]
            height_diff = (
                buildings["building_height"].sel(
                    x=slice(x - window, x + window), y=slice(y - window, y + window)
                )
                - sm_gral["height"]
            )
            vmax = abs(height_diff).max()
            vmin = -vmax
            height_diff.plot(
                cmap="bwr", vmin=vmin, vmax=vmax, ax=axs[i, 3]
            )  # type: ignore

        # Set titles
        axs[i, 0].set_title(f"Meteo - Station: {s.item()}")
        axs[i, 1].set_title(f"Meteo - Operator: {sm['operator'].item()}")
        # matching_model = config["matching"]["stations"][s.item()]
        # axs[i, 2].set_title(f"Matching - Model: {matching_model}")
        axs[i, 3].set_title("")

    # Format axes
    for ax in axs[:, : n_cols - 1].flatten():
        ax.set_xlim(-max_wind_speed, max_wind_speed)
        ax.set_ylim(-max_wind_speed, max_wind_speed)
        ax.set_aspect("equal", "box")

    for ax in axs[:, 1:].flatten():
        ax.set_ylabel("")
        ax.set_yticklabels([])

    for ax in axs[:, n_cols - 1].flatten():
        ax.set_xlabel("")
        ax.set_xticklabels([])

    for ax in axs[:-1, :].flatten():
        ax.set_xlabel("")
        ax.set_xticklabels([])

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Comparison of meteorological measurements with GRAMM and GRAL model "
            f"outputs for {len(selected_stations)} selected stations."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_comparison_of_different_matching_methods(
    fig_path: str | Path,
    matching_loss_filepath: (
        str | Path
    ) = Path(CONFIG["output_path"]) / ggp.config.MATCHING_LOSS_FILE_NAME,
    station: str = "LONGCHAMP",
):
    """
    Plot comparison of wind speed and direction using different matching loss methods.

    Creates a two-panel figure showing:
    - Wind speed time series for different matching loss methods
    - Wind direction time series for different matching loss methods

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure.
    matching_loss_filepath : str | Path
        Path to the matching loss NetCDF file containing loss metrics for different
        methods.
    station : str, optional
        Name of the station to analyze. Default is "LONGCHAMP".
    """
    gramm_meteo = p.model.get_gramm_meteo_data()
    matching_loss = xr.open_dataset(matching_loss_filepath)
    matched = gramm_meteo.sel(
        station=station, sim_id=matching_loss.idxmin("sim_id")["matching_loss"]
    )
    wind_speed = ggp.processing.wind_speed_from_vector(matched["u"], matched["v"])
    wind_direction = ggp.processing.direction_from_vector(matched["u"], matched["v"])

    fig, axs = plt.subplots(2, 1, figsize=(64, 6), dpi=300)
    wind_speed.plot(hue="loss_type", ax=axs[0], lw=0.5)
    wind_direction.plot(hue="loss_type", ax=axs[1], lw=0.5)

    axs[0].set_title(f"Wind Speed at Station {station}")
    axs[0].set_ylabel("Wind Speed (m/s)")
    axs[1].set_title(f"Wind Direction at Station {station}")
    axs[1].set_ylabel("Wind Direction (degrees)")
    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Comparison of wind speed and direction from GRAMM model "
            "using different matching loss methods."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_model_wind_speed_vs_synoptic(
    fig_path: str | Path,
):
    """
    Plot comparison of model wind speeds versus synoptic wind speed.

    Creates a two-panel figure showing:
    - GRAMM wind speed (from measurements) vs synoptic wind speed
    - GRAL wind speed (at measurement stations) vs synoptic wind speed
    Both colored by atmospheric stability class.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure.
    """
    gramm_meteo = p.model.get_gramm_meteo_data()
    gral_meteo = p.model.get_gral_meteo_data()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    for model, ax in zip(["gramm", "gral"], axs):
        if model == "gramm":
            ax.scatter(
                ggp.processing.wind_speed_from_vector(
                    gramm_meteo["u"], gramm_meteo["v"]
                ).mean("station"),
                gral_meteo["synoptic_wind_speed"],
                c=gral_meteo["stab_class"],
            )
        elif model == "gral":
            ax.scatter(
                ggp.processing.wind_speed_from_vector(
                    gral_meteo["u"], gral_meteo["v"]
                ).mean("station"),
                gral_meteo["synoptic_wind_speed"],
                c=gral_meteo["stab_class"],
            )

        # Create custom legend
        viridis_cmap = plt.get_cmap("viridis")
        ax.legend(
            handles=[
                mpatches.Patch(color=viridis_cmap(i / 6), label=f"Class {i}")
                for i in range(7)
            ]
        )
        # Plot 1:1 line
        max_speed = max(
            ggp.processing.wind_speed_from_vector(gral_meteo["u"], gral_meteo["v"])
            .mean("station")
            .max(),
            gral_meteo["synoptic_wind_speed"].max(),
        )
        ax.plot([0, max_speed], [0, max_speed], color="black", linestyle="--")
        ax.set_xlabel("Average wind speed at measurement sites [m/s]")
        ax.set_ylabel("Synoptic wind speed [m/s]")
        ax.set_title(
            f"Comparison of {model.upper()} wind speed\nand measurement site wind speed"
        )
        ax.grid()

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Comparison of GRAMM and GRAL wind speeds versus synoptic wind speed, "
            "colored by atmospheric stability class."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_hodographs(
    fig_path: str | Path,
    station_identifier: str = "PACHEM",
    n_sim_ids: int = 40,
    sim_id_step: int = 25,
):
    """
    Plot hodographs (wind components by altitude) comparing GRAMM and GRAL models.

    Creates a grid of subplots showing wind components (u vs v) colored by altitude
    for different simulation IDs. GRAMM data is shown in green, GRAL data in red.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure.
    station_identifier : str, optional
        Substring to match station names. Default is "PACHEM".
    n_sim_ids : int, optional
        Number of simulation IDs to plot. Default is 40.
    sim_id_step : int, optional
        Step size between simulation IDs. Default is 25.
    """
    gramm_meteo = p.model.get_gramm_meteo_data()
    gral_meteo = p.model.get_gral_meteo_data()

    # Filter data by station
    gramm_data = gramm_meteo.where(
        gramm_meteo.station.str.contains(station_identifier), drop=True
    )
    gral_data = gral_meteo.where(
        gral_meteo.station.str.contains(station_identifier), drop=True
    )

    # Create simulation ID array
    max_sim_ids = min(len(gramm_data.sim_id), len(gral_data.sim_id))
    sim_ids = np.arange(0, min(n_sim_ids * sim_id_step, max_sim_ids), sim_id_step)

    # Set up subplot grid
    n_cols = 5
    n_rows = len(sim_ids) // n_cols + int(len(sim_ids) % n_cols > 0)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        constrained_layout=True,
    )

    # Get altitude range for consistent colorbar
    vmin = min(gramm_data.altitude.min().values, gral_data.altitude.min().values)
    vmax = max(gramm_data.altitude.max().values, gral_data.altitude.max().values)

    # Plot each simulation ID
    for sim_id, ax in zip(sim_ids, axs.flatten()):
        # Plot GRAMM data (green)
        gramm_data.isel(sim_id=sim_id).plot.scatter(
            x="u",
            y="v",
            hue="altitude",
            cmap="Greens",
            add_colorbar=False,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        # Plot GRAL data (red)
        gral_data.isel(sim_id=sim_id).plot.scatter(
            x="u",
            y="v",
            hue="altitude",
            cmap="Reds",
            add_colorbar=False,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        # Set equal aspect ratio and symmetric axes
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        max_range = max(abs(xlims[0]), abs(xlims[1]), abs(ylims[0]), abs(ylims[1]))
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_title(f"Sim ID: {sim_id}")
        ax.set_xlabel("U wind component (m/s)")
        ax.set_ylabel("V wind component (m/s)")
        ax.set_aspect("equal", adjustable="box")

    # Hide unused subplots
    for ax in axs.flatten()[len(sim_ids) :]:
        ax.axis("off")

    # Add colorbars for GRAMM and GRAL
    cax1 = fig.add_axes((1.02, 0.55, 0.02, 0.35))
    cax2 = fig.add_axes((1.02, 0.1, 0.02, 0.35))

    sm1 = plt.cm.ScalarMappable(
        cmap="Greens", norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm2 = plt.cm.ScalarMappable(
        cmap="Reds", norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
    )

    fig.colorbar(sm1, cax=cax1, label="GRAMM Altitude (m)")
    fig.colorbar(sm2, cax=cax2, label="GRAL Altitude (m)")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Hodographs comparing GRAMM and GRAL models for "
            f"station {station_identifier} over {len(sim_ids)} simulation IDs."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_stability_class_and_wind_speed_by_season(
    fig_path: str | Path,
    gramm_meteo_timeseries_path: str | Path = CONFIG["output_path"]
    + "/"
    + ggp.config.GRAMM_METEO_TIMESERIES_FILE_NAME,
    gral_meteo_timeseries_path: str | Path = CONFIG["output_path"]
    + "/"
    + ggp.config.GRAL_METEO_TIMESERIES_FILE_NAME,
    loss_type: str = "rmse - filter: True",
):
    """
    Plot stability class distribution and wind speed by season and hour of day.

    Creates a 4-panel figure showing atmospheric stability class frequency distributions
    and mean wind speed for each season (spring, summer, autumn, winter). Uses the
    matched model data (GRAMM or GRAL) based on station-specific matching configuration.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure.
    gramm_meteo_timeseries_path : str | Path, optional
        Path to GRAMM meteorological timeseries NetCDF file.
    gral_meteo_timeseries_path : str | Path, optional
        Path to GRAL meteorological timeseries NetCDF file.
    loss_type : str, optional
        Loss type to use for model selection. Default is "rmse - filter: True".
    """
    gramm_meteo_timeseries = ggp.io.preprocess_gramm_meteo(
        xr.open_dataset(gramm_meteo_timeseries_path)
    ).sel(best_sim_id=0)
    gral_meteo_timeseries = ggp.io.preprocess_gral_meteo(
        xr.open_dataset(gral_meteo_timeseries_path)
    ).sel(best_sim_id=0)

    # Select matched model for each station
    model_selection = {
        "gramm": gramm_meteo_timeseries,
        "gral": gral_meteo_timeseries,
    }
    model_meteo_timeseries = xr.concat(
        [
            model_selection[m].sel(station=s)
            for s, m in CONFIG["matching"]["stations"].items()
        ],
        dim="station",
        coords="minimal",
        compat="override",
    )
    # Define seasons
    time_periods = {
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
        "winter": [12, 1, 2],
    }

    fig, axs = plt.subplots(
        1, len(time_periods), figsize=(12, 4), sharex=True, sharey=True, dpi=200
    )

    for i, (season, months) in enumerate(time_periods.items()):
        stab_class_data = {}
        speed_data = {}
        mean = model_meteo_timeseries.sel(loss_type=loss_type).mean("station")
        for label, group in mean.sel(
            time=gral_meteo_timeseries.time.dt.month.isin(months),
        ).groupby("time.hour"):
            stab_class_data[label] = (
                group.stab_class.to_pandas()
                .value_counts()
                .sort_index()
                .reindex(range(1, 8), fill_value=0)
            )
            speed_data[label] = group.wind_speed.mean()

        pd.DataFrame(stab_class_data).T.plot.area(ax=axs[i], legend=False)
        twin_ax = axs[i].twinx()
        line_color = "navy"
        twin_ax.plot(
            list(speed_data.keys()),
            list(speed_data.values()),
            color=line_color,
            linestyle="--",
        )
        axs[i].set_xlabel("Hour of day")
        axs[i].set_ylabel("Frequency")
        axs[i].set_title(f"{chr(i+97)}) {season.capitalize()}")
        axs[i].set_xticks(range(0, 25, 6))
        axs[i].set_xticks(range(0, 25), minor=True)

        # Set the label for the twin axis
        if i == len(time_periods) - 1:
            twin_ax.set_ylabel("Mean wind speed (m/s)", color=line_color)
        twin_ax.tick_params(axis="y", labelcolor=line_color)
        twin_ax.spines["right"].set_color(line_color)
        if i != len(time_periods) - 1:
            twin_ax.set_yticklabels([])
        twin_ax.set_ylim(0, 5)

    # Add legend to the right of the last subplot
    handles, labels = axs[-1].get_legend_handles_labels()
    labels = [chr(int(label) + 96).upper() for label in labels]
    fig.legend(handles, labels, loc="center right", title="Stability Class")
    fig.tight_layout(rect=(0, 0, 0.9, 1))

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Stability class distribution and wind speed by season, "
            f"using {loss_type} matched model data."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_meteo_timeseries_comparison(fig_path: str | Path):
    """
    Plot comparison of meteorological measurements and model outputs over time.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure.
    """
    # gramm_meteo_timeseries = ggp.load("gramm_meteo_timeseries", CONFIG)

    gral_meteo_timeseries = ggp.load("gral_meteo_timeseries", CONFIG)
    meteo = p.meteo.get_meteo_measurements()
    station = "Romainville"

    time_period = slice("2023-07-01", "2023-10-30")

    w = meteo.sel(station=station, time=time_period)
    w_m = gral_meteo_timeseries.sel(
        station=station,
        loss_type="rmse - filter: True",
        best_sim_id=range(5),
        time=time_period,
    ).mean("best_sim_id")
    w_m["direction"] = ggp.processing.direction_from_vector(w_m["u"], w_m["v"])
    w_m["speed"] = ggp.processing.wind_speed_from_vector(w_m["u"], w_m["v"])

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    w.wind_speed.plot(ax=axs[0], label="Observed")
    w_m.speed.plot(ax=axs[0], label="Simulated")

    w.wind_direction.plot(ax=axs[1], label="Observed")
    w_m.direction.plot(ax=axs[1], label="Simulated")

    axs[0].legend()
    axs[0].set_title("")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Wind Speed [m/s]")
    axs[0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[0].grid()
    axs[0].text(
        0.02,
        0.98,
        "(a)",
        transform=axs[0].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )

    axs[1].set_title("")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Wind Direction [degrees]")
    axs[1].grid()
    axs[1].text(
        0.02,
        0.98,
        "(b)",
        transform=axs[1].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )
    plt.subplots_adjust(hspace=0.0)
    rmse = np.sqrt(((w_m.speed - w.wind_speed) ** 2).mean().item())
    mean_observed = w.wind_speed.mean().item()
    bias = (w_m.speed - w.wind_speed).mean().item()
    print(f"Mean observed wind speed: {mean_observed:.2f} m/s")
    print(f"Bias for wind speed: {bias:.2f} m/s")
    print(f"RMSE for wind speed: {rmse:.2f} m/s")
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Comparison of meteorological measurements and model outputs over time "
            f" period {time_period} for station {station}. RMSE: {rmse:.2f} m/s, "
            f"Bias: {bias:.2f} m/s for a mean observed wind speed of "
            f"{mean_observed:.2f} m/s."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
