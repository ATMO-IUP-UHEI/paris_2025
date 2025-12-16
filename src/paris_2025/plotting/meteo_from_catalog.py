from pathlib import Path

import ggpymanager as ggp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import paris_2025 as p
from paris_2025.plotting.common import get_metadata


def calculate_wind_rmse(sm, sm_model, dim):
    """Calculate root mean square error for wind components."""
    return np.sqrt(
        np.mean(
            (
                (sm_model["ux"] - sm["u_wind"]) ** 2
                + (sm_model["vy"] - sm["v_wind"]) ** 2
            ).min(dim)
        )
    )


def calculate_wind_mae(sm, sm_model, dim):
    """Calculate mean absolute error for wind components."""
    return np.mean(
        np.abs(
            (sm_model["ux"] - sm["u_wind"]) ** 2 + (sm_model["vy"] - sm["v_wind"]) ** 2
        ).min(dim)
    )


def plot_gral_wind_components_by_stability_class(
    fig_path: str | Path, year: str = "2023"
):
    """Plot wind components (ux vs vy) by stability class for non-Lidar stations."""
    gral_meteo = p.model.get_gral_meteo_data()

    # Filter out Lidar stations
    non_lidar_meteo = gral_meteo.where(gral_meteo.operator != "lidar", drop=True)

    # Create facet plot
    n_rows = int(np.ceil(len(non_lidar_meteo.station) / 3))
    g = non_lidar_meteo.plot.scatter(
        x="ux",
        y="vy",
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
    ) = "/Users/rmaiwald/Levante/Paris/Input/Buildings/buildings.nc",
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

    config = p.config.load_config()
    meteo = p.meteo.get_meteo_measurements()
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
            sm_gramm.plot.scatter(x="ux", y="vy", ax=axs[i, 1], alpha=0.5)
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
            sm_gral.plot.scatter(x="ux", y="vy", ax=axs[i, 2], alpha=0.5)
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
            height_diff.plot(cmap="bwr", vmin=vmin, vmax=vmax, ax=axs[i, 3])  # type: ignore

        # Set titles
        axs[i, 0].set_title(f"Meteo - Station: {s.item()}")
        axs[i, 1].set_title(f"Meteo - Operator: {sm['operator'].item()}")
        matching_model = config["matching"]["stations"][s.item()]
        axs[i, 2].set_title(f"Matching - Model: {matching_model}")
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
    ) = "/Users/rmaiwald/Levante/Paris/Output/matching_loss.nc",
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
    wind_speed = ggp.processing.wind_speed_from_vector(matched["ux"], matched["vy"])
    wind_direction = ggp.processing.direction_from_vector(matched["ux"], matched["vy"])

    fig, axs = plt.subplots(2, 1, figsize=(64, 6), dpi=300)
    wind_speed.plot(hue="loss_type", ax=axs[0], lw=0.5)
    wind_direction.plot(hue="loss_type", ax=axs[1], lw=0.5)

    axs[0].set_title(f"Wind Speed at Station {station}")
    axs[0].set_ylabel("Wind Speed (m/s)")
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
                    gramm_meteo["ux"], gramm_meteo["vy"]
                ).mean("station"),
                gral_meteo["synoptic_wind_speed"],
                c=gral_meteo["stab_class"],
            )
        elif model == "gral":
            ax.scatter(
                ggp.processing.wind_speed_from_vector(
                    gral_meteo["ux"], gral_meteo["vy"]
                ).mean("station"),
                gral_meteo["synoptic_wind_speed"],
                c=gral_meteo["stab_class"],
            )

        # Create custom legend
        ax.legend(
            handles=[
                mpl.patches.Patch(color=plt.cm.viridis(i / 6), label=f"Class {i}")
                for i in range(7)
            ]
        )
        # Plot 1:1 line
        max_speed = max(
            ggp.processing.wind_speed_from_vector(gral_meteo["ux"], gral_meteo["vy"])
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
