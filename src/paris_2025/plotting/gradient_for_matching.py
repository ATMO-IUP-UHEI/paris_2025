from pathlib import Path

import ggpymanager as ggp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import paris_2025 as p
from paris_2025.config import CONFIG
from paris_2025.plotting.common import get_metadata


def plot_vertical_profile_by_stability(
    data_array,
    status,
    title="Vertical Profile by Stability Class",
    xlabel="Value",
    figsize=(8, 6),
    ax=None,
):
    """
    Plot vertical profiles grouped by stability class with viridis colormap.

    Parameters:
    -----------
    data_array : xarray.DataArray
        Data to plot with dimensions including 'sim_id' and coordinate to use as
        vertical axis
    title : str
        Plot title
    xlabel : str
        Label for x-axis
    figsize : tuple
        Figure size
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.
    """
    if ax is None:
        print("Creating new figure and axes")
        fig, ax = plt.subplots(figsize=figsize)

    # Get unique stability classes and create viridis colormap
    stability_classes, color_map = create_stability_class_colors()

    # Group by stability class and plot
    for stability_class in stability_classes:
        mask = status.init_stability_class == stability_class
        data = data_array.where(mask, drop=True)

        # Plot all simulations for this stability class with the same color
        for sim_idx in range(data.shape[0]):
            if "vertical_level" in data.coords:
                vertical_level = data.vertical_level
            elif "z" in data.coords:
                vertical_level = data.z
            else:
                raise ValueError("No vertical coordinate found in data_array.")
            ax.plot(
                data.isel(sim_id=sim_idx),
                vertical_level,
                color=color_map[int(stability_class)],
                lw=0.5,
                alpha=0.7,
                label=f"Class {int(stability_class)}" if sim_idx == 0 else "",
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Vertical Level [m]")
    ax.set_title(title)


def create_stability_class_colors() -> tuple[np.ndarray, dict[int, tuple]]:
    stability_classes = np.arange(1, 8)
    cmap = plt.get_cmap("viridis", len(stability_classes))

    # Create color mapping for stability classes
    color_map = {int(sc): cmap(i) for i, sc in enumerate(stability_classes)}
    return stability_classes, color_map


def compute_normalized_vertical_gradient(
    data_array: xr.DataArray, vertical_coord_name: str
) -> xr.DataArray:
    # vertical_coord_dim = str(data_array[vertical_coord_name].dims[0])
    vertical_profile = data_array.load().groupby(vertical_coord_name).mean()
    normalized = vertical_profile / vertical_profile.mean(vertical_coord_name)
    gradient = normalized.diff(vertical_coord_name).sum(vertical_coord_name)
    # vertical_coord_dim = str(data_array[vertical_coord_name].dims[0])
    # vertical_profile = data_array.sortby(vertical_coord_name)
    # normalized = vertical_profile / vertical_profile.mean(vertical_coord_dim)
    # gradient = normalized.diff(vertical_coord_dim).sum(vertical_coord_dim)
    return gradient


def plot_gradient_overview(status, gramm_meteo, gral_concentration, method):
    # Get unique stability classes and create viridis colormap
    stability_classes, color_map = create_stability_class_colors()

    grouped = status.groupby("init_wind_speed")

    n_rows = len(grouped)

    grid = True
    fig, axs = plt.subplots(n_rows, 6, figsize=(30, 5 * n_rows))
    for ax_row in axs:
        for ax in ax_row:
            ax.grid(grid)

    for i, (wind_speed, group) in enumerate(grouped):
        if method == "stations":
            vertical_profile = (
                gral_concentration.concentration.groupby("z")
                .mean()
                .sum("source_group")
                .sel(sim_id=group.sim_id)
                .rename({"z": "vertical_level"})
            )
            wind_vertical_profile = (
                gramm_meteo.wind_speed.groupby("altitude")
                # gral_meteo.wind_speed.groupby("altitude")
                .mean()
                .sel(sim_id=group.sim_id)
                .rename({"altitude": "vertical_level"})
            )
        elif method == "all":
            vertical_profile = group.concentration_vertical_profile
            wind_vertical_profile = group.wind_speed_vertical_gradient.rename(
                {"z": "vertical_level"}
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Plot wind speed vertical profile
        plot_vertical_profile_by_stability(
            wind_vertical_profile,
            status,
            title="Wind Speed" if i == 0 else "",
            xlabel="Wind Speed Gradient",
            ax=axs[i, 0],
        )  # type: ignore
        normalized_wind = wind_vertical_profile / wind_vertical_profile.mean(
            "vertical_level"
        )
        plot_vertical_profile_by_stability(
            normalized_wind,
            status,
            title="Normalized Wind Speed Gradient" if i == 0 else "",
            xlabel="Normalized Wind Speed Gradient",
            ax=axs[i, 1],
        )  # type: ignore

        # Plot concentration vertical profile
        plot_vertical_profile_by_stability(
            vertical_profile,
            status,
            title="Concentration Profile" if i == 0 else "",
            xlabel="Concentration [ppm]",
            ax=axs[i, 3],
        )  # type: ignore

        # Plot normalized concentration vertical profile
        normalized = vertical_profile / vertical_profile.mean(
            "vertical_level"
        )  # .isel(vertical_level=0)

        plot_vertical_profile_by_stability(
            normalized,
            status,
            title="Normalized Concentration Profile" if i == 0 else "",
            xlabel="Normalized Concentration",
            ax=axs[i, 4],
        )  # type: ignore

        # Plot scatter of vertical gradient of wind speed vs lowest level
        # concentration
        # subset = vertical_profile  # .sel(vertical_level=slice(30, 100))
        subset = normalized_wind  # .sel(vertical_level=slice(30, 100))
        x_data = subset.diff("vertical_level").sum("vertical_level")
        y_data = vertical_profile.isel(vertical_level=0)
        c_data = [
            color_map[sc]
            for sc in status.init_stability_class.sel(sim_id=subset.sim_id).values
        ]
        axs[i, 2].scatter(x_data, y_data, c=c_data)
        axs[i, 2].set_xlabel("Mean vertical Gradient of Wind Speed")
        axs[i, 2].set_ylabel("Concentration at Lowest Level [ppm]")
        axs[i, 2].set_title(
            "Wind Profile vs. Lowest Level Concentration" if i == 0 else "",
        )

        # Plot scatter of vertical gradient vs lowest level concentration
        # subset = vertical_profile  # .sel(vertical_level=slice(30, 100))
        subset = normalized  # .sel(vertical_level=slice(30, 100))
        x_data = subset.diff("vertical_level").sum("vertical_level")
        y_data = vertical_profile.isel(vertical_level=0)
        c_data = [
            color_map[sc]
            for sc in status.init_stability_class.sel(sim_id=subset.sim_id).values
        ]
        axs[i, 5].scatter(x_data, y_data, c=c_data)
        axs[i, 5].set_xlabel("Mean vertical Gradient of Concentration")
        axs[i, 5].set_ylabel("Concentration at Lowest Level [ppm]")
        axs[i, 5].set_title(
            "Profile vs. Lowest Level Concentration" if i == 0 else "",
        )

        # Add wind speed textbox
        axs[i, 5].text(
            0.95,
            0.95,
            f"Wind speed: {wind_speed:.1f} m/s",
            transform=axs[i, 5].transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    # Create custom legend with unique entries
    handles = [
        mpl.lines.Line2D([0], [0], color=color_map[sc], lw=4)  # type: ignore
        for sc in stability_classes
    ]
    labels = [f"Class {sc}" for sc in stability_classes]
    fig.legend(handles, labels, title="Stability Class", loc="center right")

    for col in range(6):
        xlims = [axs[row, col].get_xlim() for row in range(n_rows)]
        xmin = min(x[0] for x in xlims)
        xmax = max(x[1] for x in xlims)
        for row in range(n_rows):
            axs[row, col].set_xlim(xmin, xmax)
        ylims = [axs[row, col].get_ylim() for row in range(n_rows)]
        ymin = min(y[0] for y in ylims)
        ymax = max(y[1] for y in ylims)
        for row in range(n_rows):
            axs[row, col].set_ylim(ymin, ymax)

    # Remove vertical spacing
    plt.subplots_adjust(hspace=0)
    return fig


def create_figure(fig_path: str | Path, method: str):
    catalog = ggp.Catalog(CONFIG["domain"]["gral"]["conf_path"], model="gral")
    status = xr.load_dataset(catalog.status_log_path)
    status["concentration_vertical_profile"] = ggp.utils.ugm3_to_ppm(
        status.concentration_vertical_profile, "co2"
    )
    gral_concentration = p.model.get_co2_data()
    gral_concentration["concentration"] = ggp.utils.ugm3_to_ppm(
        gral_concentration.concentration, "co2"
    )
    gramm_meteo = p.model.get_gramm_meteo_data()
    gramm_meteo = gramm_meteo.sel(station=gramm_meteo.operator.str.find("lidar") < 0)
    # meteo = p.model.get_gral_meteo_data()
    # gral_meteo = p.model.get_gral_meteo_data()
    fig = plot_gradient_overview(status, gramm_meteo, gral_concentration, method)
    plt.savefig(
        fig_path,
        metadata=get_metadata(f"Gradient Overview Plot - Method: {method}"),
        bbox_inches="tight",
    )
    plt.close(fig)
