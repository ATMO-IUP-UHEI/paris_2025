"""Plotting functions for comparing modeled and measured CO2 concentrations."""

import logging
from pathlib import Path
from typing import Literal

import ggpymanager as ggp
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Remove for colormap "flare" # noqa: F401
import xarray as xr
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from paris_2025.config import CONFIG
from paris_2025.plotting import DATA_COLORS, RC_PARAMS
from paris_2025.plotting._loaders import cache_data, load_sector_enhancement_data
from paris_2025.plotting.common import (
    _append_mean_station,
    create_ax_plot,
    get_data,
    get_metadata,
    share_ax_lim,
    station_line_plot,
    station_scatter_plot,
    station_sector_plot,
)

plt.rcParams.update(RC_PARAMS)


def get_plot_data(name, afternoon_only=False, main_wind_direction_only=False):
    background, co2, co2_model = cache_data()
    if name == "background":
        data = [background.sel(station=s) for s in co2_model.station.values]
        title = "Background CO2"
        axis_label = "Background CO2 [ppm]"
    elif name == "measured":
        data = [co2.sel(station=s) for s in co2_model.station.values]
        title = "Measured CO2"
        axis_label = "Measured CO2 [ppm]"
    elif name == "modeled_Origins.earth":
        data = [
            co2_model.sel(prior="Origins.earth").reset_coords(drop=True).sel(station=s)
            for s in co2_model.station.values
        ]
        title = "Modeled CO2 (Origins.earth)"
        axis_label = "Modeled CO2 (Origins.earth) [ppm]"
    elif name == "modeled_TNO":
        data = [
            co2_model.sel(prior="TNO").reset_coords(drop=True).sel(station=s)
            for s in co2_model.station.values
        ]
        title = "Modeled CO2 (TNO)"
        axis_label = "Modeled CO2 (TNO) [ppm]"
    else:
        raise ValueError(f"Unknown plot data name: {name}")
    if afternoon_only:
        data = [
            d.where(
                (d.time.dt.hour >= 12) & (d.time.dt.hour < 16),
                # drop=True,
            )
            for d in data
        ]
    if main_wind_direction_only:
        main_wind_code = "SAC"
        background_station = (
            ggp.load("background_co2", CONFIG)
            .dynamic_background_station.str.contains(main_wind_code)
            .compute()
        )
        time_mask = (background_station).reset_coords(drop=True)
        data = [d.where(time_mask, drop=True) for d in data]
    return data, title, axis_label


def plot_tracer_model_scatter_plots(fig_path: str | Path):
    combinations = [
        ("background", "measured"),
        ("modeled_Origins.earth", "measured"),
        ("modeled_TNO", "measured"),
        ("modeled_Origins.earth", "modeled_TNO"),
    ]

    xlims = (400, 520)
    ylims = (400, 520)
    col_wrap = 5
    co2 = cache_data()[1]

    for afternoon_only in [True, False]:
        afternoon_label = "_afternoon" if afternoon_only else ""
        afternoon_title = " (12:00-16:00)" if afternoon_only else ""
        for main_wind_direction_only in [True, False]:
            wind_label = "_main_wind_direction" if main_wind_direction_only else ""
            wind_title = " (main wind direction SW)" if main_wind_direction_only else ""
            for x_, y_ in combinations:
                x_data, x_title, x_label = get_plot_data(
                    x_,
                    afternoon_only=afternoon_only,
                    main_wind_direction_only=main_wind_direction_only,
                )
                y_data, y_title, y_label = get_plot_data(
                    y_,
                    afternoon_only=afternoon_only,
                    main_wind_direction_only=main_wind_direction_only,
                )
                suptitle = (
                    f"{x_title} vs. {y_title} (rmse - filter: True) "
                    f"{afternoon_title}{wind_title}"
                )
                logging.info(suptitle)

                from_template = str(fig_path).format(
                    x_title=x_title,
                    y_title=y_title,
                    afternoon_label=afternoon_label,
                    wind_label=wind_label,
                )

                station_scatter_plot(
                    fig_path=from_template,
                    data_x=x_data,
                    data_y=y_data,
                    co2=co2,
                    suptitle=suptitle,
                    xlabel=x_label,
                    ylabel=y_label,
                    xlims=xlims,
                    ylims=ylims,
                    col_wrap=col_wrap,
                    norm="linear",
                )


def plot_bias_rmse_by_location(fig_path: str | Path):
    """Plot bias and RMSE statistics by station location and height.

    Parameters
    ----------
    fig_path : str or Path
        Path to save the figure
    """
    background, co2, co2_model = cache_data()
    co2_model = co2_model
    # Filter out stations outside the GRAL domain
    co2 = co2.sel(station=co2.in_gral_domain)
    high_cost_mask = co2.instrument == "Picarro"
    high_cost_stations = list(co2.station.sel(station=high_cost_mask).values)
    mid_cost_stations = list(co2.station.sel(station=~high_cost_mask).values)
    x = co2.sel(station=high_cost_stations + list(mid_cost_stations)).sel(
        time=(12 <= co2.time.dt.hour) & (co2.time.dt.hour <= 16)
    )
    y_tno = (
        co2_model.sel(prior="TNO").sel(
            station=high_cost_stations + list(mid_cost_stations)
        )
        # + background
    ).sel(time=(12 <= co2.time.dt.hour) & (co2.time.dt.hour <= 16))
    y = (
        co2_model.sel(prior="Origins.earth").sel(
            station=high_cost_stations + list(mid_cost_stations)
        )
        #   + background
    ).sel(time=(12 <= co2.time.dt.hour) & (co2.time.dt.hour <= 16))

    position_x = co2.sel(station=high_cost_stations + list(mid_cost_stations)).x.values
    position_y = co2.sel(station=high_cost_stations + list(mid_cost_stations)).y.values

    bias = (y - x).mean(dim="time")
    rmse = np.sqrt(((y - x) ** 2).mean(dim="time"))
    bias_tno = (y_tno - x).mean(dim="time")
    rmse_tno = np.sqrt(((y_tno - x) ** 2).mean(dim="time"))
    bias_background = (background - x).mean(dim="time")
    rmse_background = np.sqrt(((background - x) ** 2).mean(dim="time"))

    t = co2.sel(station=high_cost_stations + list(mid_cost_stations)).type.values

    y_labels = ["Mean Bias", "RMSE"]
    plot_data = [(bias, bias_tno, bias_background), (rmse, rmse_tno, rmse_background)]
    x_data = [co2.height, position_x, position_y]
    x_labels = ["Height above ground level [m]", "x position [m]", "y position [m]"]

    for x_label, xd in zip(x_labels, x_data):
        for y_label, yd in zip(y_labels, plot_data):
            fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
            for station_type in np.unique(co2.type):
                scatter = axs[0].scatter(
                    xd, yd[0].where(t == station_type), label=station_type
                )
                axs[0].hlines(
                    yd[0].where(t == station_type).mean(),
                    xd.min(),
                    xd.max(),
                    color=scatter.get_facecolors(),
                )
                axs[1].scatter(xd, yd[1].where(t == station_type), label=station_type)
                axs[1].hlines(
                    yd[1].where(t == station_type).mean(),
                    xd.min(),
                    xd.max(),
                    color=scatter.get_facecolors(),
                )
            axs[0].set_title("Origins.Earth")
            axs[0].set_xlabel(x_label)
            axs[0].set_ylabel(f"{y_label} Afternoon [ppm]")
            axs[1].set_title("TNO")
            axs[1].set_xlabel(x_label)
            plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

            y_title = y_label.replace(" ", "_")
            x_title = x_label.split("[")[0].strip().replace(" ", "_")
            from_template = str(fig_path).format(
                x_title=x_title,
                y_title=y_title,
            )
            plt.savefig(
                from_template,
                metadata=get_metadata(f"{y_label} by {x_label}"),
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_timeseries_comparison(
    fig_path: str | Path,
    inventory: str,
    n_best: int,
    stations: list,
    start_time: str,
    duration=14,
    rolling=3,
):
    """Plot time series comparison of modeled vs measured CO2."""
    background, co2, co2_model = cache_data()
    gral_co2 = co2_model.sel(prior=inventory)
    time_slice = slice(
        start_time, np.datetime64(start_time) + np.timedelta64(duration, "D")
    )

    fig, axs = plt.subplots(
        len(stations),
        1,
        figsize=(18, 4 * len(stations)),
        sharex=True,
        gridspec_kw={"hspace": 0.2, "wspace": 0.2},
    )

    if len(stations) == 1:
        axs = [axs]

    for i, (ax, station) in enumerate(zip(axs, stations)):
        # Model
        (gral_co2.sel(station=station).rolling(time=rolling).mean()).sel(
            time=time_slice
        ).plot(
            ax=ax,
            label="Model",
        )

        # Model uncertainty
        # TODO: Fix using ensemble method
        # extended_co2 = (
        #     gral_co2.sel(station=station)
        #     .isel(best_sim_id=slice(0, 10))
        #     .rolling(time=rolling)
        #     .mean()
        # ).sel(time=time_slice)
        # ax.fill_between(
        #     extended_co2.time,
        #     extended_co2.min(dim="best_sim_id"),
        #     extended_co2.max(dim="best_sim_id"),
        #     alpha=0.3,
        #     label="Model uncertainty",
        # )

        # Measurement
        co2.sel(station=station).sel(time=time_slice).plot(
            ax=ax,
            label="Measurement",
        )  # type: ignore

        # Background
        background.sel(station=station, time=time_slice).plot(
            ax=ax,
            label="Background",
        )  # type: ignore

        # Set axis labels
        ax.set_ylabel(r"CO$_2$ [ppm]")
        ax.set_xlabel("")
        ax.set_title(None)

        # Create title
        fw = ax.xaxis.label.get_fontweight()
        title = (
            f"{co2.code.sel(station=station).values}"
            f" {co2['height'].sel(station=station).values}m"
        )
        ax.text(
            0.5,
            1.06,
            title,
            transform=ax.transAxes,
            va="center",
            ha="center",
            fontsize=15,
            fontweight=fw,
        )

        if i == 0:
            ax.legend(loc="upper left")

    plt.xlabel("Time")

    plt.savefig(
        fig_path,
        metadata=get_metadata(f"Time series comparison starting {start_time}"),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_cycles_per_station(
    fig_path: str | Path,
    loss_type: str,
    prior: str,
    time_slice: str | slice,
    time_str: str,
    groupby: str,
    season: str,
):
    background, co2, co2_model = cache_data(loss_type=loss_type)
    stations = co2_model.station.values

    season_mask = {
        "": xr.ones_like(co2.time),
        "spring": co2.time.dt.month.isin([3, 4, 5]),
        "summer": co2.time.dt.month.isin([6, 7, 8]),
        "fall": co2.time.dt.month.isin([9, 10, 11]),
        "winter": co2.time.dt.month.isin([12, 1, 2]),
    }

    co2_model = _append_mean_station(co2_model)
    co2 = _append_mean_station(co2)
    background = _append_mean_station(background)
    model_data = [
        co2_model.reset_coords(drop=True)
        .where(season_mask[season])
        .sel(
            prior=prior,
            station=s,
            time=time_slice,
        )
        for s in stations
    ]
    measurement_data = [
        co2.reset_coords(drop=True)
        .where(season_mask[season])
        .sel(station=s, time=time_slice)
        for s in stations
    ]
    background_data = [
        background.where(season_mask[season]).sel(time=time_slice, station=s)
        for s in stations
    ]
    labels = ["Background", "Measurement", "Model"]

    ylabel = "CO$_2$ [ppm]"
    ylims = (412, 465)
    suptitle = f"CO2 vs. {groupby} ({prior}, {loss_type}) {time_str} {season}"

    col_wrap = 8
    station_line_plot(
        model=co2_model.sel(station=stations),
        model_data=model_data,
        measurement_data=measurement_data,
        background_data=background_data,
        labels=labels,
        groupby=groupby,
        suptitle=suptitle,
        ylabel=ylabel,
        ylims=ylims,
        col_wrap=col_wrap,
    )
    plt.savefig(
        fig_path,
        metadata=get_metadata(suptitle),
        bbox_inches="tight",
    )
    plt.close()


def plot_full_timeseries_daily_mean(
    fig_path: str | Path, loss_type: str, prior: str, afternoon_only: bool = True
):

    background, co2, co2_model = cache_data(loss_type=loss_type)
    afternoon_start_hour = 12
    afternoon_end_hour = 16
    afternoon_mask = co2.time.dt.hour.isin(
        range(afternoon_start_hour, afternoon_end_hour)
    )
    if not afternoon_only:
        afternoon_mask = xr.ones_like(co2.time, dtype=bool)

    for station in co2_model.station.values:
        width = RC_PARAMS["figure.figsize"][0]
        fig, ax = plt.subplots(figsize=(width, 3))
        background.sel(station=station, time=afternoon_mask).resample(
            time="1D"
        ).mean().plot(
            add_legend=False,
            color=DATA_COLORS["background"],
            label="Background",
            ax=ax,
        )  # type: ignore
        co2.sel(station=station, time=afternoon_mask).resample(time="1D").mean().plot(
            add_legend=False,
            color=DATA_COLORS["measurement"],
            label="Measurement",
            ax=ax,
        )
        co2_model.sel(prior=prior).sel(station=station, time=afternoon_mask).resample(
            time="1D"
        ).mean().plot(
            add_legend=False,
            color=DATA_COLORS["model"],
            label="Model",
            ax=ax,
        )
        ax.legend()
        if afternoon_only:
            title = (
                f"Daily mean afternoon ({afternoon_start_hour}-{afternoon_end_hour} h) "
                f"CO$_2$ at station {station} ({prior}, {loss_type})"
            )
        else:
            title = f"Daily mean CO$_2$ at station {station} ({prior}, {loss_type})"
        ax.set_title("")
        ax.set_ylabel("CO$_2$ [ppm]")
        from_template = str(fig_path).format(
            station=station,
        )
        fig.savefig(
            from_template,
            metadata=get_metadata(title),
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_sector_cycles_per_station(
    fig_path: str | Path,
    inventory: str,
    groupby: Literal["hour", "day", "week", "month"],
    loss_type: str = "rmse - filter: True",
    ylabel: str = "CO2 [ppm]",
    ylims: tuple = (410, 470),
    col_wrap: int = 10,
) -> None:
    """Plot stacked sector cycles per station and save to *fig_path*.

    Loads sector-resolved enhancement data via
    :func:`load_sector_enhancement_data`, delegates the rendering to
    :func:`station_sector_plot`, and saves the figure.

    Parameters
    ----------
    fig_path : str or Path
        Destination path for the saved figure.
    inventory : str
        Inventory name substring, e.g. ``"TNO"`` or ``"Origins.earth"``.
    groupby : str
        Temporal grouping: ``"hour"``, ``"day"``, ``"week"``, or ``"month"``.
    loss_type : str
        Loss-type filter forwarded to :func:`load_sector_enhancement_data`.
    ylabel : str
        Y-axis label.
    ylims : tuple
        (ymin, ymax) applied to all subplots.
    col_wrap : int
        Number of subplot columns.
    """
    model_enhancement, co2, background = load_sector_enhancement_data(
        loss_type=loss_type
    )
    suptitle = f"CO2 sector contributions — {inventory} — by {groupby}"
    fig, axs = station_sector_plot(
        model_enhancement=model_enhancement,
        co2=co2,
        background=background,
        inventory=inventory,
        groupby=groupby,
        suptitle=suptitle,
        ylabel=ylabel,
        ylims=ylims,
        col_wrap=col_wrap,
    )
    plt.savefig(
        fig_path,
        metadata=get_metadata(suptitle),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_diurnal_cycle_by_weekday(
    fig_path: str | Path,
    inventory: str,
    loss_type: str = "rmse - filter: True",
    ylim: tuple = (420, 460),
) -> None:
    """Plot mean diurnal CO₂ cycle averaged over all high-quality stations,
    with one subplot per day of the week.

    Parameters
    ----------
    fig_path : str or Path
        Path to save the figure.
    inventory : str
        Inventory name substring, e.g. ``"TNO"`` or ``"Origins.earth"``.
    loss_type : str
        Loss-type filter forwarded to :func:`load_sector_enhancement_data`.
    ylim : tuple
        (ymin, ymax) for all subplots.
    """
    days = {
        "Monday": [0],
        "Tuesday": [1],
        "Wednesday": [2],
        "Thursday": [3],
        "Friday": [4],
        "Saturday": [5],
        "Sunday": [6],
    }

    model_enhancement, co2, background = load_sector_enhancement_data(
        loss_type=loss_type
    )

    station_mask = co2.instrument.str.contains("Picarro|HPP") & co2.in_gral_domain

    model = model_enhancement.sel(
        type=model_enhancement.type.str.contains(f"VPRM|{inventory}")
    ).sum("type")
    modeled = model + background

    combined = xr.Dataset(
        {
            "model": modeled.reset_coords(drop=True),
            "measurement": co2.reset_coords(drop=True),
            "background": background.reset_coords(drop=True),
        }
    )
    combined = combined.where(
        combined.model.notnull()
        & combined.measurement.notnull()
        & combined.background.notnull()
    )
    combined = combined.sel(station=station_mask)

    suptitle = f"{inventory} - Diurnal cycle of CO$_2$ enhancement by day of week"
    fig, axs = plt.subplots(1, len(days), figsize=(15, 4), sharey=True, sharex=True)
    fig.suptitle(suptitle)

    for var in combined.data_vars:
        data = combined[var]
        for i, (day_name, day_indices) in enumerate(days.items()):
            diurnal_cycle = (
                data.sel(time=data.time.dt.dayofweek.isin(day_indices))
                .mean("station")
                .groupby("time.hour")
                .mean()
            )
            diurnal_cycle.plot(ax=axs[i])  # type: ignore
            axs[i].set_xlabel("Hour of Day")
            axs[i].set_ylim(ylim)
            axs[i].set_title(day_name)

    axs[-1].legend(
        ["Model + Background", "CO$_2$", "Background"],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.savefig(fig_path, metadata=get_metadata(suptitle), bbox_inches="tight")
    plt.close(fig)


def tracer_by_axes_plot(fig_path: str | Path, station_list, plot_info_list):
    stations = np.array(station_list)
    plot_infos = np.array(plot_info_list)
    assert stations.shape == plot_infos.shape
    n_rows, n_cols = stations.shape

    fig, axs = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            create_ax_plot(
                stations[i, j],
                plot_infos[i, j],
                axs[i, j],  # type: ignore
                fig,
            )

    stations = xr.DataArray(stations, dims=["row", "col"])
    plot_infos = xr.DataArray(plot_infos, dims=["row", "col"])

    share_ax_lim(
        "both",
        axs[plot_infos.str.contains("hist2d")].flatten(),  # type: ignore
        xlim=(400, 500),
        ylim=(400, 500),
    )
    share_ax_lim(
        "both", axs[plot_infos.str.contains("groupby")].flatten()  # type: ignore
    )
    for ax in axs[:-1].flatten():  # pyright: ignore[reportIndexIssue]
        ax.tick_params(labelbottom=False)
    plt.savefig(
        fig_path,
        metadata=get_metadata("Tracer comparison by axes."),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_ensemble_spread_vs_mismatch(
    fig_path: str | Path, absolute: bool = False
) -> None:
    """Plot 2D histogram of ensemble spread vs. model-measurement mismatch.

    X-axis: ensemble spread (max − min across best simulations).
    Y-axis: mismatch (measurement − background − model mean).
    A marginal distribution of the mismatch is shown on the right.
    """
    model_co2 = ggp.load("concentration_timeseries", config=CONFIG)
    series = (
        model_co2.sel(loss_type="rmse - filter: True")
        .co2_timeseries.sel(
            type=model_co2.type.str.contains("Origins.earth|VPRM"),
        )
        .sum("type")
        .compute()
    )
    filtered = series.where(series.loss_diff < 0.1)
    diff = filtered.max("best_sim_id") - filtered.min("best_sim_id")
    co2 = ggp.load("co2_measurements", config=CONFIG).sel(station=model_co2.station)
    background = ggp.load("background_co2", config=CONFIG).binned_background
    diff_co2 = (co2.co2 - background - filtered.mean("best_sim_id")).compute()
    if absolute:
        diff_co2 = abs(diff_co2)

    not_nan = diff.notnull() & diff_co2.notnull()
    x_data = diff.values[not_nan]
    y_data = diff_co2.values[not_nan]

    fig, ax = plt.subplots(figsize=(8, 5))
    h = ax.hist2d(x_data, y_data, bins=50, norm=mcolors.LogNorm(), cmap="rainbow")

    ax.axhline(0, color="k", lw=1, linestyle="--")
    ax.set_xlabel("Ensemble spread: max \u2212 min CO$_2$ (ppm)")
    ax.set_ylabel("Mismatch: measurement \u2212 model (ppm)")
    ax.grid(False)

    divider = make_axes_locatable(ax)
    ax_marg = divider.append_axes("right", size="20%", pad=0.2)
    ax_cbar = divider.append_axes("right", size="5%", pad=0.2)

    fig.colorbar(h[3], cax=ax_cbar, label="Count")

    ax_marg.hist(
        y_data, bins=80, orientation="horizontal", color="steelblue", alpha=0.8
    )
    ax_marg.axhline(0, color="k", lw=1, linestyle="--")
    ax_marg.set_ylim(ax.get_ylim())
    ax_marg.tick_params(labelleft=False)
    ax_marg.set_xlabel("Count")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "2D histogram of ensemble spread (x) vs. model-measurement mismatch (y)."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_tracer_custom_grid(
    fig_path: str | Path,
    station_list: list,
    plot_info_list: list,
    figsize: tuple = (8, 6),
    ylabel_row0: str = "CO$_2$ on weekdays (ppm)",
    ylabel_row1: str = "CO$_2$ on Sundays (ppm)",
) -> None:
    """Plot a custom grid of tracer comparisons with configurable stations and plot
    types.

    Parameters
    ----------
    fig_path : str or Path
        Path to save the figure.
    station_list : list of list of str
        2D list of station names, one per subplot.
    plot_info_list : list of list of str
        2D list of plot info strings corresponding to stations.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 6).
    ylabel_row0 : str, optional
        Y-axis label for first row of subplots.
    ylabel_row1 : str, optional
        Y-axis label for second row of subplots.
    """
    stations = np.array(station_list)
    plot_infos = np.array(plot_info_list)
    assert (
        stations.shape == plot_infos.shape
    ), "station_list and plot_info_list must have same shape"
    n_rows, n_cols = stations.shape

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i in range(n_rows):
        for j in range(n_cols):
            create_ax_plot(
                stations[i, j],
                plot_infos[i, j],
                axs[i, j],  # type: ignore
                fig,
            )

    # stations_da = xr.DataArray(stations, dims=["row", "col"])
    plot_infos_da = xr.DataArray(plot_infos, dims=["row", "col"])

    share_ax_lim(
        "both",
        axs[plot_infos_da.str.contains("hist2d")].flatten(),  # type: ignore
        xlim=(400, 500),
        ylim=(400, 500),
    )
    share_ax_lim(
        "both", axs[plot_infos_da.str.contains("groupby")].flatten()  # type: ignore
    )

    for id, ax in zip("abcdefgh", axs.flatten()):  # type: ignore
        ax.set_title(f" ({id})", loc="left")
        # ax.grid()

    for ax in axs[:-1].flatten():  # pyright: ignore[reportIndexIssue]
        ax.tick_params(tick1On=False, labelbottom=False, which="both", axis="x")
        ax.set_xlabel("")

    for ax in axs[:, 1:].flatten():  # pyright: ignore[reportIndexIssue]
        ax.tick_params(tick1On=False, labelleft=False, which="both", axis="y")
        ax.set_ylabel("")

    if n_rows >= 1:
        axs[0, 0].set_ylabel(ylabel_row0)  # type: ignore
    if n_rows >= 2:
        axs[1, 0].set_ylabel(ylabel_row1)  # type: ignore

    fig.subplots_adjust(wspace=0.05)

    # Create legend
    handles, labels = axs[0, 0].get_legend_handles_labels()  # type: ignore
    fig.legend(handles, labels, loc="center left", ncol=1, bbox_to_anchor=(0.9, 0.5))

    plt.savefig(
        fig_path,
        metadata=get_metadata("Custom grid of tracer comparisons."),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_tracer_custom_grid_with_sector_legends(
    fig_path: str | Path,
    station_list: list,
    plot_info_list: list,
    inventories: list | None = None,
    legend_positions: list | None = None,
    figsize: tuple = (8, 6),
    ylabel_row0: str = "CO$_2$ (ppm)",
    ylabel_row1: str = "CO$_2$ (ppm)",
) -> None:
    """Plot a custom grid with sector-based legends for different inventories.

    Parameters
    ----------
    fig_path : str or Path
        Path to save the figure.
    station_list : list of list of str
        2D list of station names, one per subplot.
    plot_info_list : list of list of str
        2D list of plot info strings corresponding to stations.
    inventories : list, optional
        List of inventory names. Default is ["Origins.earth", "TNO"].
    legend_positions : list, optional
        List of (x, y) bbox anchor positions for each inventory legend.
        Default is [(0.9, 0.7), (0.9, 0.3)].
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 6).
    ylabel_row0 : str, optional
        Y-axis label for first row of subplots.
    ylabel_row1 : str, optional
        Y-axis label for second row of subplots.
    """
    from paris_2025.plotting import INVENTORY_COLORS, INVENTORY_SECTORS

    if inventories is None:
        inventories = ["Origins.earth", "TNO"]
    if legend_positions is None:
        legend_positions = [(0.9, 0.7), (0.9, 0.3)]

    stations = np.array(station_list)
    plot_infos = np.array(plot_info_list)
    assert (
        stations.shape == plot_infos.shape
    ), "station_list and plot_info_list must have same shape"
    n_rows, n_cols = stations.shape

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i in range(n_rows):
        for j in range(n_cols):
            create_ax_plot(
                stations[i, j],
                plot_infos[i, j],
                axs[i, j],  # type: ignore
                fig,
            )

    # stations_da = xr.DataArray(stations, dims=["row", "col"])
    plot_infos_da = xr.DataArray(plot_infos, dims=["row", "col"])

    share_ax_lim(
        "both",
        axs[plot_infos_da.str.contains("hist2d")].flatten(),  # type: ignore
        xlim=(400, 500),
        ylim=(400, 500),
    )
    share_ax_lim(
        "both", axs[plot_infos_da.str.contains("groupby")].flatten()  # type: ignore
    )

    for id, ax in zip("abcdefgh", axs.flatten()):  # type: ignore
        ax.set_title(f" ({id})", loc="left")

    for ax in axs[:-1].flatten():  # pyright: ignore[reportIndexIssue]
        ax.tick_params(tick1On=False, labelbottom=False, which="both", axis="x")
        ax.set_xlabel("")

    for ax in axs[:, 1:].flatten():  # pyright: ignore[reportIndexIssue]
        ax.tick_params(tick1On=False, labelleft=False, which="both", axis="y")
        ax.set_ylabel("")

    if n_rows >= 1:
        axs[0, 0].set_ylabel(ylabel_row0)  # type: ignore
    if n_rows >= 2:
        axs[1, 0].set_ylabel(ylabel_row1)  # type: ignore

    fig.subplots_adjust(wspace=0.05)

    # Create sector-based legends for each inventory
    model_enhancement = get_data("Model Enhancement")

    for inventory, bbox in zip(inventories, legend_positions):
        labels = model_enhancement.type.sel(
            type=model_enhancement.type.str.contains(inventory)
        ).values.tolist()
        if inventory == "Origins.earth":
            labels = np.array(labels)[
                [2, 1, 0, 5, 4, 3]
            ].tolist()  # reorder to match plot
        handles = [
            patches.Patch(color=INVENTORY_COLORS[label], alpha=0.3) for label in labels
        ]
        labels = [INVENTORY_SECTORS[label] for label in labels]

        # Add background and measurement to legend
        handles += [
            Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--"),
            Line2D([0], [0], color="k", linewidth=1.5),
        ]
        labels += ["Background", "Measurement"]
        fig.legend(
            handles,
            labels,
            title=inventory,
            loc="center left",
            ncol=1,
            bbox_to_anchor=bbox,
        )

    plt.savefig(
        fig_path,
        metadata=get_metadata("Custom grid of tracer comparisons with sector legends."),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_timeseries_with_quantile_bands(
    fig_path: str | Path,
    start_date: str,
    end_date: str,
    station: str = "JUS_30",
    cmap_name: str = "crest",
    n_quantiles: int = 21,
) -> None:
    """Plot time series with quantile bands and daily gridlines.

    Parameters
    ----------
    fig_path : str or Path
        Path to save the figure.
    start_date : str
        Start date as string (e.g., "2023-01-15").
    end_date : str
        End date as string (e.g., "2023-02-15").
    station : str, optional
        Station name. Default is "JUS_30".
    cmap_name : str, optional
        Colormap name for quantile bands. Default is "crest".
    n_quantiles : int, optional
        Number of quantiles for uncertainty bands. Default is 21.
    """

    # Load data
    model_co2 = ggp.load("concentration_timeseries", config=CONFIG)
    co2 = ggp.load("co2_measurements", CONFIG)
    background = ggp.load("background_co2", config=CONFIG).binned_background

    # Prepare series data
    series = (
        model_co2.sel(loss_type="rmse - filter: True", station=station)
        .co2_timeseries.sel(
            type=model_co2.type.str.contains("Origins.earth|VPRM"),
        )
        .sum("type")
        .compute()
    )
    series = series + background.sel(station=station)

    time_period = slice(start_date, end_date)
    quantiles = np.linspace(0, 1, n_quantiles)

    mean_plot = (
        series.where(series.loss_diff < 0.1).mean("best_sim_id").sel(time=time_period)
    )
    qda = (
        series.where(series.loss_diff < 0.1)
        .dropna("best_sim_id", how="all")
        .quantile(quantiles, dim="best_sim_id")
        .sel(time=time_period)
    )

    cmap = plt.get_cmap(cmap_name)
    time = qda.time.values

    width = RC_PARAMS["figure.figsize"][0]
    fig, ax = plt.subplots(figsize=(width, 4))

    # Plot quantile bands
    for i in range(len(quantiles) - 1):
        q_lo, q_hi = quantiles[i], quantiles[i + 1]
        y_lo = qda.sel(quantile=q_lo).values
        y_hi = qda.sel(quantile=q_hi).values
        band_center = (q_lo + q_hi) / 2
        color_val = 1 - abs(band_center - 0.5) * 2
        ax.fill_between(time, y_lo, y_hi, color=cmap(color_val), alpha=1, linewidth=0)

    # Plot data lines
    bg_sel = background.sel(station=station).sel(time=time_period)
    meas_sel = co2.co2.sel(station=station).sel(time=time_period)

    ax.plot(
        time,
        mean_plot.values,
        color=DATA_COLORS["model"],
        label="Model",
    )
    ax.plot(
        bg_sel.time.values,
        bg_sel.values,
        color=DATA_COLORS["background"],
        label="Background",
    )
    ax.plot(
        meas_sel.time.values,
        meas_sel.values,
        color=DATA_COLORS["measurement"],
        label="Measurement",
    )

    # Reorder legend
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 0]  # Background, Measurement, Model
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    ax.legend(handles=handles, labels=labels, loc="upper left")

    # Add daily gridlines
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.grid(True, which="both")
    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("CO$_2$ [ppm]")

    plt.savefig(
        fig_path,
        metadata=get_metadata(f"Time series from {start_date} to {end_date}"),
        bbox_inches="tight",
    )
    plt.close(fig)
