"""Plotting functions for comparing modeled and measured CO2 concentrations."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import ggpymanager as ggp
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Remove for colormap "flare" # noqa: F401
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from matplotlib import patches

import paris_2025 as p
from paris_2025.config import CONFIG
from paris_2025.plotting.common import get_metadata


@lru_cache()
def cache_data(loss_type: str | None = "rmse - filter: True"):
    conc_series = xr.open_mfdataset(
        CONFIG["output_path"] + "/" + ggp.config.CONCENTRATION_TIMESERIES_FILE_NAME
    )
    t = conc_series.type
    mask = xr.concat(
        [
            t.str.contains("Origins.earth") | t.str.contains("VPRM"),
            t.str.contains("TNO") | t.str.contains("VPRM"),
        ],
        dim="prior",
    )
    mask["prior"] = ["Origins.earth", "TNO"]
    if loss_type is not None:
        conc_series = conc_series.sel(loss_type=loss_type)
    time_series = conc_series.co2_timeseries.where(mask).sum("type")
    with ProgressBar():
        time_series = time_series.compute()  # type: ignore
    background = ggp.load("background_co2", CONFIG)["binned_background_by_label"].sel(
        height_bins=time_series.height
    )
    co2 = ggp.load("co2_measurements", CONFIG).co2
    co2_model = background.reset_coords(drop=True) + time_series.reset_coords(
        names=["x", "y"], drop=True
    )

    return background, co2, co2_model


def get_plot_data(name, afternoon_only=False, main_wind_direction_only=False):
    dynamic_background, co2, co2_model = cache_data()
    co2_model = co2_model.sel(best_sim_id=0)
    if name == "background":
        data = [
            dynamic_background.co2.reset_coords(drop=True).assign_coords(station=s)
            for s in co2_model.station.values
        ]
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
                drop=True,
            )
            for d in data
        ]
    if main_wind_direction_only:
        main_wind_code = "SAC"
        time_mask = (dynamic_background.code == main_wind_code).reset_coords(drop=True)
        data = [d.where(time_mask, drop=True) for d in data]
    return data, title, axis_label


def station_scatter_plot(
    fig_path: str | Path,
    data_x,
    data_y,
    co2,
    suptitle,
    xlabel,
    ylabel,
    xlims,
    ylims,
    bins=(50, 50),
    cmap="flare",
    col_wrap=4,
    plot_one_to_one=True,
    plot_mean_std=False,
    show_infos=True,
    aspect_equal=True,
    norm="log",
):
    """Create scatter plots comparing measured and modeled CO2 at multiple stations.

    Parameters
    ----------
    fig_path : str or Path
        Path to save the figure
    data_x : list of xr.DataArray
        Measured CO2 data for each station
    data_y : list of xr.DataArray
        Modeled CO2 data for each station
    co2 : xr.Dataset
        Dataset containing station metadata (code, height, etc.)
    suptitle : str
        Super title for the figure
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    xlims : tuple
        X-axis limits
    ylims : tuple
        Y-axis limits
    bins : tuple, optional
        Number of bins for 2D histogram
    cmap : str, optional
        Colormap name
    col_wrap : int, optional
        Number of columns in subplot grid
    plot_one_to_one : bool, optional
        Whether to plot 1:1 line
    plot_mean_std : bool, optional
        Whether to plot mean and std deviation
    show_infos : bool, optional
        Whether to show RMSE, bias, and correlation info
    aspect_equal : bool, optional
        Whether to set equal aspect ratio
    norm : str or mpl.colors.Normalize, optional
        Normalization for color scale, either 'log' or 'linear'
    """
    n_plots = len(data_x)
    n_rows = int(np.ceil(n_plots / col_wrap))
    fig, axs = plt.subplots(
        n_rows,
        col_wrap,
        figsize=(18, 4 * n_rows),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.2, "wspace": 0.2},
    )

    fig.suptitle(suptitle, fontsize=16)

    im = None  # Initialize for type checking
    for i in range(len(data_x)):
        ax = axs.flatten()[i]
        ds = xr.Dataset({"x_plot": data_x[i], "y_plot": data_y[i]})
        ds = ds.dropna(dim="time")
        station = data_x[i].station

        # Check if there is enough data
        N = len(ds.time)
        if N < 100:
            station_code = co2["code"].sel(station=station).values
            print(f"Skipping {station_code} due to insufficient data")
            ax.axis("off")
            continue

        # Plot 2d histogram
        h, xedges, yedges, im = ax.hist2d(
            ds["x_plot"],
            ds["y_plot"],
            bins=bins,
            cmap=cmap,
            range=[xlims, ylims],
            norm=norm,
            density=True,
            cmin=1 / N,
        )

        if plot_one_to_one:
            ax.plot(xlims, ylims, "k--")

        if plot_mean_std:
            # Plot rmse, bias, and std
            diff = ds["y_plot"]
            diff["x"] = data_x[i]
            # Round x to bins of 2d histogram
            diff["x"] = (
                (np.floor(diff["x"] / xlims[1] * h.shape[0]) % h.shape[0])
                * xlims[1]
                / h.shape[0]
            )
            count = diff.groupby("x").count()
            rmse = diff.groupby("x").apply(lambda x: np.sqrt((x**2).mean()))
            bias = diff.groupby("x").mean()
            std = diff.groupby("x").std()

            ax.plot(
                bias["x"],
                (bias).where(count > 20),
                color="k",
                linewidth=2,
            )
            ax.plot(
                bias["x"],
                (bias - std).where(count > 20),
                color="k",
                linewidth=2,
                linestyle="--",
            )
            ax.plot(
                bias["x"],
                (bias + std).where(count > 20),
                color="k",
                linewidth=2,
                linestyle="--",
            )

        if show_infos:
            # Calculate RMSE, bias, and R-value
            rmse = np.sqrt(np.mean((ds["x_plot"] - ds["y_plot"]) ** 2))
            bias = np.mean(ds["x_plot"] - ds["y_plot"])
            corr = np.corrcoef(ds["x_plot"], ds["y_plot"])[0, 1]

            # Add RMSE and correlation to plot
            ax.text(
                0.05,
                0.95,
                f"RMSE: {rmse:.1f} ppm\nBias: {bias:.1f} ppm\nR: {corr:.2f}\nN: {N}",
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=1.0, edgecolor="lightgray"),
            )

        if aspect_equal:
            ax.set_aspect("equal")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        fw = ax.xaxis.label.get_fontweight()
        station_code = co2.code.sel(station=station).values
        station_height = co2["height"].sel(station=station).values
        ax.text(
            0.5,
            1.06,
            f"{station_code} {station_height}m",
            transform=ax.transAxes,
            va="center",
            ha="center",
            fontsize=15,
            fontweight=fw,
        )

        # Add fancy box for title
        lw = ax.spines["left"].get_linewidth()
        offwhite = "#F8F8FF"
        box = patches.FancyBboxPatch(
            (0.0, 1.0),
            1.0,
            0.12,
            boxstyle="square,pad=0.0",
            transform=ax.transAxes,
            edgecolor="lightgray",
            facecolor=offwhite,
            lw=lw,
            zorder=-10,
        )
        fig.patches.extend([box])

    # Set axis labels
    for r in range(n_rows):
        axs[r, 0].set_ylabel(ylabel)
    for c in range(col_wrap):
        axs[-1, c].set_xlabel(xlabel)

    # Delete empty plots
    for i in range(n_plots, len(axs.flatten())):
        fig.delaxes(axs.flatten()[i])

    # One colorbar for all plots
    if im is None:
        raise ValueError("No valid data was plotted")

    x0, y0, dx, dy = axs[-1, -2].get_position().bounds
    new_ax = fig.add_axes((x0 + dx + 0.04, y0, 0.02, dy))
    label = "Log density" if norm == "log" else "Density"
    cbar = fig.colorbar(im, cax=new_ax, label=label)
    cbar.set_ticks([])

    plt.savefig(
        fig_path,
        metadata=get_metadata(suptitle),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_tracer_model_scatter_plots(fig_path: str | Path):
    combinations = [
        ("background", "measured"),
        ("modeled_Origins.earth", "measured"),
        ("modeled_TNO", "measured"),
        ("modeled_Origins.earth", "modeled_TNO"),
    ]

    xlims = (400, 500)
    ylims = (400, 500)
    col_wrap = 4
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
    dynamic_background, co2, co2_model = cache_data()
    co2_model = co2_model.sel(best_sim_id=0)
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
    bias_background = (dynamic_background.co2 - x).mean(dim="time")
    rmse_background = np.sqrt(((dynamic_background.co2 - x) ** 2).mean(dim="time"))

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
    dynamic_background, co2, co2_model = cache_data()
    gral_co2 = co2_model.sel(prior=inventory).sel(best_sim_id=slice(0, n_best))
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
        (
            gral_co2.isel(best_sim_id=0)
            .sel(station=station)
            .rolling(time=rolling)
            .mean()
        ).sel(time=time_slice).plot(
            ax=ax,
            label="Model",
        )

        # Model uncertainty
        extended_co2 = (
            gral_co2.sel(station=station)
            .isel(best_sim_id=slice(0, 10))
            .rolling(time=rolling)
            .mean()
        ).sel(time=time_slice)
        ax.fill_between(
            extended_co2.time,
            extended_co2.min(dim="best_sim_id"),
            extended_co2.max(dim="best_sim_id"),
            alpha=0.3,
            label="Model uncertainty",
        )

        # Measurement
        co2.sel(station=station).sel(time=time_slice).plot(
            ax=ax,
            label="Measurement",
        )  # type: ignore

        # Background
        dynamic_background.co2.sel(time=time_slice).plot(
            ax=ax,
            label="Background",
        )

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


def station_line_plot(
    model: xr.DataArray,
    model_data: list[xr.DataArray],
    measurement_data: list[xr.DataArray],
    background_data: list[xr.DataArray],
    labels: list[str],
    groupby: str,
    suptitle: str,
    ylabel: str,
    ylims: tuple,
    col_wrap=4,
):
    n_plots = len(model_data)
    n_rows = int(np.ceil(n_plots / col_wrap))
    fig, axs = plt.subplots(
        n_rows,
        col_wrap,
        figsize=(4.5 * col_wrap, 4 * n_rows),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.2, "wspace": 0.1},
    )

    fig.suptitle(suptitle, fontsize=16)

    for i, station in enumerate(model.station):
        ax = axs.flatten()[i]

        ds = xr.Dataset(
            {
                "model": model_data[i],
                "measurement": measurement_data[i],
                "background": background_data[i],
            }
        )
        ds = ds.dropna(dim="time")

        N = len(ds.time)
        if N < 100:
            print(f"Skipping {station.values} due to insufficient data")
            ax.axis("off")
            continue

        if groupby == "hour":
            ds[groupby] = ds["time"].dt.hour
            time = xr.DataArray(
                np.arange(0, 24),
                dims=["hour"],
                coords={"hour": np.arange(0, 24)},
            )
            xticks = np.arange(0, 30, 6)
            xlabel = "Time of day [h]"
        elif groupby == "day":
            ds[groupby] = ds["time"].dt.dayofweek
            time = xr.DataArray(
                np.arange(0, 7),
                dims=["day"],
                coords={"day": np.arange(0, 7)},
            )
            xticks = np.arange(1, 7, 1)
            xlabel = "Day of week"
        elif groupby == "week":
            ds[groupby] = ds["time"].dt.isocalendar().week
            time = xr.DataArray(
                np.arange(1, 53),
                dims=["week"],
                coords={"week": np.arange(1, 53)},
            )
            xticks = np.arange(4, 56, 8)
            xlabel = "Week of year"
        elif groupby == "month":
            ds[groupby] = ds["time"].dt.month
            time = xr.DataArray(
                np.arange(1, 13),
                dims=["month"],
                coords={"month": np.arange(1, 13)},
            )
            xticks = np.arange(1, 13, 1)
            xlabel = "Month of year"
        else:
            raise ValueError(f"Unknown groupby: {groupby}")

        ds = ds.set_coords(groupby)
        for j, var in enumerate(["model", "measurement", "background"]):
            # data = ds.groupby(groupby).median()[var]
            data = ds.groupby(groupby).mean()[var]
            # Set missing times to nan but confirm to time
            data = data.reindex({groupby: time}, method=None)
            ax.plot(
                time,
                data,
                label=labels[j],
            )
            if not var == "background":
                lower_q = ds.groupby(groupby).quantile(0.25)[var]
                lower_q = lower_q.reindex({groupby: time}, method=None)
                upper_q = ds.groupby(groupby).quantile(0.75)[var]
                upper_q = upper_q.reindex({groupby: time}, method=None)
                ax.fill_between(
                    time,
                    lower_q,
                    upper_q,
                    alpha=0.2,
                    label=f"{labels[j]} 25-75% quantile",
                )
        ax.set_xticks(xticks)

        # ax.fill_between(
        #     np.unique(
        #         ds["model"]
        #         .groupby(f"time.{groupby}")
        #         .mean()
        #         .sel(station=station)[groupby]
        #     ),
        #     ds["model_min"].groupby(f"time.{groupby}").mean().sel(station=station),
        #     ds["model_max"].groupby(f"time.{groupby}").mean().sel(station=station),
        #     color=color,
        #     alpha=0.3,
        # )

        # xlims = ds[groupby].min(), ds[groupby].max()
        # ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        # ax.legend()
        # plt.colorbar(im, ax=ax, label="count")
        # ax.set_title(co2.label.sel(station=station).values)
        fw = ax.xaxis.label.get_fontweight()
        if "code" in model.coords:
            station_label = (
                f"{model.code.sel(station=station).values} "
                f"{model['height'].sel(station=station).values}m"
            )
        else:
            station_label = str(station.values)
        ax.text(
            0.5,
            1.06,
            station_label,
            transform=ax.transAxes,
            va="center",
            ha="center",
            # rotation=-90,
            fontsize=12,
            fontweight=fw,
        )
        # Linewidth of spines
        lw = ax.spines["left"].get_linewidth()
        offwhite = "#F8F8FF"
        box = patches.FancyBboxPatch(
            (0.0, 1.0),
            1.0,
            0.12,
            boxstyle="square,pad=0.0",
            transform=ax.transAxes,
            edgecolor="lightgray",
            facecolor=offwhite,
            lw=lw,
            zorder=-10,
        )
        fig.patches.extend([box])

    # Set axis labels
    for r in range(n_rows):
        axs[r, 0].set_ylabel(ylabel)
    for c in range(col_wrap):
        axs[-1, c].set_xlabel(xlabel)  # type: ignore
    # Delete splines of empty plots
    for i in range(n_plots, len(axs.flatten())):
        axs.flatten()[i].axis("off")
    # One legend for all plots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs.flatten()[n_plots].legend(handles, labels, loc="upper left", title="Legend")


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
            best_sim_id=0,
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
    labels = ["Model", "Measurement", "Background"]

    ylabel = "CO2 [ppm]"
    ylims = (410, 460)
    suptitle = f"CO2 vs. {groupby} ({prior}, {loss_type}) {time_str} {season}"

    col_wrap = 10
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
        plt.figure(figsize=(18, 6))
        (co2_model.sel(prior=prior, best_sim_id=0)).sel(
            station=station, time=afternoon_mask
        ).resample(time="1D").mean().plot(add_legend=False)
        co2.sel(station=co2_model.station).co2.sel(
            station=station, time=afternoon_mask
        ).resample(time="1D").mean().plot(add_legend=False)
        background.co2.sel(time=afternoon_mask).resample(time="1D").mean().plot(
            add_legend=False
        )
        plt.legend(["Modeled", "Measured", "Background"])
        if afternoon_only:
            title = (
                f"Daily mean afternoon ({afternoon_start_hour}-{afternoon_end_hour} h) "
                f"CO$_2$ at station {station} ({prior}, {loss_type})"
            )
        else:
            title = f"Daily mean CO$_2$ at station {station} ({prior}, {loss_type})"
        plt.title(title)
        from_template = str(fig_path).format(
            station=station,
        )
        plt.savefig(
            from_template,
            metadata=get_metadata(title),
            bbox_inches="tight",
        )
        plt.close()


@lru_cache()
def _load_sector_enhancement_data(
    loss_type: str = "rmse - filter: True",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load model sector enhancement, measured CO2 and background CO2.

    Loads the concentration time series retaining the ``type`` dimension so that
    individual sector contributions remain visible, converts units to ppm, and
    aligns everything to the availability mask of the measurements.

    Parameters
    ----------
    loss_type : str
        Loss-type selector passed to ``sel(loss_type=...)``.

    Returns
    -------
    model_enhancement : xr.DataArray
        Sector-resolved enhancement (dims: time, station, type), masked to
        measurement availability, in ppm.
    co2 : xr.DataArray
        Measured CO2 (dims: time, station), in ppm.
    background : xr.DataArray
        Binned background CO2, broadcast to station heights (dims: time, station).
    """
    co2 = ggp.load("co2_measurements", CONFIG).co2
    measurements_available = co2.notnull()

    conc_ds = xr.open_dataset(
        Path(CONFIG["output_path"]) / ggp.config.CONCENTRATION_TIMESERIES_FILE_NAME
    )

    model_enhancement = (
        conc_ds["co2_timeseries"]
        .sel(
            loss_type=loss_type,
            best_sim_id=0,
        )
        .where(measurements_available)
    )

    background = ggp.load("background_co2", CONFIG)["binned_background_by_label"].sel(
        height_bins=model_enhancement.height
    )
    background = background.where(measurements_available)

    return model_enhancement, co2, background


def _append_mean_station(
    da: xr.DataArray, station_name: str = "Mean", add_sunday: bool = True
) -> xr.DataArray:
    """Append a virtual station that is the mean across all stations.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with a "station" dimension.
    station_name : str
        Name for the appended mean station.
    add_sunday : bool
        Whether to also append a mean station that only includes Sundays (dayofweek=6).

    Returns
    -------
    xr.DataArray
        Original DataArray with an additional station that is the mean across all
        stations.
    """
    mean = da.mean(dim="station").expand_dims(station=[station_name])
    station_coords = [
        coord
        for coord in da.coords
        if coord != "station" and "station" in da[coord].dims
    ]
    da_clean = da.drop_vars(station_coords)
    mean_clean = mean.drop_vars(station_coords, errors="ignore")

    if add_sunday:
        mean_clean_sunday = mean_clean.sel(
            time=mean_clean.time.dt.dayofweek == 6
        ).assign_coords(station=[f"{station_name} Sundays"])
        return xr.concat(
            [da_clean, mean_clean, mean_clean_sunday],
            dim="station",
            join="outer",
            coords="different",
            compat="equals",
        )
    else:
        return xr.concat(
            [da_clean, mean_clean],
            dim="station",
            join="outer",
            coords="different",
            compat="equals",
        )


def station_sector_plot(
    model_enhancement: xr.DataArray,
    co2: xr.DataArray,
    background: xr.DataArray,
    inventory: str,
    groupby: Literal["hour", "day", "week", "month"],
    suptitle: str,
    ylabel: str,
    ylims: tuple,
    col_wrap: int = 10,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot stacked sector contributions per station, grouped by a time dimension.

    Mirrors the layout of :func:`station_line_plot`: a grid of subplots with one
    panel per station, shared y-axis, titled header boxes, and a shared legend.
    Each panel shows the anthropogenic sector contributions (stacked fill_between)
    from *inventory* on top of background + VPRM, overlaid with the measured CO2
    and the background line.

    Parameters
    ----------
    model_enhancement : xr.DataArray
        Sector-resolved enhancement with dims (time, station, type).
    co2 : xr.DataArray
        Measured CO2 with dims (time, station).
    background : xr.DataArray
        Background CO2 with dims (time, station).
    inventory : str
        Substring to filter ``model_enhancement.type`` for anthropogenic sectors
        (e.g. ``"TNO"`` or ``"Origins.earth"``).
    groupby : str
        One of ``"hour"``, ``"day"``, ``"week"``, ``"month"``.
    suptitle : str
        Figure super-title.
    ylabel : str
        Y-axis label.
    ylims : tuple
        (ymin, ymax) for all subplots.
    col_wrap : int
        Number of columns in the subplot grid.

    Returns
    -------
    fig, axs
    """
    if groupby == "hour":
        groupby_key = "time.hour"
        time_vals = np.arange(0, 24)
        xticks = np.arange(0, 25, 6)
        xlabel = "Time of day [h]"
    elif groupby == "day":
        groupby_key = "time.dayofweek"
        time_vals = np.arange(0, 7)
        xticks = np.arange(0, 7)
        xlabel = "Day of week"
    elif groupby == "week":
        groupby_key = None  # handled via isocalendar below
        time_vals = np.arange(1, 53)
        xticks = np.arange(4, 56, 8)
        xlabel = "Week of year"
    elif groupby == "month":
        groupby_key = "time.month"
        time_vals = np.arange(1, 13)
        xticks = np.arange(1, 13)
        xlabel = "Month of year"
    else:
        raise ValueError(f"Unknown groupby: {groupby!r}")

    def _groupby(da: xr.DataArray):
        """Return a groupby object, using isocalendar for weeks."""
        if groupby == "week":
            week = da.time.dt.isocalendar().week.astype(int).rename("week")
            return da.groupby(week)
        return da.groupby(groupby_key)

    model_enhancement = _append_mean_station(model_enhancement)
    co2 = _append_mean_station(co2)
    background = _append_mean_station(background)

    stations = model_enhancement.station.values
    n_plots = len(stations)
    n_rows = int(np.ceil(n_plots / col_wrap))

    fig, axs = plt.subplots(
        n_rows,
        col_wrap,
        figsize=(4.5 * col_wrap, 4 * n_rows),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.2, "wspace": 0.1},
    )
    fig.suptitle(suptitle, fontsize=16)

    for i, (s, ax) in enumerate(zip(stations, axs.flatten())):
        hourly = (
            _groupby(
                model_enhancement.sel(
                    type=model_enhancement.type.str.contains(inventory),
                    station=s,
                )
            )
            .mean()
            .to_pandas()
        )
        vprm = (
            _groupby(
                model_enhancement.sel(
                    type=model_enhancement.type.str.contains("VPRM"),
                    station=s,
                ).sum(dim="type")
            )
            .mean()
            .to_pandas()
        )
        bg = _groupby(background.sel(station=s)).mean().to_pandas()
        meas = _groupby(co2.sel(station=s)).mean().to_pandas()

        x = time_vals
        base = (bg + vprm).reindex(time_vals).values  # type: ignore[call-overload]
        cumulative = base.copy()
        top = cumulative.copy()

        for col in hourly.columns:
            col_vals = hourly[col].reindex(time_vals).values  # type: ignore
            top = cumulative + col_vals  # type: ignore[operator]
            ax.fill_between(x, cumulative, top, alpha=0.5, label=col)
            cumulative = top

        ax.plot(x, top, color="k", linewidth=0.8, label="Model total")
        ax.plot(
            time_vals,
            meas.reindex(time_vals).values,  # type: ignore[call-overload]
            color="k",
            linestyle="-",
            linewidth=1.5,
            label="Measurements",
        )
        ax.plot(
            time_vals,
            bg.reindex(time_vals).values,  # type: ignore[call-overload]
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label="Background",
        )

        ax.set_xticks(xticks)
        ax.set_ylim(ylims)
        ax.grid(alpha=0.3)

        fw = ax.xaxis.label.get_fontweight()
        station_label = str(s)
        if "height" in model_enhancement.coords:
            h = model_enhancement["height"].sel(station=s).values
            station_label = f"{s} {h}m"
        ax.text(
            0.5,
            1.06,
            station_label,
            transform=ax.transAxes,
            va="center",
            ha="center",
            fontsize=10,
            fontweight=fw,
        )
        lw = ax.spines["left"].get_linewidth()
        offwhite = "#F8F8FF"
        box = patches.FancyBboxPatch(
            (0.0, 1.0),
            1.0,
            0.12,
            boxstyle="square,pad=0.0",
            transform=ax.transAxes,
            edgecolor="lightgray",
            facecolor=offwhite,
            lw=lw,
            zorder=-10,
        )
        fig.patches.extend([box])

    for r in range(n_rows):
        axs[r, 0].set_ylabel(ylabel)
    for c in range(col_wrap):
        axs[-1, c].set_xlabel(xlabel)  # type: ignore

    for j in range(n_plots, len(axs.flatten())):
        axs.flatten()[j].axis("off")

    handles, labels = axs.flatten()[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_ax = (
        axs.flatten()[n_plots] if n_plots < len(axs.flatten()) else axs.flatten()[-1]
    )
    legend_ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        title="Legend",
    )

    return fig, axs


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
    :func:`_load_sector_enhancement_data`, delegates the rendering to
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
        Loss-type filter forwarded to :func:`_load_sector_enhancement_data`.
    ylabel : str
        Y-axis label.
    ylims : tuple
        (ymin, ymax) applied to all subplots.
    col_wrap : int
        Number of subplot columns.
    """
    model_enhancement, co2, background = _load_sector_enhancement_data(
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
        Loss-type filter forwarded to :func:`_load_sector_enhancement_data`.
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

    model_enhancement, co2, background = _load_sector_enhancement_data(
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
            axs[i].grid()

    axs[-1].legend(
        ["Model + Background", "CO$_2$", "Background"],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.tight_layout()

    plt.savefig(fig_path, metadata=get_metadata(suptitle), bbox_inches="tight")
    plt.close(fig)
