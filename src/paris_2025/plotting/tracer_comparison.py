"""Plotting functions for comparing modeled and measured CO2 concentrations."""

from functools import lru_cache
from pathlib import Path

import ggpymanager as ggp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Remove for colormap "flare" # noqa: F401
import xarray as xr
from dask.diagnostics.progress import ProgressBar

import paris_2025 as p
from paris_2025.plotting.common import get_metadata


@lru_cache()
def cache_data():
    conc_series = xr.open_mfdataset(
        p.CONFIG["output_path"] + "/" + ggp.config.CONCENTRATION_TIMESERIES_FILE_NAME
    ).sel(best_sim_id=0)
    t = conc_series.type
    mask = xr.concat(
        [
            t.str.contains("Origins.earth") | t.str.contains("VPRM 2023"),
            t.str.contains("TNO") | t.str.contains("VPRM 2023"),
        ],
        dim="prior",
    )
    mask["prior"] = ["Origins.earth", "TNO"]
    time_series = ggp.utils.ugm3_to_ppm(
        conc_series.co2_timeseries.sel(loss_type="rmse - filter: True")
        .where(mask)
        .sum("type"),
        "co2",
    )
    with ProgressBar():
        time_series = time_series.compute()  # type: ignore
    dynamic_background = p.background.get_background_co2()
    co2 = p.tracers.get_co2_measurements()
    co2_model = dynamic_background.co2.reset_coords(
        drop=True
    ) + time_series.reset_coords(names=["x", "y"], drop=True)

    return dynamic_background, co2, co2_model


def get_plot_data(name, afternoon_only=False, main_wind_direction_only=False):
    dynamic_background, co2, co2_model = cache_data()
    if name == "background":
        data = [
            dynamic_background.co2.reset_coords(drop=True).assign_coords(station=s)
            for s in co2_model.station.values
        ]
        title = "Background CO2"
        axis_label = "Background CO2 [ppm]"
    elif name == "measured":
        data = [co2.co2.sel(station=s) for s in co2_model.station.values]
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
        box = mpl.patches.FancyBboxPatch(  # type: ignore
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
    # Filter out stations outside the GRAL domain
    co2 = co2.sel(station=co2.in_gral_domain)
    high_cost_mask = co2.instrument == "Picarro"
    high_cost_stations = list(co2.station.sel(station=high_cost_mask).values)
    mid_cost_stations = list(co2.station.sel(station=~high_cost_mask).values)
    x = co2.co2.sel(station=high_cost_stations + list(mid_cost_stations)).sel(
        time=(12 <= co2.co2.time.dt.hour) & (co2.co2.time.dt.hour <= 16)
    )
    y_tno = (
        co2_model.sel(prior="TNO").sel(
            station=high_cost_stations + list(mid_cost_stations)
        )
        # + background
    ).sel(time=(12 <= co2.co2.time.dt.hour) & (co2.co2.time.dt.hour <= 16))
    y = (
        co2_model.sel(prior="Origins.earth").sel(
            station=high_cost_stations + list(mid_cost_stations)
        )
        #   + background
    ).sel(time=(12 <= co2.co2.time.dt.hour) & (co2.co2.time.dt.hour <= 16))

    position_x = co2.co2.sel(
        station=high_cost_stations + list(mid_cost_stations)
    ).x.values
    position_y = co2.co2.sel(
        station=high_cost_stations + list(mid_cost_stations)
    ).y.values

    bias = (y - x).mean(dim="time")
    rmse = np.sqrt(((y - x) ** 2).mean(dim="time"))
    bias_tno = (y_tno - x).mean(dim="time")
    rmse_tno = np.sqrt(((y_tno - x) ** 2).mean(dim="time"))
    bias_background = (dynamic_background.co2 - x).mean(dim="time")
    rmse_background = np.sqrt(((dynamic_background.co2 - x) ** 2).mean(dim="time"))

    t = co2.co2.sel(station=high_cost_stations + list(mid_cost_stations)).type.values

    y_labels = ["Mean Bias", "RMSE"]
    plot_data = [(bias, bias_tno, bias_background), (rmse, rmse_tno, rmse_background)]
    x_data = [co2.co2.height, position_x, position_y]
    x_labels = ["Height above ground level [m]", "x position [m]", "y position [m]"]

    for x_label, xd in zip(x_labels, x_data):
        for y_label, yd in zip(y_labels, plot_data):
            fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
            for station_type in np.unique(co2.co2.type):
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
    co2_measurements,
    gral_co2,
    background,
    stations,
    start_time,
    duration=14,
    rolling=3,
):
    """Plot time series comparison of modeled vs measured CO2.

    Parameters
    ----------
    fig_path : str or Path
        Path to save the figure
    co2_measurements : xr.Dataset
        Measured CO2 data
    gral_co2 : xr.DataArray
        Modeled CO2 data
    background : xr.DataArray
        Background CO2
    stations : list
        List of station IDs to plot
    start_time : str
        Start time for the plot
    duration : int, optional
        Duration in days
    rolling : int, optional
        Rolling mean window size
    """
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
            gral_co2.sel(station=station)
            .rolling(time=rolling)
            .mean()
            # gral_co2.isel(rank=0).sel(station=station).rolling(time=rolling).mean()
        ).sel(time=time_slice).plot(
            ax=ax,
            label="Model",
        )

        # Model uncertainty
        """
        extended_co2 = (
            gral_co2.sel(station=station)
            .isel(rank=slice(0, 10))
            .rolling(time=rolling)
            .mean()
        ).sel(time=time_slice)
        ax.fill_between(
            extended_co2.time,
            extended_co2.min(dim="rank"),
            extended_co2.max(dim="rank"),
            alpha=0.3,
            label="Model uncertainty",
        )
        """

        # Measurement
        co2_measurements["co2"].sel(station=station).sel(time=time_slice).plot(
            ax=ax,
            label="Measurement",
        )

        # Background
        background.sel(time=time_slice).plot(
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
            f"{co2_measurements.code.sel(station=station).values}"
            f" {co2_measurements['height'].sel(station=station).values}m"
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
