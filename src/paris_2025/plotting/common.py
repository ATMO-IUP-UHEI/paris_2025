import datetime
import sys
from typing import Literal

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from paris_2025.plotting import DATA_COLORS, RC_PARAMS
from paris_2025.plotting._loaders import load_combined_data


def get_metadata(description=None):
    """Get metadata for the plots."""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    caller_name = sys._getframe().f_back.f_code.co_name  # type: ignore
    _description = f"Created by function '{caller_name}' on {date_str}."
    if description is not None:
        _description += f"\n{description}"
    return {"Description": _description}


def save_table_as_png(df, filename, caption="", figsize=None):
    """Save a DataFrame as a PNG with background gradient styling similar to pandas
    style.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save
    filename : str
        Output filename (should end with .png)
    caption : str
        Table caption
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated.
    """
    # Normalize values column-wise for color mapping
    # Choose colormap separately for each column
    normalized = df.copy()
    colormaps = {}

    for col in df.columns:
        col_data = df[col].values
        has_negative = (col_data < 0).any()

        # Choose colormap based on data in this column
        if has_negative:
            colormaps[col] = plt.get_cmap("RdBu_r")  # Diverging colormap
            # For diverging colormap, normalize symmetrically around 0
            max_abs = max(abs(col_data.min()), abs(col_data.max()))
            if max_abs > 0:
                normalized[col] = (col_data + max_abs) / (2 * max_abs)
            else:
                normalized[col] = 0.5
        else:
            colormaps[col] = plt.get_cmap("Blues")  # Sequential colormap
            # For sequential colormap, normalize from min to max
            if col_data.max() != col_data.min():
                normalized[col] = (col_data - col_data.min()) / (
                    col_data.max() - col_data.min()
                )
            else:
                normalized[col] = 0.5

    # Auto-calculate figure size if not provided
    if figsize is None:
        n_rows, n_cols = df.shape
        figsize = (max(8, n_cols * 1.5), max(4, n_rows * 0.4 + 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("tight")
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style cells with background gradient
    for i in range(len(df)):
        for j, col in enumerate(df.columns):
            cell = table[(i + 1, j)]
            cm = colormaps[col]
            color = cm(normalized.iloc[i, j])
            cell.set_facecolor(color)
            cell.set_text_props(weight="normal")

    # Style header
    for j in range(len(df.columns)):
        cell = table[(0, j)]
        cell.set_facecolor("#40466e")
        cell.set_text_props(weight="bold", color="white")

    # Style row labels
    for i in range(len(df)):
        cell = table[(i + 1, -1)]
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold")

    # Add caption
    if caption:
        plt.title(caption, fontsize=12, weight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Table saved to {filename}")


def station_scatter_plot(
    fig_path,
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
            rmse = np.sqrt(np.mean((ds["x_plot"] - ds["y_plot"]) ** 2)).compute()
            bias = (ds["x_plot"] - ds["y_plot"]).mean().compute()
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

        station_code = co2.code.sel(station=station).values
        station_height = co2["height"].sel(station=station).values

        title = f"{station_code} {station_height}m"
        add_title(ax, fig, title)

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

    width = RC_PARAMS["figure.figsize"][0]
    fig, axs = plt.subplots(
        n_rows,
        col_wrap,
        figsize=(width, 2 * n_rows),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.2, "wspace": 0.1},
    )

    # fig.suptitle(suptitle, fontsize=16)

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
            xticks = np.arange(0, 24, 6)
            xlabel = "Hour [UTC]"
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
        for j, var in enumerate(["background", "measurement", "model"]):
            # data = ds.groupby(groupby).median()[var]
            data = ds.groupby(groupby).mean()[var]
            # Set missing times to nan but confirm to time
            data = data.reindex({groupby: time}, method=None)
            ax.plot(
                time,
                data,
                label=labels[j],
                color=DATA_COLORS[var],
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
                    color=DATA_COLORS[var],
                )
        ax.set_xticks(xticks)

        ax.set_ylim(ylims)
        # Minor ticks every 10 units
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.grid(True, which="both")

        # if "code" in model.coords:
        #     station_label = (
        #         f"{model.code.sel(station=station).values} "
        #         f"{model['height'].sel(station=station).values}m"
        #     )
        # else:
        station_label = str(station.values)
        add_title(ax, fig, station_label)

    # Set axis labels
    for r in range(n_rows):
        axs[r, 0].set_ylabel(ylabel)
        for c in range(1, col_wrap):
            axs[r, c].tick_params(left=False, which="both")

    for ax in axs.flatten()[n_plots - col_wrap : n_plots]:
        # ax.set_xticklabels(xticks)  # type: ignore
        ax.tick_params(labelbottom=True)
        ax.set_xlabel(xlabel)  # type: ignore

    for ax in axs.flatten()[: n_plots - col_wrap]:
        ax.tick_params(bottom=False)

    # Delete splines of empty plots
    for i in range(n_plots, len(axs.flatten())):
        axs.flatten()[i].axis("off")
    # One legend for all plots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs.flatten()[n_plots].legend(
        handles,
        labels,
        title="Legend",
        bbox_to_anchor=(0.0, -0.1),
        loc="lower left",
    )


def get_nan_value(dtype):
    if np.issubdtype(dtype, np.floating):
        return np.nan
    elif np.issubdtype(dtype, np.integer):
        return -9999
    elif np.issubdtype(dtype, np.datetime64):
        return np.datetime64("NaT")
    elif np.issubdtype(dtype, str):
        return ""
    elif np.issubdtype(dtype, bool):
        return False
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _append_mean_station(
    da: xr.DataArray,
    station_name: str = "Mean",
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Append a virtual station that is the mean across all stations.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with a "station" dimension.
    station_name : str
        Name for the appended mean station.
    mask : xr.DataArray, optional
        Boolean mask to apply before averaging. Should be broadcastable to da.

    Returns
    -------
    xr.DataArray
        Original DataArray with an additional station that is the mean across all
        stations.
    """
    if mask is None:
        mask = xr.ones_like(da, dtype=bool)
    mean = da.where(mask).mean(dim="station").expand_dims(station=[station_name])
    # Convert station coordinate to same type as da.station
    for coord in da.coords:
        if coord not in mean.coords:
            mean.coords[coord] = (
                da[coord].dims,
                np.array([get_nan_value(da[coord].dtype)]),
            )

    da_appended = xr.concat(
        [da, mean],
        dim="station",
        join="outer",
        coords="minimal",
        compat="equals",
    )
    da_appended["station"] = da_appended["station"].astype(str)
    return da_appended


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
        xlabel = "Hour [UTC]"
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


def plot_station_hist2d(x: xr.DataArray, y: xr.DataArray, ax: Axes) -> Axes:
    # Test if data has the right dimensions
    assert isinstance(x, xr.DataArray)
    assert isinstance(y, xr.DataArray)

    # Add the plot to ax
    bins = 100
    cmap = "inferno"
    N = len(x)
    xlims = (300, 600)
    ylims = (300, 600)
    ax.hist2d(
        x,
        y,
        bins=bins,
        cmap=cmap,
        range=[xlims, ylims],
        # norm=norm,
        density=True,
        cmin=1 / N,
    )

    return ax


def plot_station_groupby(
    co2_model: xr.DataArray,
    background: xr.DataArray,
    co2: xr.DataArray,
    groupby: str,
    ax: Axes,
) -> Axes:
    for label, da in zip(
        ["background", "measurement", "model"],
        [background, co2, co2_model],
    ):
        if groupby == "hour":
            da[groupby] = da["time"].dt.hour
            xticks = np.arange(0, 30, 6)
            xlabel = "Hour [UTC]"
        else:
            raise KeyError()

        mean = da.groupby(groupby).mean()
        ax.plot(mean[groupby], mean, label=label.capitalize(), color=DATA_COLORS[label])
        ax.set_xticks(xticks)
        ax.set_xticks(xticks)
        ax.set_xlabel(xlabel)

        if label != "background":
            # Plot quantiles as shaded area
            lower_q = da.groupby(groupby).quantile(0.25)
            # lower_q = lower_q.reindex({groupby: time}, method=None)
            upper_q = da.groupby(groupby).quantile(0.75)
            # upper_q = upper_q.reindex({groupby: time}, method=None)
            ax.fill_between(
                lower_q[groupby],
                lower_q,
                upper_q,
                alpha=0.2,
                label=f"{label.capitalize()} 25-75% quantile",
                color=DATA_COLORS[label],
                # label=f"{labels[j]} 25-75% quantile",
            )

    return ax


def plot_station_groupby_sector(
    da: xr.DataArray,
    background: xr.DataArray,
    co2: xr.DataArray,
    inventory: str,
    groupby: str,
    ax: Axes,
) -> Axes:
    from paris_2025.plotting import INVENTORY_COLORS

    if groupby == "hour":
        da[groupby] = da["time"].dt.hour
        xticks = np.arange(0, 30, 6)
        time_vals = np.arange(0, 24)
        xlabel = "Hour [UTC]"
    else:
        raise KeyError()
    mean = (
        da.sel(type=da.type.str.contains(inventory)).groupby(groupby).mean().to_pandas()
    )
    vprm = (
        da.sel(type=da.type.str.contains("VPRM"))
        .sum("type")
        .groupby(groupby)
        .mean()
        .to_pandas()
    )
    bg = background.groupby(da[groupby]).mean().to_pandas()
    meas = co2.groupby(da[groupby]).mean().to_pandas()

    x = time_vals
    base = (bg + vprm).values  # type: ignore[call-overload]
    cumulative = base.copy()
    top = cumulative.copy()

    for col in mean.columns:
        col_vals = mean[col].values  # type: ignore
        top = cumulative + col_vals  # type: ignore[operator]
        ax.fill_between(
            x,
            cumulative,  # type: ignore
            top,
            alpha=0.5,
            label=col,
            color=INVENTORY_COLORS[col],
        )
        cumulative = top

    ax.plot(x, top, color="k", linewidth=0.8, label="Model total")  # type: ignore
    ax.plot(
        x,
        meas.values,  # type: ignore[call-overload]
        color="k",
        linestyle="-",
        linewidth=1.5,
        label="Measurements",
    )
    ax.plot(
        x,
        bg.values,  # type: ignore[call-overload]
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="Background",
    )

    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    return ax


def add_title(ax: Axes, fig, title):
    # Get font size and weight from axes
    fs = RC_PARAMS["legend.fontsize"]
    fw = ax.xaxis.label.get_fontweight()
    # Add title as text to allow for better positioning
    ax.text(
        0.5,
        1.055,
        title,
        transform=ax.transAxes,
        va="center",
        ha="center",
        fontsize=fs,
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
    return ax


def share_ax_lim(
    type: str, axs: list[Axes], xlim: tuple | None = None, ylim: tuple | None = None
):
    match type:
        case "x":
            sharex = True
            sharey = False
        case "y":
            sharex = False
            sharey = True
        case "both":
            sharex = True
            sharey = True
        case _:
            raise ValueError("Argument type requires 'x', 'y', or 'both'")

    if sharex:
        if xlim is None:
            lims = np.array([ax.get_xlim() for ax in axs])
            xlim = np.min(lims[:, 0]), np.max(lims[:, 1])
            assert lims.shape == (len(axs), 2)
        for ax in axs:
            ax.set_xlim(*xlim)
    if sharey:
        if ylim is None:
            lims = np.array([ax.get_ylim() for ax in axs])
            ylim = np.min(lims[:, 0]), np.max(lims[:, 1])
            assert lims.shape == (len(axs), 2)
        for ax in axs:
            ax.set_ylim(*ylim)


def get_data(name: str, filters=[]) -> xr.DataArray:
    combined, model_enhancement = load_combined_data()

    match name:
        case "Origins.earth":
            data = combined.sel(dataset="Origins.earth")
        case "TNO":
            data = combined.sel(dataset="TNO")
        case "CO2":
            data = combined.sel(dataset="CO2")
        case "Background":
            data = combined.sel(dataset="Background")
        case "Model Enhancement":
            data = model_enhancement
        case _:
            raise ValueError(f"Unknown dataset {name}")

    for f in filters:
        data = f(data)
    return data


def create_ax_plot(station: str, plot_info: str, ax: Axes, fig: Figure) -> Axes:
    # Parse plot info
    parts = plot_info.split()

    # Check if mask is specified
    filters = []
    if "filter" in parts:
        filter_idx = parts.index("filter")
        mask_type = parts[filter_idx + 1]
        if mask_type == "Sunday":
            filters.append(lambda x: x.sel(time=x.time.dt.dayofweek == 6))
        elif mask_type == "weekday":
            filters.append(lambda x: x.sel(time=x.time.dt.dayofweek < 5))
        elif mask_type == "afternoon":
            filters.append(lambda x: x.sel(time=x.time.dt.hour.between(12, 17)))
        else:
            raise ValueError(f"Unknown filter type {mask_type}")
        # Remove filter info from parts
        parts = parts[:filter_idx]

    # 2D histogram
    if parts[0] == "hist2d":
        assert (
            parts[2] == "vs" and len(parts) == 4
        ), "hist2d plot info should be in format 'hist2d <var1> vs <var2>'"
        x_var = parts[1]
        y_var = parts[3]
        plot_station_hist2d(
            get_data(x_var, filters).sel(station=station),
            get_data(y_var, filters).sel(station=station),
            ax,
        )

    # Groupby plot
    elif parts[0] == "groupby":
        assert (
            len(parts) == 3
        ), "groupby plot info should be in format 'groupby <time> <inventory>' where "
        "inventory is the variable to group by"
        groupby = parts[1]
        inventory = parts[2]
        plot_station_groupby(
            co2_model=get_data(inventory, filters).sel(station=station),
            background=get_data("Background", filters).sel(station=station),
            co2=get_data("CO2", filters).sel(station=station),
            groupby=groupby,
            ax=ax,
        )

    # Groupby sector plot
    elif parts[0] == "groupby_sector":
        assert (
            len(parts) == 3
        ), "groupby_sector plot info should be in format 'groupby_sector <time> "
        "<inventory>' where var is the variable to group by and inventory is the "
        "inventory to plot"
        groupby = parts[1]
        inventory = parts[2]
        plot_station_groupby_sector(
            da=get_data("Model Enhancement", filters).sel(station=station),
            background=get_data("Background", filters).sel(station=station),
            co2=get_data("CO2", filters).sel(station=station),
            inventory=parts[2],
            groupby=groupby,
            ax=ax,
        )

    # Catch
    else:
        raise ValueError(f"Unknown plot type {parts[0]}")

    add_title(ax, fig, station)
    return ax
