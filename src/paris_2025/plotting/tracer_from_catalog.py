from pathlib import Path

import ggpymanager as ggp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors as mcolors
from matplotlib import patches
from matplotlib.lines import Line2D

import paris_2025 as p
from paris_2025.config import CONFIG
from paris_2025.plotting.common import get_metadata

# Default paths from config
_DEFAULT_SOURCE_GROUPS_PATH = Path(CONFIG["source_groups_path"])
_DEFAULT_AREA_ID_PATH = Path(CONFIG["area_id_path"])


def plot_source_group_contribution_to_stations(
    fig_path_1: str | Path,
    fig_path_2: str | Path,
    source_groups_path: str | Path = _DEFAULT_SOURCE_GROUPS_PATH,
    area_id_path: str | Path = _DEFAULT_AREA_ID_PATH,
):
    """Plot contribution of source groups to stations."""
    source_groups = xr.open_dataset(source_groups_path)
    area_id = xr.open_dataset(area_id_path)
    co2 = p.model.get_co2_data()
    ppm = ggp.utils.ugm3_to_ppm(co2["concentration"], gas="CO2")
    ppm = ppm.where(ppm > 0)  # type: ignore

    ppm.plot(
        col="station", col_wrap=6, cmap="jet", norm=mcolors.LogNorm()
    )  # type: ignore
    plt.savefig(
        fig_path_1,
        metadata=get_metadata("Each pixel shows the contribution of a source group."),
    )
    plt.clf()

    fp = (
        ppm.groupby(source_groups["area_id"])
        .sum("source_group")
        .fillna(0.0)
        .mean("sim_id", skipna=False)
    )
    fp = fp.drop_vars(["x", "y", "z"])
    mindex = pd.MultiIndex.from_arrays(
        [area_id["y_center"].values, area_id["x_center"].values],
        names=["y_center", "x_center"],
    )
    fp["area_id"] = (("area_id"), mindex)
    fp = fp.unstack()
    fp.attrs = {
        "long_name": "Mean contribution over all source groups of that area",
        "units": "ppm",
    }
    fp.plot(col="station", col_wrap=6, cmap="jet")  # type: ignore
    plt.savefig(
        fig_path_2,
        metadata=get_metadata(
            "Mean over all source groups of that area,"
            " each pixel shows the contribution to a station."
        ),
    )
    plt.clf()


def plot_concentration_at_station_per_simulation(fig_path: str | Path):
    """Plot concentration at each station per simulation."""
    co2 = p.model.get_co2_data()
    ppm = ggp.utils.ugm3_to_ppm(co2["concentration"], gas="CO2")
    ppm.mean("source_group").plot()  # type: ignore
    plt.xticks(rotation=90)
    plt.savefig(
        fig_path,
        metadata=get_metadata("Mean concentration at each station per simulation."),
    )
    plt.clf()


def plot_hourly_vprm_concentration(fig_path: str | Path):
    """Plot mean hourly VPRM concentration with temporal factors applied."""
    concentration_timeseries = xr.open_dataset(
        CONFIG["output_path"] + "/" + ggp.config.CONCENTRATION_TIMESERIES_FILE_NAME
    ).sel(best_sim_id=0)

    # Calculate hourly mean concentration
    vprm_data = ggp.utils.ugm3_to_ppm(
        concentration_timeseries["co2_timeseries"]
        .sel(
            loss_type="rmse - filter: True",
            type=concentration_timeseries.type.str.contains("VPRM 2023 GEE"),
            time="2023-08",
        )
        .sum("type"),
        "co2",
    )
    # Type checking: ensure vprm_data is xr.DataArray
    assert isinstance(vprm_data, xr.DataArray)
    concentration = vprm_data.groupby("time.hour").mean("time")

    concentration.plot(cbar_kwargs={"label": "Mixing ratio [ppm]"})  # type: ignore
    plt.xticks(rotation=90)
    plt.xlabel("Station")
    plt.ylabel("Hour of day")
    plt.title("Mean hourly contribution from VPRM GEE in August 2023")
    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Mean hourly VPRM concentration with temporal factors applied."
        ),
    )
    plt.clf()


def plot_ensemble_spread_night_and_day(fig_path: str | Path):
    model_co2 = ggp.load("concentration_timeseries", config=p.CONFIG)
    series = (
        model_co2.sel(loss_type="rmse - filter: True")
        .co2_timeseries.sel(
            type=model_co2.type.str.contains("Origins.earth|VPRM"),
        )
        .sum("type")
        .compute()
    )
    # Histogram with logarithmic x-axis
    is_night = series.time.dt.hour.isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23])
    filtered = series.where(series.loss_diff < 0.1)
    diff = filtered.max("best_sim_id") - filtered.min("best_sim_id")

    plot_data = {
        "Night": (is_night, "#202020", "Night (21:00-08:00 UTC)"),
        "Day": (~is_night, "#ffcf0e", "Day (09:00-20:00 UTC)"),
    }

    fig = plt.figure(figsize=(8, 5), dpi=300)
    for name in ["Night", "Day"]:
        mask, color, label = plot_data[name]
        data = diff.where(mask).values.flatten()
        plt.hist(
            data,
            label=label,
            color=color,
            bins=10 ** np.linspace(-2, np.log10(diff.max().values), 20),
            alpha=0.5,
        )
        # Mean line
        mean_value = np.nanmean(data)
        median_value = np.nanmedian(data)

        plt.axvline(mean_value, color="white", linestyle="-", lw=2)
        plt.axvline(
            mean_value,
            color=color,
            linestyle=":",
            lw=2,
            label=f"{name} Mean: {mean_value:.2f} ppm",
        )

        plt.axvline(median_value, color="white", linestyle="-", lw=4)
        plt.axvline(
            median_value,
            color=color,
            linestyle="-",
            lw=2,
            label=f"{name} Median: {median_value:.2f} ppm",
        )
    plt.xscale("log")
    plt.legend()
    plt.xlabel("Difference between max and min CO$_2$ in ensemble (ppm)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Histogram of ensemble spread (max-min CO2) split by day and night."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_ensemble_spread_cycles(fig_path: str | Path):
    model_co2 = ggp.load("concentration_timeseries", config=p.CONFIG)
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

    bin_edges = range(0, 25, 4)
    bin_labels = [f"{s:02d}\u2013{e:02d}" for s, e in zip(bin_edges, bin_edges[1:])]
    data_by_bin = [
        diff.isel(
            time=((diff.time.dt.hour >= s) & (diff.time.dt.hour < e)).values
        ).values.flatten()
        for s, e in zip(bin_edges, bin_edges[1:])
    ]
    data_by_bin = [d[~np.isnan(d)] for d in data_by_bin]

    seasons = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }
    data_by_season = [
        diff.isel(time=diff.time.dt.month.isin(months).values).values.flatten()
        for months in seasons.values()
    ]
    data_by_season = [d[~np.isnan(d)] for d in data_by_season]

    BOX_COLOR = "steelblue"
    MEDIAN_COLOR = "firebrick"
    MEAN_COLOR = "darkorange"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)

    legend_handles = [
        patches.Patch(
            facecolor=BOX_COLOR, alpha=0.7, edgecolor="black", label="IQR (25\u201375%)"
        ),
        Line2D([0], [0], color=MEDIAN_COLOR, linewidth=2, label="Median"),
        Line2D([0], [0], color=MEAN_COLOR, marker="o", label="Mean"),
    ]

    for ax, data, labels, xlabel in [
        (ax1, data_by_bin, bin_labels, "Hour (UTC)"),
        (ax2, data_by_season, list(seasons.keys()), "Season"),
    ]:
        positions = list(range(len(labels)))
        bp = ax.boxplot(data, positions=positions, showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(BOX_COLOR)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color(MEDIAN_COLOR)
            median.set_linewidth(2)
        ax.plot(positions, [d.mean() for d in data], "o-", color=MEAN_COLOR)
        ax.set_xlabel(xlabel)
        ax.set_xticks(positions, labels)
        ax.grid(axis="y")

    ax1.set_ylabel("Max \u2212 Min CO$_2$ in ensemble (ppm)")
    ax2.tick_params(labelleft=False)
    ax2.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Boxplots of ensemble spread (max-min CO2) grouped by time of day and "
            "season."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
