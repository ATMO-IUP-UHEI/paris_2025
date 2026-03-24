from pathlib import Path

import ggpymanager as ggp
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors as mcolors
from tqdm import tqdm

from paris_2025.config import CONFIG
from paris_2025.plotting._loaders import (
    load_and_prepare_matching_data,
    load_matching_analysis_data,
)
from paris_2025.plotting.common import get_metadata


def plot_colormesh_of_loss(fig_path: str | Path):
    """Plot colormesh of matching loss for different loss types."""
    matching_loss = xr.open_dataset(
        Path(CONFIG["output_path"]) / ggp.config.MATCHING_LOSS_FILE_NAME
    )
    matching_loss.matching_loss.plot(col="loss_type", col_wrap=3)

    plt.savefig(
        fig_path,
        metadata=get_metadata("Colormesh of matching loss for different loss types."),
    )
    plt.clf()


def plot_matching_loss_distribution(fig_path: str | Path):
    """Plot distribution of best matching simulations."""
    matching_loss = xr.open_dataset(
        Path(CONFIG["output_path"]) / ggp.config.MATCHING_LOSS_FILE_NAME
    )

    n_sim_ids = 10
    ls = []
    for i in range(n_sim_ids):
        sim_ids = matching_loss["matching_loss"].idxmin("sim_id")
        ls.append(sim_ids.astype(int))
        matching_loss["matching_loss"].loc[dict(sim_id=sim_ids)] = np.nan
    sim_ids = xr.concat(ls, dim="ranking")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for i in range(len(axs)):
        for lt in matching_loss["loss_type"]:
            if i == 0:
                data = sim_ids.sel(
                    loss_type=lt, ranking=0
                ).values.flatten()  # type: ignore
                axs[i].set_title("Distribution of best matching simulations")
            else:
                data = sim_ids.sel(loss_type=lt).values.flatten()  # type: ignore
                axs[i].set_title(
                    f"Distribution of the {n_sim_ids} best " "matching simulations"
                )
            out = np.sort(np.bincount(data))
            not_s = (out == 0).sum()
            axs[i].plot(out, label=f"{lt.values} - {not_s} not used")
        axs[i].set_yscale("log")
        axs[i].set_xlabel("Number of simulations")
        axs[i].set_ylabel("Times selected")
        axs[i].legend()

    plt.savefig(
        fig_path,
        metadata=get_metadata("Distribution of best matching simulations."),
    )
    plt.clf()


def plot_n_stations_per_time(fig_path: str | Path):
    """Plot number of stations per time from matching loss."""
    matching_loss = xr.open_dataset(
        Path(CONFIG["output_path"]) / ggp.config.MATCHING_LOSS_FILE_NAME
    )

    plt.figure(figsize=(12, 6))
    matching_loss.n_stations_per_time.plot()
    plt.title("Number of stations per time")

    plt.savefig(
        fig_path,
        metadata=get_metadata("Number of stations per time from matching loss."),
    )
    plt.clf()


def generate_frequency_distribution(
    data: xr.DataArray, name: str, density: bool = False
) -> pd.Series:
    """Generate frequency distribution of values in the DataArray."""
    series = data.to_pandas().value_counts().sort_index()
    series.index = np.round(series.index, 1)
    series.name = name
    if density:
        series = series / series.sum()
    return series


def _calculate_difference(
    a: xr.DataArray, circular: bool = False, n_rankings: int = 9
) -> xr.DataArray:
    """Calculate difference between ranked simulations and the best simulation.

    Parameters
    ----------
    a : xr.DataArray
        Data array with 'best_sim_id' dimension
    circular : bool, optional
        If True, calculate circular difference (for angular data like wind
        direction). Default is False.
    n_rankings : int, optional
        Number of rankings to compare (excludes best). Default is 9
        (compares ranks 1-9 to rank 0).

    Returns
    -------
    xr.DataArray
        Difference array with best_sim_id dimension from 1 to n_rankings
    """
    if not circular:
        return a.sel(best_sim_id=range(1, n_rankings + 1)) - a.sel(best_sim_id=0)
    else:
        # Circular difference for wind direction
        diff = a.sel(best_sim_id=range(1, n_rankings + 1)) - a.sel(best_sim_id=0)
        diff = (diff + 180) % 360 - 180
        return diff


def plot_selected_meteo_conditions_by_variable(
    fig_path: str | Path,
    variable: str = "synoptic_wind_speed",
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    source_groups_path: str | Path = Path(CONFIG["source_groups_path"]),
    n_sim_ids: int = 10,
):
    """Plot which meteorological conditions are selected grouped by variable.

    Creates a multi-panel figure showing frequency distributions comparing all catalog
    entries with selected entries for different loss types.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    variable : str, optional
        Meteorological variable to group by. Options: "synoptic_wind_speed",
        "synoptic_wind_direction", "stab_class"
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    source_groups_path : str | Path, optional
        Path to source groups data
    n_sim_ids : int, optional
        Number of top simulation IDs to consider
    """
    (
        _,
        gral_meteo,
        matching_loss,
        _,
        sim_ids,
    ) = load_and_prepare_matching_data(
        gral_concentration_path, matching_loss_path, source_groups_path, n_sim_ids
    )

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for lt, ax in zip(matching_loss["loss_type"], axs.flatten()):
        catalog = []
        selected = []
        for ranking in [[0], range(n_sim_ids)]:
            catalog.append(
                generate_frequency_distribution(
                    gral_meteo[variable], name=f"rank_{ranking}"
                )
            )
            selected.append(
                generate_frequency_distribution(
                    gral_meteo[variable].sel(
                        sim_id=np.unique(sim_ids.sel(loss_type=lt, ranking=ranking))
                    ),
                    name=f"selected_rank_{ranking}",
                )
            )
        pd.concat(catalog, axis="columns").plot(
            kind="bar",
            ax=ax,
            alpha=1,
            color="#BCD5E8",
            label="All catalog entries",
            legend=False,
        )
        pd.concat(selected, axis="columns").plot(
            kind="bar",
            ax=ax,
            alpha=1,
            color=("#5A829F", "#CD933C"),
            label="Selected entries",
            legend=False,
        )
        ax.set_title(f"{lt.values}")

    fig.suptitle(f"Which meteorological conditions are selected grouped by {variable}?")

    # Add legend
    labels = ["All catalog entries", "Selected entries", f"Top {n_sim_ids}"]
    handles = [
        mpatches.Patch(color="#BCD5E8", label="All catalog entries"),
        mpatches.Patch(color="#5A829F", label="Selected entries"),
        mpatches.Patch(color="#CD933C", label=f"Top {n_sim_ids}"),
    ]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Frequency distribution of selected meteorological conditions "
            f"grouped by {variable}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_meteo_selection_frequency_by_variable(
    fig_path: str | Path,
    variable: str = "synoptic_wind_speed",
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    source_groups_path: str | Path = Path(CONFIG["source_groups_path"]),
    n_sim_ids: int = 10,
):
    """Plot how often meteorological conditions are selected grouped by variable.

    Creates a multi-panel figure showing density distributions of how frequently
    different meteorological conditions are selected for different loss types.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    variable : str, optional
        Meteorological variable to group by
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    source_groups_path : str | Path, optional
        Path to source groups data
    n_sim_ids : int, optional
        Number of top simulation IDs to consider
    """
    (
        _,
        gral_meteo,
        matching_loss,
        _,
        sim_ids,
    ) = load_and_prepare_matching_data(
        gral_concentration_path, matching_loss_path, source_groups_path, n_sim_ids
    )

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for lt, ax in zip(matching_loss["loss_type"], axs.flatten()):
        selected = []
        for ranking in [[0], range(n_sim_ids)]:
            selected.append(
                generate_frequency_distribution(
                    gral_meteo[variable]
                    .sel(sim_id=sim_ids.sel(loss_type=lt, ranking=ranking))
                    .stack(z_=("ranking", "time")),
                    name="Selected entries",
                    density=True,
                )
            )
        pd.concat(selected, axis="columns").plot(
            kind="bar",
            ax=ax,
            alpha=1,
            color=("#5A829F", "#CD933C"),
            label="Selected entries",
        )
        ax.set_title(f"{lt.values}")

    fig.suptitle(
        f"How often are meteorological conditions selected grouped by {variable}?"
    )

    # Add legend
    labels = ["Selected entries", f"Top {n_sim_ids}"]
    handles = [
        mpatches.Patch(color="#5A829F", label="Selected entries"),
        mpatches.Patch(color="#CD933C", label=f"Top {n_sim_ids}"),
    ]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Density distribution of meteorological condition selection frequency "
            f"grouped by {variable}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_co2_concentration_violin_by_loss_type(
    fig_path: str | Path,
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    source_groups_path: str | Path = Path(CONFIG["source_groups_path"]),
    n_sim_ids: int = 10,
):
    """Plot violin plots of CO2 concentration distributions by loss type.

    Creates violin plots showing the distribution of mean CO2 concentrations for
    best matching simulations across different loss types.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    source_groups_path : str | Path, optional
        Path to source groups data
    n_sim_ids : int, optional
        Number of top simulation IDs to consider
    """
    gral_concentration, _, matching_loss, source_groups, sim_ids = (
        load_and_prepare_matching_data(
            gral_concentration_path,
            matching_loss_path,
            source_groups_path,
            n_sim_ids,
        )
    )

    origins_earth_source_group_mask = (
        source_groups["type"].str.contains("Origins.earth").compute()
    )

    con_data = (
        gral_concentration["concentration"]
        .mean("station", skipna=True)
        .sel(source_group=origins_earth_source_group_mask)
        .sum("source_group")
    )

    fig = plt.figure(figsize=(10, 6))
    plt.violinplot(
        [
            con_data.sel(sim_id=sim_ids.sel(loss_type=lt, ranking=0))
            for lt in matching_loss["loss_type"]
        ],
        showmeans=True,
        showextrema=True,
        side="both",
    )

    # Add mean values below each violin
    for i, lt in enumerate(matching_loss["loss_type"]):
        data = con_data.sel(sim_id=sim_ids.sel(loss_type=lt, ranking=0))
        mean_val = data.mean().values
        plt.text(i + 1, 1, f"Mean:\n{mean_val:.2f}", ha="center", va="top", fontsize=9)

    plt.ylim(0, 10)
    plt.xticks(
        np.arange(len(matching_loss["loss_type"])) + 1,
        matching_loss["loss_type"].values.tolist(),
        rotation=90,
    )
    plt.xlabel("Loss type")
    plt.ylabel("Mean CO2 concentration [ppm]")
    plt.title("Distribution of mean CO2 concentration for best matching simulations")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Violin plots of mean CO2 concentration distributions by loss type."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_co2_distribution_by_meteo_variable(
    fig_path: str | Path,
    variable: str = "synoptic_wind_speed",
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    source_groups_path: str | Path = Path(CONFIG["source_groups_path"]),
    n_sim_ids: int = 10,
):
    """Plot CO2 concentration distributions grouped by meteorological variable.

    Creates split violin plots showing CO2 distributions for top 1 vs top N
    simulations, grouped by meteorological conditions.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    variable : str, optional
        Meteorological variable to group by
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    source_groups_path : str | Path, optional
        Path to source groups data
    n_sim_ids : int, optional
        Number of top simulation IDs to consider
    """
    (
        gral_concentration,
        gral_meteo,
        matching_loss,
        source_groups,
        sim_ids,
    ) = load_and_prepare_matching_data(
        gral_concentration_path,
        matching_loss_path,
        source_groups_path,
        n_sim_ids,
    )

    origins_earth_source_group_mask = (
        source_groups["type"].str.contains("Origins.earth").compute()
    )

    con_data = (
        gral_concentration["concentration"]
        .mean("station", skipna=True)
        .sel(source_group=origins_earth_source_group_mask)
        .sum("source_group")
    )

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for lt, ax in zip(matching_loss["loss_type"], axs.flatten()):
        for ranking, side in zip([[0], range(n_sim_ids)], ["low", "high"]):
            con_subset = con_data.sel(sim_id=sim_ids.sel(loss_type=lt, ranking=ranking))
            gral_meteo_subset = (
                gral_meteo[variable]
                .sel(sim_id=sim_ids.sel(loss_type=lt, ranking=ranking))
                .compute()
            )

            parts = ax.violinplot(
                [g for _, g in con_subset.groupby(gral_meteo_subset)],
                showmeans=True,
                showextrema=False,
                side=side,
            )

            # Change color of the violin parts
            color = "#5A829F" if side == "low" else "#CD933C"
            for partname in ("cmins", "cmaxes", "cmeans"):
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor(color)

            # Change color of the violin bodies
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

        ax.set_title(f"{lt.values}")

    fig.suptitle(f"What are the CO2 distributions grouped by {variable}?")
    axs[0, 0].set_ylabel("Mean CO2 concentration [ppm]")
    axs[1, 0].set_ylabel("Mean CO2 concentration [ppm]")

    # Add legend
    labels = ["Selected entries", f"Top {n_sim_ids}"]
    handles = [
        mpatches.Patch(color="#5A829F", label="Selected entries"),
        mpatches.Patch(color="#CD933C", label=f"Top {n_sim_ids}"),
    ]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"CO2 concentration distributions grouped by {variable}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_matching_loss_by_meteo_variable(
    fig_path: str | Path,
    variable: str = "synoptic_wind_speed",
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    source_groups_path: str | Path = Path(CONFIG["source_groups_path"]),
    n_sim_ids: int = 10,
):
    """Plot matching loss distributions grouped by meteorological variable.

    Creates split violin plots showing matching loss distributions for top 1 vs top N
    simulations, grouped by meteorological conditions.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    variable : str, optional
        Meteorological variable to group by
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    source_groups_path : str | Path, optional
        Path to source groups data
    n_sim_ids : int, optional
        Number of top simulation IDs to consider
    """
    (
        _,
        gral_meteo,
        matching_loss,
        _,
        sim_ids,
    ) = load_and_prepare_matching_data(
        gral_concentration_path, matching_loss_path, source_groups_path, n_sim_ids
    )
    gral_meteo.load()
    matching_loss.load()
    sim_ids.load()

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for lt, ax in tqdm(zip(matching_loss["loss_type"], axs.flatten())):
        for ranking, side in zip([[0], range(n_sim_ids)], ["low", "high"]):
            gral_meteo_subset = (
                gral_meteo[variable]
                .sel(sim_id=sim_ids.sel(loss_type=lt, ranking=ranking))
                .compute()
            )
            loss_subset = matching_loss["matching_loss"].sel(
                loss_type=lt, sim_id=sim_ids.sel(loss_type=lt, ranking=ranking)
            )

            parts = ax.violinplot(
                [g for _, g in loss_subset.groupby(gral_meteo_subset)],
                showmeans=True,
                showextrema=False,
                side=side,
            )

            # Change color of the violin parts
            color = "#5A829F" if side == "low" else "#CD933C"
            for partname in ("cmins", "cmaxes", "cmeans"):
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor(color)

            # Change color of the violin bodies
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

        ax.set_title(f"{lt.values}")

    fig.suptitle(f"What are the matching loss distributions grouped by {variable}?")
    axs[0, 0].set_ylabel("Matching loss")
    axs[1, 0].set_ylabel("Matching loss")

    # Add legend
    labels = ["Selected entries", f"Top {n_sim_ids}"]
    handles = [
        mpatches.Patch(color="#5A829F", label="Selected entries"),
        mpatches.Patch(color="#CD933C", label=f"Top {n_sim_ids}"),
    ]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.savefig(
        fig_path,
        metadata=get_metadata(f"Matching loss distributions grouped by {variable}."),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_concentration_vs_meteo_differences(
    fig_path: str | Path,
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    loss_type: str = "rmse - filter: True",
    n_best: int = 25,
):
    """Plot hexbin of concentration diffs vs meteorological variable diffs.

    Creates a 3-panel figure showing how concentration differences correlate with
    differences in wind speed, wind direction, and stability class when comparing
    alternative matching simulations to the best matching simulation.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    loss_type : str, optional
        Type of loss metric to use for selecting best simulations
    n_best : int, optional
        Number of best simulations to analyze
    """
    concentration, loss, speed, direction, stab_class = load_matching_analysis_data(
        gral_concentration_path, matching_loss_path, loss_type, n_best
    )

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Concentration vs wind speed
    axs[0].hexbin(
        _calculate_difference(concentration).values.flatten(),
        _calculate_difference(speed).values.flatten(),
        gridsize=30,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    axs[0].set_xlabel("Concentration difference to best simulation [ppm]")
    axs[0].set_ylabel("Wind speed difference [m/s]")
    axs[0].set_title("Concentration vs Wind Speed")

    # Concentration vs wind direction
    axs[1].hexbin(
        _calculate_difference(concentration).values.flatten(),
        _calculate_difference(direction, circular=True).values.flatten(),
        gridsize=30,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    axs[1].set_xlabel("Concentration difference to best simulation [ppm]")
    axs[1].set_ylabel("Wind direction difference [°]")
    axs[1].set_title("Concentration vs Wind Direction")

    # Concentration vs stability class
    axs[2].hexbin(
        _calculate_difference(concentration).values.flatten(),
        _calculate_difference(stab_class).values.flatten(),
        gridsize=30,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    axs[2].set_xlabel("Concentration difference to best simulation [ppm]")
    axs[2].set_ylabel("Stability class difference [-]")
    axs[2].set_title("Concentration vs Stability Class")

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Hexbin plots comparing concentration differences vs meteorological "
            "variable differences for alternative matching simulations."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_loss_vs_max_loss_difference(
    fig_path: str | Path,
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    loss_type: str = "rmse - filter: True",
    n_best: int = 25,
):
    """Plot hexbin of best simulation loss vs maximum loss difference.

    Shows how the loss of the best matching simulation relates to the spread
    in loss values across alternative matching simulations.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    loss_type : str, optional
        Type of loss metric to use
    n_best : int, optional
        Number of best simulations to analyze
    """
    _, loss, _, _, _ = load_matching_analysis_data(
        gral_concentration_path, matching_loss_path, loss_type, n_best
    )
    loss.load()

    fig = plt.figure(figsize=(8, 6))
    plt.hexbin(
        loss.sel(best_sim_id=0),
        _calculate_difference(loss).max("best_sim_id"),
        gridsize=30,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(label="Log density")
    plt.xlabel("Loss of best simulation [-]")
    plt.ylabel("Max loss difference to best simulation [-]")
    plt.title("Loss of Best Simulation vs Max Loss Difference")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Hexbin plot of best simulation loss vs maximum loss difference."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_loss_difference_vs_concentration_difference(
    fig_path: str | Path,
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    loss_type: str = "rmse - filter: True",
    n_best: int = 25,
):
    """Plot hexbin of max loss difference vs max concentration difference.

    Shows how the spread in matching loss relates to the spread in predicted
    concentrations.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    loss_type : str, optional
        Type of loss metric to use
    n_best : int, optional
        Number of best simulations to analyze
    """
    concentration, loss, _, _, _ = load_matching_analysis_data(
        gral_concentration_path, matching_loss_path, loss_type, n_best
    )
    concentration.load()
    loss.load()

    fig = plt.figure(figsize=(8, 6))
    plt.hexbin(
        _calculate_difference(loss).max("best_sim_id"),
        _calculate_difference(abs(concentration)).max("best_sim_id"),
        gridsize=30,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(label="Log density")
    plt.xlabel("Max loss difference to best simulation [-]")
    plt.ylabel("Max concentration difference to best simulation [ppm]")
    plt.title("Loss Difference vs Concentration Difference")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Hexbin plot of max loss difference vs max concentration difference."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_loss_vs_max_concentration_difference(
    fig_path: str | Path,
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    loss_type: str = "rmse - filter: True",
    n_best: int = 25,
):
    """Plot hexbin of best simulation loss vs max concentration difference.

    Shows how the loss of the best matching simulation relates to the spread
    in predicted concentrations across alternative simulations.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    loss_type : str, optional
        Type of loss metric to use
    n_best : int, optional
        Number of best simulations to analyze
    """
    concentration, loss, _, _, _ = load_matching_analysis_data(
        gral_concentration_path, matching_loss_path, loss_type, n_best
    )

    concentration.load()
    loss.load()

    fig = plt.figure(figsize=(8, 6))
    plt.hexbin(
        loss.sel(best_sim_id=0),
        abs(_calculate_difference(concentration)).max("best_sim_id"),
        gridsize=30,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(label="Log density")
    plt.xlabel("Loss of best simulation [-]")
    plt.ylabel("Max concentration difference to best simulation [ppm]")
    plt.title("Best Simulation Loss vs Max Concentration Difference")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Hexbin plot of best simulation loss vs max concentration difference."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_concentration_vs_max_concentration_difference(
    fig_path: str | Path,
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    loss_type: str = "rmse - filter: True",
    n_best: int = 25,
):
    """Plot hexbin of best simulation concentration vs max conc difference.

    Shows how the concentration predicted by the best simulation relates to
    the sensitivity of concentration to choice of matching simulation.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    loss_type : str, optional
        Type of loss metric to use
    n_best : int, optional
        Number of best simulations to analyze
    """
    concentration, _, _, _, _ = load_matching_analysis_data(
        gral_concentration_path, matching_loss_path, loss_type, n_best
    )

    concentration.load()

    fig = plt.figure(figsize=(8, 6))
    plt.hexbin(
        concentration.sel(best_sim_id=0),
        abs(_calculate_difference(concentration)).max("best_sim_id"),
        gridsize=30,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(label="Log density")
    plt.xlabel("Concentration of best simulation [ppm]")
    plt.ylabel("Max concentration difference to best simulation [ppm]")
    plt.title("Best Simulation Concentration vs Max Concentration Difference")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Hexbin plot of best simulation concentration vs max "
            "concentration difference."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_wind_speed_vs_max_concentration_difference(
    fig_path: str | Path,
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    loss_type: str = "rmse - filter: True",
    n_best: int = 25,
):
    """Plot hexbin of best simulation wind speed vs max concentration difference.

    Shows how wind speed in the best matching simulation relates to the
    sensitivity of concentration predictions to choice of matching simulation.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    loss_type : str, optional
        Type of loss metric to use
    n_best : int, optional
        Number of best simulations to analyze
    """
    concentration, _, speed, _, _ = load_matching_analysis_data(
        gral_concentration_path, matching_loss_path, loss_type, n_best
    )
    concentration.load()
    speed.load()

    fig = plt.figure(figsize=(8, 6))
    plt.hexbin(
        speed.sel(best_sim_id=0),
        abs(_calculate_difference(concentration)).max("best_sim_id"),
        gridsize=30,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(label="Log density")
    plt.xlabel("Wind speed of best simulation [m/s]")
    plt.ylabel("Max concentration difference to best simulation [ppm]")
    plt.title("Wind Speed vs Max Concentration Difference")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Hexbin plot of wind speed vs max concentration difference."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_wind_direction_vs_max_concentration_difference(
    fig_path: str | Path,
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    loss_type: str = "rmse - filter: True",
    n_best: int = 25,
):
    """Plot hexbin of best simulation wind direction vs max concentration difference.

    Shows how wind direction in the best matching simulation relates to the
    sensitivity of concentration predictions to choice of matching simulation.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    loss_type : str, optional
        Type of loss metric to use
    n_best : int, optional
        Number of best simulations to analyze
    """
    concentration, _, _, direction, _ = load_matching_analysis_data(
        gral_concentration_path, matching_loss_path, loss_type, n_best
    )

    concentration.load()
    direction.load()

    fig = plt.figure(figsize=(8, 6))
    plt.hexbin(
        direction.sel(best_sim_id=0),
        abs(_calculate_difference(concentration)).max("best_sim_id"),
        gridsize=30,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(label="Log density")
    plt.xlabel("Wind direction of best simulation [°]")
    plt.ylabel("Max concentration difference to best simulation [ppm]")
    plt.title("Wind Direction vs Max Concentration Difference")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Hexbin plot of wind direction vs max concentration difference."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_stability_class_vs_max_concentration_difference(
    fig_path: str | Path,
    gral_concentration_path: str | Path = Path(CONFIG["gral_co2_path"]) / "co2.nc",
    matching_loss_path: str | Path = Path(CONFIG["output_path"])
    / ggp.config.MATCHING_LOSS_FILE_NAME,
    loss_type: str = "rmse - filter: True",
    n_best: int = 25,
):
    """Plot hexbin of best simulation stability class vs max concentration difference.

    Shows how atmospheric stability class in the best matching simulation relates
    to the sensitivity of concentration predictions to choice of matching simulation.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    gral_concentration_path : str | Path, optional
        Path to GRAL concentration data
    matching_loss_path : str | Path, optional
        Path to matching loss data
    loss_type : str, optional
        Type of loss metric to use
    n_best : int, optional
        Number of best simulations to analyze
    """
    concentration, _, _, _, stab_class = load_matching_analysis_data(
        gral_concentration_path, matching_loss_path, loss_type, n_best
    )

    concentration.load()
    stab_class.load()

    fig = plt.figure(figsize=(8, 6))
    plt.hexbin(
        stab_class.sel(best_sim_id=0),
        abs(_calculate_difference(concentration)).max("best_sim_id"),
        gridsize=30,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    plt.colorbar(label="Log density")
    plt.xlabel("Stability class of best simulation [-]")
    plt.ylabel("Max concentration difference to best simulation [ppm]")
    plt.title("Stability Class vs Max Concentration Difference")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Hexbin plot of stability class vs max concentration difference."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_loss_vs_co2_spread_distribution(
    fig_path: str | Path,
):

    model_co2 = ggp.load("concentration_timeseries", config=CONFIG)
    da = (
        model_co2.sel(
            loss_type="rmse - filter: True",
            type=model_co2.type.str.contains("Origins.earth|VPRM"),
        )
        .co2_timeseries.sum("type")
        .compute()
    )
    co2_spread = abs(da.isel(best_sim_id=range(1, 5)) - da.isel(best_sim_id=0)).max(
        "best_sim_id"
    )
    loss_spread = (
        da.loss_diff.isel(best_sim_id=range(1, 5)) - da.loss_diff.isel(best_sim_id=0)
    ).max("best_sim_id")
    loss_spread_aligned = loss_spread.broadcast_like(co2_spread)

    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 8], wspace=0.02)
    ax_hist = fig.add_subplot(gs[0])
    ax_2d = fig.add_subplot(gs[1], sharey=ax_hist)

    threshold_color = "#565656"

    # Histogram for RMSE spread < 0.1
    mask = loss_spread_aligned.values.flatten() < 0.1
    ax_hist.hist(
        co2_spread.values.flatten()[mask],
        bins=100,
        orientation="horizontal",
        color=threshold_color,
        edgecolor="none",
        alpha=1.0,
        label="Only for RMSE diff < 0.1 m/s",
    )
    ax_hist.set_ylabel("CO$_2$ spread [ppm]")
    ax_hist.set_xlabel("Count")
    ax_hist.invert_xaxis()
    ax_hist.legend(loc="upper left", bbox_to_anchor=(-0.5, 1.1), frameon=False)

    h = ax_2d.hist2d(
        loss_spread_aligned.values.flatten(),
        co2_spread.values.flatten(),
        bins=100,
        norm=mcolors.LogNorm(),
        cmap="flare",
    )
    plt.colorbar(h[3], ax=ax_2d, label="Density")

    # Box around the region < 0.1
    # Use double linewidth and clipping to ensure the border is drawn inside the box
    rect = patches.Rectangle(
        (0, 0),
        0.1,
        100,
        linewidth=5,
        # linestyle=":",
        edgecolor=threshold_color,
        facecolor="none",
        zorder=10,
        label="RMSE difference threshold",
    )
    ax_2d.add_patch(rect)

    # Clip the rectangle to its own shape to create an inner border
    clip_rect = patches.Rectangle((0, 0), 0.1, 100, transform=ax_2d.transData)
    rect.set_clip_path(clip_rect)

    ax_2d.set_ylim(0, 100)
    ax_2d.set_xlabel("RMSE difference in wind vector [m/s]")
    ax_2d.tick_params(axis="y", left=False, labelleft=False)

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "2D histogram of RMSE loss difference vs CO2 concentration spread for the "
            "best 5 entries."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
