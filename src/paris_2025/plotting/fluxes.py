import calendar
import logging
from pathlib import Path

import ggpymanager as ggp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

import paris_2025 as p
from paris_2025.config import CONFIG
from paris_2025.plotting._loaders import load_flux_maps_data
from paris_2025.plotting.common import get_metadata


def plot_flux_by_type(fig_path: str | Path):
    """Plot horizontal bar chart of flux by source type."""
    source_group_ds = xr.open_dataset(p.model_input.fluxes.SOURCE_GROUP_NETCDF_PATH)
    flux_by_type = source_group_ds.source_flux.load().groupby("type").sum().compute()

    fig, ax = plt.subplots(figsize=(10, 6))
    flux_by_type.to_pandas().plot.barh(ax=ax)
    ax.set_xlabel(
        f"{source_group_ds.source_flux.attrs['long_name']} "
        f"[{source_group_ds.source_flux.attrs['units']}]"
    )
    ax.set_title("Total Flux by Source Type")

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata("Total flux by source type."),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_temporal_scaling_factors(fig_path: str | Path):
    """Plot temporal scaling factors over time."""
    temporal_factor = xr.open_dataset(
        p.model_input.fluxes.TEMPORAL_PROFILES_NETCDF_PATH
    )

    temporal_factor.temporal.plot()
    plt.xlabel("Time")
    plt.ylabel("Temporal scaling factor")
    plt.title("Temporal Scaling Factors by Type")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        fig_path,
        metadata=get_metadata("Temporal scaling factors over time by type."),
        bbox_inches="tight",
    )
    plt.close()


def plot_temporal_scaling_vprm(fig_path: str | Path):
    from paris_2025.plotting import INVENTORY_COLORS

    # VPRM temporal profiles
    temporal_profiles = ggp.load("temporal_profiles", p.CONFIG)
    for t in ["VPRM R", "VPRM GEE"]:
        profile = temporal_profiles.sel(
            type=temporal_profiles.type.str.contains(t.replace(" ", ".*"))
        ).temporal
        color = INVENTORY_COLORS[t]
        profile.plot(hue="type", color=color, alpha=0.5, add_legend=False)
        profile.resample(time="1D").mean().plot(
            hue="type",
            color=color,
            add_legend=False,
            linewidth=2,
        )

    # Create a custom legend
    # labels = ["R hourly", "R daily mean", "GEE hourly", "GEE daily mean"]
    handles = [
        Line2D([0], [0], linewidth=lw, color=c, alpha=alpha, label=l)
        for lw, c, alpha, l in zip(
            [1, 2, 1, 2],
            [
                INVENTORY_COLORS["VPRM R"],
                INVENTORY_COLORS["VPRM R"],
                INVENTORY_COLORS["VPRM GEE"],
                INVENTORY_COLORS["VPRM GEE"],
            ],
            [0.5, 1, 0.5, 1],
            ["R hourly", "R daily mean", "GEE hourly", "GEE daily mean"],
        )
    ]
    plt.legend(handles=handles, title="VPRM Temporal Profiles", loc="lower right")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "VPRM temporal scaling profiles (hourly and daily mean)."
        ),
        bbox_inches="tight",
    )
    plt.close()


def plot_temporal_scaling_factor_cycles(fig_path: str | Path):
    """Plot diurnal, weekly, and annual temporal scaling factor cycles by sector.

    Creates a figure with two rows (one per inventory: TNO, Origins.earth) and
    three columns (diurnal, weekly, annual cycle). Panel headers use the same
    FancyBboxPatch style as other station plots in this package.

    Parameters
    ----------
    fig_path : str | Path
        Destination path for the saved figure.
    """
    from paris_2025.plotting import INVENTORY_COLORS, INVENTORY_SECTORS

    temporal_factor = xr.open_dataset(
        p.model_input.fluxes.TEMPORAL_PROFILES_NETCDF_PATH
    )["temporal"].sortby("type")

    inventories = ["TNO 2018", "Origins.earth 2023"]

    groupby_configs = [
        {
            "key": "time.hour",
            "dim": "hour",
            "title": "Diurnal cycle",
            "xlabel": "Hour of day",
            "xticks": np.arange(0, 24, 6),
            "xtick_labels": None,
        },
        {
            "key": "time.dayofweek",
            "dim": "dayofweek",
            "title": "Weekly cycle",
            "xlabel": "Day of week",
            "xticks": np.arange(0, 7),
            "xtick_labels": list(calendar.day_abbr),
        },
        {
            "key": "time.month",
            "dim": "month",
            "title": "Annual cycle",
            "xlabel": "Month",
            "xticks": np.arange(1, 13),
            "xtick_labels": list(calendar.month_abbr)[1:],
        },
    ]

    n_rows = len(inventories)
    n_cols = len(groupby_configs)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        # figsize=(14, 4 * n_rows),
        sharey="row",
        gridspec_kw={"hspace": 0.35, "wspace": 0.15},
    )

    offwhite = "#F8F8FF"

    for row, inventory in enumerate(inventories):
        tf = temporal_factor.sel(type=temporal_factor.type.str.contains(inventory))
        lw = axs[row, 0].spines["left"].get_linewidth()

        for col, cfg in enumerate(groupby_configs):
            ax = axs[row, col]
            grouped = tf.groupby(cfg["key"]).mean()  # type: ignore[call-overload]
            dim = cfg["dim"]

            for t in grouped.type.values:
                if t not in INVENTORY_SECTORS:
                    logging.warning(
                        f"Type '{t}' not found in INVENTORY_SECTORS. Skipping this "
                        f"type for plotting."
                    )
                    continue
                label = INVENTORY_SECTORS[str(t)]
                color = INVENTORY_COLORS[str(t)]
                ax.plot(
                    grouped.coords[dim], grouped.sel(type=t), label=label, color=color
                )

            ax.set_xlabel(cfg["xlabel"])
            ax.grid(alpha=0.3)
            ax.set_xticks(cfg["xticks"])
            if cfg["xtick_labels"] is not None:
                ax.set_xticklabels(cfg["xtick_labels"], rotation=45, ha="right")

            # Panel title box (top row only to avoid duplication)
            fw = ax.xaxis.label.get_fontweight()
            if row == 0:
                ax.text(
                    0.5,
                    1.06,
                    cfg["title"],
                    transform=ax.transAxes,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight=fw,
                )
                box = FancyBboxPatch(
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
                fig.patches.append(box)

        axs[row, 0].set_ylabel(f"{inventory}\nTemporal scaling factor")
        handles, labels = axs[row, -1].get_legend_handles_labels()
        axs[row, -1].legend(
            handles,
            labels,
            bbox_to_anchor=(1.05, 1.0),
            loc="upper left",
            title="Sector",
        )

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Diurnal, weekly and annual temporal scaling factors "
            "by sector and inventory."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_flux_maps(
    fig_path: str | Path,
    cadastre_path: str | Path = Path(CONFIG["domain"]["gral"]["conf_path"])
    / "cadastre.dat",
    source_groups_path: str | Path = p.model_input.fluxes.SOURCE_GROUP_NETCDF_PATH,
    point_path: str | Path = Path(CONFIG["domain"]["gral"]["conf_path"]) / "point.dat",
):
    """Plot spatial distribution of fluxes.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    cadastre_path : str | Path, optional
        Path to the GRAL cadastre.dat file
    source_groups_path : str | Path, optional
        Path to the source groups NetCDF file
    point_path : str | Path, optional
        Path to the GRAL point.dat file
    """
    cadastre_emissions, source_groups, point_da, GRAL = load_flux_maps_data(
        cadastre_path, source_groups_path, point_path
    )

    mask = xr.zeros_like(cadastre_emissions.type)
    inventories = [
        "Origins.earth",
        "TNO",
        "VPRM 2023 GEE",
        "VPRM 2023 R",
        "VPRM 2024 GEE",
        "VPRM 2024 R",
    ]
    for inventory in inventories:
        mask[cadastre_emissions.type.str.contains(inventory)] = inventory
    maps = cadastre_emissions.groupby(mask, restore_coord_dims=True).sum()
    maps["x"] = cadastre_emissions["x"]
    maps["y"] = cadastre_emissions["y"]
    vprm_gee = maps.sel(type=["VPRM 2023 GEE", "VPRM 2024 GEE"]).mean("type")
    vprm_r = maps.sel(type=["VPRM 2023 R", "VPRM 2024 R"]).mean("type")

    # Drop the VPRM GEE and R maps to avoid plotting them twice
    maps = maps.sel(type=~maps.type.str.contains("VPRM"))
    # Add the averaged VPRM GEE and R maps back to the dataset
    vprm_gee = vprm_gee.expand_dims(type=["VPRM GEE"])
    vprm_r = vprm_r.expand_dims(type=["VPRM R"])
    maps = xr.concat([maps, vprm_gee, vprm_r], dim="type")

    # Convert from kg/h to kg/y/m^2
    maps = maps / (GRAL["dx"] * GRAL["dy"]) * 365 * 24
    # Convert from kg/h to kg/year
    point_da *= 365 * 24

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 5), dpi=300)
    point_size = 1e7

    def point_norm(x):
        return np.sqrt(x)

    pcfs = {}
    for ax, inv, id in zip(
        axs.flatten(), maps.type.values, ["(a)", "(b)", "(c)", "(d)"]
    ):
        if inv in ["Origins.earth", "TNO"]:
            vmin = 0
            vmax = 100
            cmap = "viridis"
            scaling = 1

            inv_point_da = point_da.where(point_da.type.str.contains(inv), drop=True)
            if len(inv_point_da.index) > 0:
                ax.scatter(
                    x=inv_point_da["x"].values,
                    y=inv_point_da["y"].values,
                    zorder=10,
                    color="red",
                    edgecolor="k",
                    linewidth=0.5,
                    s=inv_point_da.values / point_size,
                    label="Point sources",
                )
        else:
            vmin = -10
            vmax = 10
            cmap = "PiYG_r"
            scaling = -1 if "GEE" in inv else 1

        pcf = (maps.sel(type=inv) * scaling).plot(
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
            add_colorbar=False,
        )  # type: ignore
        pcfs[inv] = pcf
        ax.set_aspect("equal")
        ax.set_title(inv)
        ax.set_title(id, loc="left")
        # Turn ticks off
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        if id == "(b)":
            # Create legend for scatter
            lower_point_emissions = 5e7
            upper_point_emissions = lower_point_emissions * 10
            handles = [
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    color="red",
                    marker="o",
                    markeredgecolor="k",
                    markersize=point_norm(e / point_size),
                    markeredgewidth=0.5,
                )
                for e in [lower_point_emissions, upper_point_emissions]
            ]
            labels = [
                f"{e/1e6:.0f} kt/yr"
                for e in [lower_point_emissions, upper_point_emissions]
            ]
            ax.legend(handles, labels, loc="lower left")

    fig.subplots_adjust(wspace=0.02)
    plt.colorbar(pcfs["TNO"], ax=axs[0], label="Annual CO$_2$ flux (kg/m$^2$)")
    plt.colorbar(pcfs["VPRM R"], ax=axs[1], label="Annual CO$_2$ flux (kg/m$^2$)")

    plt.savefig(
        fig_path,
        metadata=get_metadata("Spatial distribution of CO2 fluxes by inventory."),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_total_flux_by_inventory(
    fig_path: str | Path,
    cadastre_path: str | Path = Path(CONFIG["domain"]["gral"]["conf_path"])
    / "cadastre.dat",
    source_groups_path: str | Path = p.model_input.fluxes.SOURCE_GROUP_NETCDF_PATH,
    point_path: str | Path = Path(CONFIG["domain"]["gral"]["conf_path"]) / "point.dat",
):
    """Plot stacked bar chart of total CO2 flux by inventory and source type.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    cadastre_path : str | Path, optional
        Path to the GRAL cadastre.dat file
    source_groups_path : str | Path, optional
        Path to the source groups NetCDF file
    point_path : str | Path, optional
        Path to the GRAL point.dat file
    """

    cadastre_emissions, source_groups, point_da, GRAL = load_flux_maps_data(
        cadastre_path, source_groups_path, point_path
    )

    point_da["z"] = point_da.z.astype(float)
    if point_da.z.isnull().any():
        # logging.warning(
        #     "The points contain missing values in the z coordinate. All points with "
        #     "missing z values will be dropped for plotting. This may lead to missing "
        #     "points in the plots."
        # )
        # total_emissions = point_da.sum().item()
        # isnull = point_da.z.isnull()
        # point_da = point_da.sel(index=~isnull)
        # dropped_emissions = total_emissions - point_da.sum().item()
        # logging.warning(
        #     f"Dropped {dropped_emissions:.2f} kg/year of emissions from "
        #     f"{isnull.sum().item()} points."
        # )
        logging.warning("The points contain missing values in the z coordinate.")

    area_df = cadastre_emissions.sum(["x", "y"]).groupby("type").sum().to_pandas()
    point_df = point_da.groupby("type").sum().to_pandas()

    from . import INVENTORY_SECTORS

    xticklabels = INVENTORY_SECTORS

    gral_fluxes = pd.DataFrame({"area": area_df, "point": point_df})

    # Take mean of VPRM 2023/2024
    gral_fluxes.loc["VPRM GEE"] = gral_fluxes.loc[
        ["VPRM 2023 GEE", "VPRM 2024 GEE"]
    ].mean()
    gral_fluxes = gral_fluxes.drop(["VPRM 2023 GEE", "VPRM 2024 GEE"])
    gral_fluxes.loc["VPRM R"] = gral_fluxes.loc[["VPRM 2023 R", "VPRM 2024 R"]].mean()
    gral_fluxes = gral_fluxes.drop(["VPRM 2023 R", "VPRM 2024 R"])

    # Negative sign for GEE (uptake)
    gral_fluxes.loc[gral_fluxes.index.str.contains("GEE")] *= -1

    # Convert from kg/h to kt/year
    gral_fluxes *= 365 * 24 / 1e6

    fig, ax = plt.subplots()
    gral_fluxes.plot.bar(stacked=True, edgecolor="k", ax=ax)

    bars = ax.patches
    for bar in bars[len(bars) // 2 :]:
        bar.set_hatch("///")
        bar.set_alpha(0.5)

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylabel("CO$_2$ Fluxes (kt/year)")

    i = 0
    groups = ["Origins.earth", "TNO", "VPRM"]
    colors = ["#5e32a9ff", "#C96ACE", "#527d08"]

    for group, color in zip(groups, colors):
        contains_group_identifier = gral_fluxes.index.str.contains(group)
        total_flux = gral_fluxes[contains_group_identifier].sum().sum()
        width = contains_group_identifier.sum()

        ax.text(
            i + width / 2 - 0.5,
            ax.get_ylim()[1] * 0.95,
            f"{group}\n{total_flux:.0f} kt/year",
            ha="center",
            va="top",
            fontsize=10,
        )
        i += width
        ax.axvline(i - 0.5, color="k", linestyle="--", alpha=0.5)

        for j in range(i - width, i):
            area_bar = bars[j]
            point_bar = bars[len(bars) // 2 + j]
            area_bar.set_facecolor(color)
            point_bar.set_facecolor(color)
            point_bar.set_hatch("///")
            point_bar.set_alpha(0.5)

    n_sectors = len(gral_fluxes)
    area_bars = bars[:n_sectors]
    point_bars = bars[n_sectors:]
    for idx, (area_bar, point_bar) in enumerate(zip(area_bars, point_bars)):
        total = gral_fluxes.iloc[idx].sum()
        if abs(total) < 0.5:
            continue
        x = area_bar.get_x() + area_bar.get_width() / 2  # type: ignore
        # Place label just above/below the top of the stacked bar
        y_offset = 0.3 if total >= 0 else -70
        va = "bottom" if total >= 0 else "top"
        ax.text(x, total + y_offset, f"{total:.0f}", ha="center", va=va, fontsize=7)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax)

    ax.set_xticklabels(
        [xticklabels[label] for label in gral_fluxes.index], rotation=70, ha="right"
    )

    area_patch = Patch(facecolor="grey", edgecolor="k", label="Area sources")
    point_patch = Patch(
        facecolor="grey", edgecolor="k", hatch="///", label="Point sources", alpha=0.5
    )
    ax.legend(
        handles=[area_patch, point_patch],
        title="Source type",
        # bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Stacked bar chart of total CO2 flux by inventory and source type."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
