import logging
from pathlib import Path

import ggpymanager as ggp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paris_2025 as p
from paris_2025.config import CONFIG
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


def _load_flux_maps_data(
    cadastre_path: str | Path = Path(CONFIG["domain"]["gral"]["conf_path"])
    / "cadastre.dat",
    source_groups_path: str | Path = p.model_input.fluxes.SOURCE_GROUP_NETCDF_PATH,
    point_path: str | Path = Path(CONFIG["domain"]["gral"]["conf_path"]) / "point.dat",
):
    """Load and prepare data for flux map plots.

    Parameters
    ----------
    cadastre_path : str | Path
        Path to the GRAL cadastre.dat file
    source_groups_path : str | Path
        Path to the source groups NetCDF file
    point_path : str | Path
        Path to the GRAL point.dat file

    Returns
    -------
    tuple
        cadastre_emissions, source_groups, point_da, GRAL
    """
    GRAL = CONFIG["domain"]["gral"]["bbox"] | CONFIG["domain"]["gral"]
    source_groups = xr.open_dataset(source_groups_path)
    cadastre_emissions = ggp.io.readers.read_cadastre_file(cadastre_path, GRAL)
    cadastre_emissions["type"] = source_groups.sel(
        source_group=cadastre_emissions["source_group"]
    ).type.load()

    point_da = ggp.io.readers.read_point_file(point_path)["Emission [kg/h]"]
    point_da["type"] = source_groups.sel(
        source_group=point_da["source_group"].reset_coords(drop=True)
    ).type.load()

    return cadastre_emissions, source_groups, point_da, GRAL


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
    cadastre_emissions, source_groups, point_da, GRAL = _load_flux_maps_data(
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

    cadastre_emissions, source_groups, point_da, GRAL = _load_flux_maps_data(
        cadastre_path, source_groups_path, point_path
    )

    point_da["z"] = point_da.z.astype(float)
    if point_da.z.isnull().any():
        logging.warning(
            "The points contain missing values in the z coordinate. All points with "
            "missing z values will be dropped for plotting. This may lead to missing "
            "points in the plots."
        )
        total_emissions = point_da.sum().item()
        isnull = point_da.z.isnull()
        point_da = point_da.sel(index=~isnull)
        dropped_emissions = total_emissions - point_da.sum().item()
        logging.warning(
            f"Dropped {dropped_emissions:.2f} kg/year of emissions from "
            f"{isnull.sum().item()} points."
        )

    area_df = cadastre_emissions.sum(["x", "y"]).groupby("type").sum().to_pandas()
    point_df = point_da.groupby("type").sum().to_pandas()

    xticklabels = {
        "Origins.earth 2023 energie": "Power",
        "Origins.earth 2023 industrie": "Industry",
        "Origins.earth 2023 residentiel": "Combustion",
        "Origins.earth 2023 respiration_humaine": "Human respiration",
        "Origins.earth 2023 tertiaire": "Services",
        "Origins.earth 2023 transport_routier": "Traffic",
        "TNO 2018 Combustion": "Combustion",
        "TNO 2018 Industry": "Industry",
        "TNO 2018 Power": "Power",
        "TNO 2018 Traffic": "Traffic",
        "VPRM GEE": "GEE",
        "VPRM R": "R",
    }

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
    ax.set_ylabel("Emissions (kt/year)")

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
            f"{group}\n{total_flux:.1f} kt/year",
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
        bbox_to_anchor=(1.05, 1),
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
