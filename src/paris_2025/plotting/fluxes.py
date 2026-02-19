from pathlib import Path

import ggpymanager as ggp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

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
    temporal_factor = xr.open_dataset(p.model_input.fluxes.TEMPORAL_PROFILES_NETCDF_PATH)

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
        cadastre_emissions, source_groups, points_ds, GRAL
    """
    GRAL = CONFIG["domain"]["gral"]["bbox"] | CONFIG["domain"]["gral"]
    source_groups = xr.open_dataset(source_groups_path)
    cadastre_emissions = ggp.io.readers.read_cadastre_file(cadastre_path, GRAL)
    cadastre_emissions["type"] = source_groups.sel(
        source_group=cadastre_emissions["source_group"]
    ).type.load()

    points = pd.read_csv(point_path, skiprows=1)
    points.columns = points.columns.str.strip()
    points["type"] = source_groups.sel(
        source_group=points["source group"].values
    ).type.values
    points_ds = points.to_xarray()
    # Convert from kg/h to kg/year
    points_ds["Emission [kg/h]"] *= 365 * 24

    return cadastre_emissions, source_groups, points_ds, GRAL


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
    cadastre_emissions, source_groups, points_ds, GRAL = _load_flux_maps_data(
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

            inv_points_ds = points_ds.where(points_ds.type.str.contains(inv), drop=True)
            if len(inv_points_ds.index) > 0:
                ax.scatter(
                    x=inv_points_ds["x"].values,
                    y=inv_points_ds["y"].values,
                    zorder=10,
                    color="red",
                    edgecolor="k",
                    linewidth=0.5,
                    s=inv_points_ds["Emission [kg/h]"].values / point_size,
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
                plt.Line2D(
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
