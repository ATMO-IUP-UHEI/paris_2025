from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

import paris_2025 as p
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
