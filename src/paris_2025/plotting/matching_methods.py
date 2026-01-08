import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from paris_2025.plotting.common import get_metadata


def plot_matching_loss_distribution(fig_path):
    """Plot distribution of best matching simulations."""
    matching_loss = xr.open_dataset(
        "/Users/rmaiwald/Levante/Paris/Output/matching_loss.nc"
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
                    f"Distribution of the {n_sim_ids} best "
                    "matching simulations"
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
