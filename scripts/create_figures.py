from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
from windrose import WindroseAxes

import paris_2025 as p

YEAR = 2023
FIGURE_PATH = Path(p.CONFIG["figures_path"])

# Set matplotlib style like size of the figures, text size, etc. by hand
# sns.set_context("paper", font_scale=1)
plt.rcParams.update(
    {
        "figure.figsize": (7, 5),
        # "axes.titlesize": 16,
        # "axes.labelsize": 14,
        # "xtick.labelsize": 12,
        # "ytick.labelsize": 12,
        # "legend.fontsize": 12,
        # "lines.linewidth": 1.5,
        # "grid.linewidth": 0.5,
        # "grid.alpha": 0.5,
        # "font.family": "sans-serif",
        # "font.sans-serif": "Arial",
        # "axes.grid": True,
        # "axes.spines.right": False,
        # "axes.spines.top": False,
        "savefig.dpi": 300,
    }
)
if __name__ == "__main__":
    DIR = "concentration_from_catalog"
    (FIGURE_PATH / DIR).mkdir(parents=True, exist_ok=True)
    p.plotting.concentration_from_catalog.plot_source_group_contribution_to_stations(
        fig_path_1=FIGURE_PATH / DIR / "source_group_contribution_stations.png",
        fig_path_2=FIGURE_PATH / DIR / "source_group_area_contribution_stations.png",
    )
# # %%
# meteo = p.meteo.get_meteo_measurements().sel(time=str(YEAR))

# # %%
# fig, ax = plt.subplots()
# fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
# p.domain.add_domain(ax, legend=True)
# p.domain.add_basemap(ax, zoom=12, provider="CartoDB")
# p.domain.add_size_bar(ax)

# with mpl.rc_context(
#     {
#         "grid.linewidth": mpl.rcParams["grid.linewidth"] * 0.5,
#         "axes.linewidth": mpl.rcParams["lines.linewidth"] * 0.5,
#     }
# ):
#     meteo_filtered = meteo.where(meteo.in_gramm_domain, drop=True)
#     for i, station in tqdm(
#         enumerate(meteo_filtered.station), total=len(meteo_filtered.station)
#     ):
#         data = meteo.sel(station=station)
#         mask = (data.wind_speed.notnull()) & (data.wind_direction.notnull())
#         if mask.sum() / len(mask) < 0.5:
#             continue
#         wrax = inset_axes(
#             ax,
#             width=0.5,
#             height=0.5,
#             loc="center",
#             bbox_to_anchor=(data.x, data.y),
#             bbox_transform=ax.transData,
#             axes_class=WindroseAxes,
#         )
#         wrax.bar(
#             data.wind_direction[mask].values,
#             data.wind_speed[mask].values,
#             normed=True,
#             opening=0.8,
#         )
#         wrax.set_xlabel("")
#         wrax.set_ylabel("")
#         wrax.set_xticklabels([])
#         wrax.set_yticklabels([])
# plt.savefig(
#     Path(p.CONFIG["figures_path"]) / "wind_measurements/wind_roses_measurements.pdf"
# )
