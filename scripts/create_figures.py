from pathlib import Path

import matplotlib.pyplot as plt


import paris_2025 as p
import paris_2025.plotting as plotting

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
    fig_path_1 = FIGURE_PATH / DIR / "source_group_contribution_stations.png"
    fig_path_2 = FIGURE_PATH / DIR / "source_group_area_contribution_stations.png"
    if not fig_path_1.exists() or not fig_path_2.exists():
        (plotting.conc_from_catalog.plot_source_group_contribution_to_stations)(
            fig_path_1, fig_path_2
        )

    DIR = "wind_measurements"
    fig_path = FIGURE_PATH / DIR / "wind_roses_measurements.png"
    if not fig_path.exists():
        p.plotting.meteo_measurements.plot_wind_roses_of_meteo_measurements(fig_path)

    fig_path = FIGURE_PATH / DIR / "meteo_overview.png"
    if not fig_path.exists():
        p.plotting.meteo_measurements.plot_meteo_overview(fig_path)

    fig_path_1 = FIGURE_PATH / DIR / "wind_data_availability_2023.png"
    fig_path_2 = FIGURE_PATH / DIR / "wind_data_availability_2024.png"
    if not fig_path_1.exists() or not fig_path_2.exists():
        p.plotting.meteo_measurements.plot_wind_data_availability(
            fig_path_1, fig_path_2
        )
