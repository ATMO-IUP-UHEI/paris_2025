from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
from windrose import WindroseAxes
from matplotlib.cm import viridis  # type: ignore
from matplotlib.colors import Normalize

import paris_2025 as p
from paris_2025.plotting.common import get_metadata


def plot_meteo_measurements_heatmap(fig_path: str | Path):
    """Plot meteorological measurements availability as a heatmap."""
    meteo = p.meteo.get_meteo_measurements()

    fig, ax = plt.subplots(figsize=(12, 6))
    meteo.wind_speed.plot.imshow(ax=ax)
    ax.set_title("Meteorological Measurements available in Paris")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata("Meteorological measurements availability as heatmap."),
    )
    plt.close(fig)


def plot_wind_roses_of_meteo_measurements(fig_path: str | Path):
    YEAR = 2023
    meteo = p.meteo.get_meteo_measurements().sel(time=str(YEAR))

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    p.domain.add_domain(ax, legend=True)
    p.domain.add_basemap(ax, zoom=12, provider="CartoDB")
    p.domain.add_size_bar(ax)

    with mpl.rc_context(
        {
            "grid.linewidth": mpl.rcParams["grid.linewidth"] * 0.5,
            "axes.linewidth": mpl.rcParams["lines.linewidth"] * 0.5,
        }
    ):
        meteo_filtered = meteo.where(meteo.in_gramm_domain, drop=True)
        for i, station in tqdm(
            enumerate(meteo_filtered.station), total=len(meteo_filtered.station)
        ):
            data = meteo.sel(station=station)
            mask = (data.wind_speed.notnull()) & (data.wind_direction.notnull())
            if mask.sum() / len(mask) < 0.5:
                continue
            wrax = inset_axes(
                ax,
                width=0.5,
                height=0.5,
                loc="center",
                bbox_to_anchor=(data.x, data.y),
                bbox_transform=ax.transData,
                axes_class=WindroseAxes,
            )
            wrax.bar(
                data.wind_direction[mask].values,
                data.wind_speed[mask].values,
                normed=True,
                opening=0.8,
            )
            wrax.set_xlabel("")
            wrax.set_ylabel("")
            wrax.set_xticklabels([])
            wrax.set_yticklabels([])
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Wind roses of meteorological measurements with more than 50% of "
            f"measurements during {YEAR}."
        ),
    )


def plot_meteo_overview(fig_path):
    meteo = p.meteo.get_meteo_measurements()
    time_str = "11-2023"
    data = meteo.sel(time=time_str)
    norm = Normalize(vmin=0, vmax=np.max(data.altitude))
    cmap = viridis
    colors = cmap(norm(data.altitude))
    fig, axs = plt.subplots(
        nrows=len(meteo.data_vars),
        ncols=1,
        figsize=(10, 2 * len(meteo.data_vars)),
        sharex=True,
    )
    for j, var in enumerate(meteo.data_vars):
        for i in range(len(data.station.values)):
            axs[j].plot(
                data.time.values,
                data[var][:, i],
                linewidth=1,
                color=colors[i],
                alpha=0.5,
                label=data.station.values[i],
            )
        axs[j].set_ylabel(f"{var}")
    axs[0].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", ncol=5)
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Overview of meteorological measurements for {time_str}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_wind_data_availability(fig_path_1, fig_path_2):
    meteo = p.meteo.get_meteo_measurements()
    x0, y0, x1, y1 = p.domain.get_domain_as_geopandas().geometry.total_bounds
    in_domain = (meteo.x >= x0) & (meteo.x <= x1) & (meteo.y >= y0) & (meteo.y <= y1)
    years = ["2023", "2024"]
    file_paths = [fig_path_1, fig_path_2]

    for year, file_path in zip(years, file_paths):
        fig, ax = plt.subplots(figsize=(12, 6))
        data = meteo.sel(time=year, station=in_domain).notnull().mean("time")

        data.plot.scatter(
            x="x",
            y="y",
            hue="wind_speed",
            s=100,
            edgecolor="black",
            alpha=0.7,
            cmap="viridis",
            ax=ax,
        )
        p.domain.add_domain(ax)
        # p.domain.add_basemap(ax, zoom=12, provider="CartoDB")
        n_stations = (data.wind_speed > 0.5).sum().values
        ax.set_title(
            f"Data available in Paris for {year} with over 50% coverage: {n_stations}"
        )
        plt.savefig(
            file_path,
            metadata=get_metadata(
                f"Wind data availability in Paris for {year} with over 50% coverage."
            ),
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_wind_speed_by_altitude_lidar(fig_path: str | Path, year: str = "2023"):
    """Plot mean wind speed by altitude for lidar measurements."""
    meteo = p.meteo.get_meteo_measurements()
    meteo = meteo.sel(time=year)

    fig, ax = plt.subplots(figsize=(10, 6))
    lidar_data = meteo.where(meteo.operator == "lidar", drop=True)
    lidar_data.wind_speed.mean("time").plot.scatter(hue="altitude", ax=ax, s=100)
    ax.set_title(f"Mean Wind Speed by Altitude (Lidar) for {year}")
    ax.set_xlabel("Station Index")
    ax.set_ylabel("Mean Wind Speed (m/s)")

    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(f"Mean wind speed by altitude for lidar in {year}."),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_wind_speed_by_altitude_non_lidar(fig_path: str | Path, year: str = "2023"):
    """Plot mean wind speed by altitude for non-lidar measurements."""
    meteo = p.meteo.get_meteo_measurements()
    meteo = meteo.sel(time=year)

    fig, ax = plt.subplots(figsize=(10, 6))
    non_lidar_data = meteo.where(meteo.operator != "lidar", drop=True)
    non_lidar_data_sorted = non_lidar_data.sortby("altitude")
    non_lidar_data_sorted.wind_speed.mean("time").plot.scatter(
        hue="altitude", ax=ax, s=100
    )
    ax.set_title(f"Mean Wind Speed by Altitude (Non-Lidar) for {year}")
    ax.set_xlabel("Station Index")
    ax.set_ylabel("Mean Wind Speed (m/s)")

    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(f"Mean wind speed by altitude for non-lidar in {year}."),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_wind_components(fig_path: str | Path, year: str = "2023"):
    """Plot wind components (ux vs vy) for all stations."""
    meteo = p.meteo.get_meteo_measurements()
    non_lidar_mask = meteo["operator"] != "lidar"
    not_all_nan = meteo["u_wind"].notnull().any("time")
    meteo.where(
        non_lidar_mask & not_all_nan & meteo.in_gramm_domain, drop=True
    ).plot.scatter(x="u_wind", y="v_wind", col="station", col_wrap=4, alpha=0.5)
    plt.suptitle(f"Wind Components for Non-Lidar Stations in {year}", y=1.02)
    plt.savefig(
        fig_path,
        metadata=get_metadata(f"Wind components for non-lidar stations in {year}."),
        bbox_inches="tight",
    )
    plt.close()


def plot_hodographs(fig_path: str | Path, station_identifier: str = "PAARBO"):
    """Plot hodographs (wind components by altitude) for a specific station."""
    meteo = p.meteo.get_meteo_measurements()

    d = meteo.where(meteo.station.str.contains(station_identifier), drop=True)
    d = d.where(d.u_wind.notnull().any(dim="station"), drop=True)

    time_ids = np.arange(0, 500, 10)
    n_cols = 5
    n_rows = len(time_ids) // n_cols + int(len(time_ids) % n_cols > 0)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        constrained_layout=True,
    )

    # Get altitude range for consistent colorbar
    vmin = d.altitude.min().values
    vmax = d.altitude.max().values

    for time, ax in zip(time_ids, axs.flatten()):
        d.isel(time=time).plot.scatter(
            x="u_wind",
            y="v_wind",
            hue="altitude",
            cmap="viridis",
            add_colorbar=False,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        max_range = max(abs(xlims[0]), abs(xlims[1]), abs(ylims[0]), abs(ylims[1]))
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_title(f"Timestep: {time}")
        ax.set_xlabel("U wind component (m/s)")
        ax.set_ylabel("V wind component (m/s)")
        ax.set_aspect("equal", adjustable="box")

    # Hide unused subplots
    for ax in axs.flatten()[len(time_ids) :]:
        ax.axis("off")

    # Add colorbar
    cax = fig.add_axes([1.02, 0.3, 0.02, 0.4])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cax, label="Altitude (m)")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Hodographs for station {station_identifier} over {len(time_ids)} timesteps."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
