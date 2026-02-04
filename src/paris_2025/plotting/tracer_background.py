import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from windrose import WindroseAxes

import paris_2025 as p
from paris_2025.config import CONFIG
from paris_2025.plotting.common import get_metadata

WINDROSE_MAX_SPEED = 12


def plot_mean_windrose(fig_path: str | Path, year: str = "2023"):
    """Plot mean wind rose for the specified year."""
    meteo = p.meteo.get_meteo_measurements().sel(time=year)
    # Only select stations used for matching
    meteo = meteo.sel(station=list(CONFIG["matching"]["stations"].keys()))
    _, _, mean_wind_speed, mean_wind_direction = p.meteo.get_mean_wind_vars(meteo)
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax()
    ax.bar(
        mean_wind_direction.values,
        mean_wind_speed.values,
        normed=True,
        opening=0.8,
        edgecolor="white",
        bins=np.linspace(0, WINDROSE_MAX_SPEED, 6),
    )
    ax.set_legend()
    plt.savefig(
        fig_path,
        metadata=get_metadata(f"Mean wind rose for {year}."),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_co2_stations_with_windrose(fig_path: str | Path, year: str = "2023"):
    """Plot CO2 measurement stations with embedded windrose."""
    co2 = p.tracers.get_co2_measurements()
    co2 = co2.sel(time=year)

    meteo = p.meteo.get_meteo_measurements().sel(time=year)
    # Only select stations used for matching
    meteo = meteo.sel(station=list(CONFIG["matching"]["stations"].keys()))
    _, _, mean_wind_speed, mean_wind_direction = p.meteo.get_mean_wind_vars(meteo)

    fig, ax = plt.subplots(figsize=(20, 10))
    p.domain.add_domain(ax)

    co2.sel(station=~co2.in_gral_domain).plot.scatter(
        x="x",
        y="y",
        hue="instrument",
        s=100,
        edgecolor="black",
        alpha=0.7,
        cmap="viridis",
        ax=ax,
    )

    co2.sel(station=co2.in_gral_domain).plot.scatter(
        x="x",
        y="y",
        hue="instrument",
        s=100,
        edgecolor="black",
        alpha=0.7,
        cmap="tab10",
        ax=ax,
    )

    wrax = inset_axes(
        ax,
        width=1,
        height=1,
        loc="center",
        bbox_to_anchor=p.domain.get_centroid_of_domain("gral"),
        bbox_transform=ax.transData,
        axes_class=WindroseAxes,
    )
    wrax.bar(
        mean_wind_direction.values,
        mean_wind_speed.values,
        normed=True,
        opening=0.8,
    )
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"CO2 measurement stations with embedded windrose for {year}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_background_co2_stations(fig_path: str | Path, year: str = "2023"):
    """Plot background CO2 stations (Picarro instruments outside GRAL domain)."""
    co2 = p.tracers.get_co2_measurements()
    co2 = co2.sel(time=year)

    fig, ax = plt.subplots(figsize=(12, 8))
    background_stations = co2.where(
        (co2.instrument == "Picarro") & ~co2.in_gral_domain, drop=True
    )
    background_stations.plot.scatter(
        x="x",
        y="y",
        hue="station",
        s=100,
        edgecolor="black",
        cmap="tab10",
        ax=ax,
    )
    ax.set_title(f"Background CO2 Stations (Picarro) for {year}")
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Background CO2 stations (Picarro outside GRAL domain) for {year}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_background_station_counts(fig_path: str | Path, year: str = "2023"):
    """Plot bar chart of background station usage counts."""
    dynamic_background = p.background.get_dynamic_background_co2().sel(time=year)

    fig, ax = plt.subplots(figsize=(10, 6))
    dynamic_background.background_station.to_pandas().value_counts().plot.bar(ax=ax)
    ax.set_xlabel("Station")
    ax.set_ylabel("Count")
    ax.set_title(f"Background Station Usage Counts for {year}")
    plt.savefig(
        fig_path,
        metadata=get_metadata(f"Background station usage counts for {year}."),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_background_station_co2_violin(fig_path: str | Path, year: str = "2023"):
    """Plot violin plot of CO2 concentrations by background station."""
    dynamic_background = p.background.get_dynamic_background_co2().sel(time=year)

    df = pd.DataFrame(
        {
            "station": dynamic_background.background_station.values,
            "co2": dynamic_background.co2.values,
        }
    )
    # Drop nans
    df = df.dropna(axis="index")

    # Get unique stations sorted by median CO2 or count
    station_order = (
        df.groupby("station")["co2"].count().sort_values(ascending=False).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create violin plot
    ax.violinplot(
        [
            df[df["station"] == station]["co2"].dropna().values
            for station in station_order
        ],
        positions=range(len(station_order)),
        showmeans=True,
        showmedians=True,
    )

    ax.set_xticks(range(len(station_order)))
    ax.set_xticklabels(station_order, rotation=45, ha="right")
    ax.set_xlabel("Station")
    ax.set_ylabel("CO2 Concentration (ppm)")
    ax.set_title(f"Background CO2 Concentration Distribution by Station for {year}")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"CO2 concentration distribution by background station for {year}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_background_station_hourly_contribution(
    fig_path: str | Path, year: str = "2023"
):
    """
    Plot stacked area chart showing relative contribution of each
    background station by hour of day.
    """
    dynamic_background = p.background.get_dynamic_background_co2().sel(time=year)

    # Extract hour of day
    hourly_data = dynamic_background.assign_coords(hour=dynamic_background.time.dt.hour)

    # Count occurrences of each station per hour
    station_hour_counts = {}
    unique_stations = np.unique(dynamic_background.background_station.values)

    for station in unique_stations:
        station_mask = hourly_data.background_station == station
        counts_per_hour = []
        for hour in range(24):
            hour_mask = hourly_data.hour == hour
            count = (station_mask & hour_mask).sum().values
            counts_per_hour.append(count)
        station_hour_counts[station] = counts_per_hour

    # Convert to relative contributions (density that adds to 1 for each hour)
    hours = np.arange(24)
    total_per_hour = np.array(
        [sum(station_hour_counts[s][h] for s in unique_stations) for h in hours]
    )

    # Avoid division by zero
    total_per_hour = np.where(total_per_hour == 0, 1, total_per_hour)

    station_contributions = {}
    for station in unique_stations:
        station_contributions[station] = (
            np.array(station_hour_counts[station]) / total_per_hour
        )

    # Create stacked area plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Stack the contributions
    bottom = np.zeros(24)
    for station in unique_stations:
        ax.fill_between(
            hours,
            bottom,
            bottom + station_contributions[station],
            label=station,
            alpha=0.7,
        )
        bottom += station_contributions[station]

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Relative Contribution")
    ax.set_title(f"Background Station Hourly Contribution for {year}")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Relative contribution of background stations by hour of day for {year}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def get_color_map_for_stations() -> dict[str, str]:
    tab10_colors = plt.get_cmap("tab10").colors  # type: ignore
    color_map = {}
    co2 = p.tracers.get_co2_measurements()
    station_names = co2.sel(station=co2.instrument == "Picarro").station
    grouped = station_names.groupby(station_names.str[:3])
    for (l, g), c in zip(
        grouped,
        tab10_colors,
    ):
        for s in g.values:
            color_map[s] = c
    return color_map


def plot_background_station_count(
    fig_path: str | Path, background_type: str, grouper_type: str
):

    color_map = get_color_map_for_stations()

    match background_type:
        case "dynamic":
            background = p.background.get_dynamic_background_co2().load().co2
        case "minimum":
            background = p.background.get_minimum_background_co2().load()
        case "binned":
            background = p.background.get_binned_background_co2().load()
        case _:
            raise ValueError(f"Unsupported background type: {background_type}")

    match grouper_type:
        case "hour":
            label = "Hour of Day"
            grouper = "time.hour"
        case "month":
            label = "Month"
            grouper = "time.month"
        case "wind direction":
            meteo = p.meteo.get_meteo_measurements()
            grouper = (
                meteo.wind_direction.sel(station="TOUR EIFFEL")
                .sel(time=background.time)
                .load()
            )
            label = "Wind Direction (deg)"
        case _:
            raise ValueError(f"Unsupported grouper: {grouper_type}")

    if background_type == "binned":
        n_cols = len(background.height_bins)
        fig, axs = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        for ax, bin in zip(axs, background.height_bins):
            grouped = background.sel(height_bins=bin).background_station.groupby(
                grouper
            )
            df = pd.concat(
                {
                    name: g.to_pandas().value_counts().rename_axis(name)
                    for name, g in grouped
                },
                join="outer",
                axis="columns",
            )
            if df.isnull().all().all():
                logging.info(f"No data for height bin {bin.values}, skipping plot.")
                continue
            # df = df.reindex(df_index)
            df.T.plot.area(color=[color_map[i] for i in df.index], linewidth=0, ax=ax)
            ax.legend()  # bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.set_title(f"Selected stations for height bin {bin.values}")
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
    else:
        grouped = background.background_station.groupby(grouper)
        df = pd.concat(
            {
                name: g.to_pandas().value_counts().rename_axis(name)
                for name, g in grouped
            },
            join="outer",
            axis="columns",
        )
        # Dropna in index
        df = df[df.index != "nan"]
        df = df.sort_index()
        df.T.plot.area(color=[color_map[i] for i in df.index], linewidth=0)
        plt.gca().legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xlabel(label)
        plt.ylabel("Count")

    # Save figure
    plt.suptitle(
        f"Background station count by {grouper_type} for {background_type} background"
    )
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Background station count by {grouper_type} for "
            f"{background_type} background."
        ),
        bbox_inches="tight",
    )
    plt.close()


def plot_co2_diff_vs_wind_speed(fig_path: str | Path, year: str = "2023"):
    """Plot CO2 vertical difference vs wind speed for day and night."""
    meteo = p.meteo.get_meteo_measurements()
    # Drop lidar measurements
    meteo = meteo.sel(station=meteo.operator != "lidar")
    meteo = meteo.sel(station=meteo.in_gramm_domain)
    meteo = meteo.sel(time=year)
    percent = meteo.wind_speed.notnull().sum("time") / len(meteo.time) * 100
    meteo = meteo.sel(station=percent > 50)

    co2 = p.tracers.get_co2_measurements()
    co2 = co2.sel(time=year)

    time_center = 12  # Center time at noon
    time_window = 6  # 6 hours window
    during_day = np.abs(co2.time.dt.hour - time_center) < time_window

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    hb = axs[0].hist2d(
        co2.co2.sel(station="SAC_100") - co2.co2.sel(station="SAC_15"),
        meteo.mean("station").wind_speed.where(during_day),
        range=[[-100, 100], [0, 20]],
        cmap="viridis",
        bins=50,
        cmin=1,
        vmax=100,
    )[-1]
    plt.colorbar(hb, ax=axs[0], label="Counts")
    axs[0].set_xlabel("CO2 difference (SAC_100 - SAC_15) [ppm]")
    axs[0].set_ylabel("Wind speed (m/s)")
    axs[0].set_title(
        "CO2 difference vs. Wind speed during daytime (6 hours around noon)"
    )

    hb = axs[1].hist2d(
        co2.co2.sel(station="SAC_100") - co2.co2.sel(station="SAC_15"),
        meteo.mean("station").wind_speed.where(~during_day),
        range=[[-100, 100], [0, 20]],
        cmap="viridis",
        bins=50,
        cmin=1,
        vmax=100,
    )[-1]
    plt.colorbar(hb, ax=axs[1], label="Counts")
    axs[1].set_xlabel("CO2 difference (SAC_100 - SAC_15) [ppm]")
    axs[1].set_ylabel("Wind speed (m/s)")
    axs[1].set_title(
        "CO2 difference vs. Wind speed during nighttime (6 hours around midnight)"
    )

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"CO2 vertical difference vs wind speed for day and night in {year}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_co2_diff_vs_wind_direction(fig_path: str | Path, year: str = "2023"):
    """Plot CO2 vertical difference vs wind direction for day and night."""
    meteo = p.meteo.get_meteo_measurements()
    meteo = meteo.sel(station=meteo.in_gramm_domain)
    meteo = meteo.sel(time=year)
    percent = meteo.wind_speed.notnull().sum("time") / len(meteo.time) * 100
    meteo = meteo.sel(station=percent > 50)

    co2 = p.tracers.get_co2_measurements()
    co2 = co2.sel(time=year)

    during_day = np.abs(co2.time.dt.hour - 12) < 6

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    hb_day = axs[0].hexbin(
        co2.co2.sel(station="SAC_100") - co2.co2.sel(station="SAC_15"),
        meteo.mean("station").wind_direction.where(during_day),
        gridsize=50,
        cmap="viridis",
        mincnt=1,
        vmax=100,
    )
    plt.colorbar(hb_day, ax=axs[0], label="Counts")
    axs[0].set_xlabel("CO2 difference (SAC_100 - SAC_15)")
    axs[0].set_ylabel("Wind direction (deg)")
    title_day = (
        "CO2 difference vs. Wind direction during daytime " "(6 hours around noon)"
    )
    axs[0].set_title(title_day)

    hb_night = axs[1].hexbin(
        co2.co2.sel(station="SAC_100") - co2.co2.sel(station="SAC_15"),
        meteo.mean("station").wind_direction.where(~during_day),
        gridsize=50,
        cmap="viridis",
        mincnt=1,
        vmax=100,
    )
    plt.colorbar(hb_night, ax=axs[1], label="Counts")
    axs[1].set_xlabel("CO2 difference (SAC_100 - SAC_15)")
    axs[1].set_ylabel("Wind direction (deg)")
    title_night = (
        "CO2 difference vs. Wind direction during nighttime "
        "(6 hours around midnight)"
    )
    axs[1].set_title(title_night)

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"CO2 vertical difference vs wind direction for day and night in {year}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
