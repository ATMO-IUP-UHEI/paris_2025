from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paris_2025 as p
from paris_2025.config import CONFIG
from paris_2025.plotting.common import get_metadata


def plot_co2_concentration_heatmap(fig_path: str | Path):
    """Plot CO2 concentration availability as a heatmap."""
    co2 = p.tracers.get_co2_measurements()

    fig, ax = plt.subplots(figsize=(12, 6))
    co2.co2.plot.imshow(ax=ax)
    ax.set_title("CO2 Concentration available in Paris")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata("CO2 concentration availability as heatmap."),
    )
    plt.close(fig)


def plot_co2_by_height_and_instrument(fig_path: str | Path):
    """Plot mean CO2 concentration by height and instrument type."""
    co2 = p.tracers.get_co2_measurements()

    fig, ax = plt.subplots(figsize=(10, 6))
    co2.mean("time", keep_attrs=True).plot.scatter(
        hue="instrument", x="co2", y="height", ax=ax
    )
    ax.set_title("Mean CO2 Concentration in Paris by Type and Height")
    plt.savefig(
        fig_path,
        metadata=get_metadata("Mean CO2 concentration by height and instrument type."),
    )
    plt.close(fig)


def plot_average_co2_spatial(fig_path: str | Path):
    """Plot average CO2 concentration spatially across Paris."""
    co2 = p.tracers.get_co2_measurements()

    fig, ax = plt.subplots(figsize=(12, 8))
    co2.mean("time", keep_attrs=True).plot.scatter(
        x="x",
        y="y",
        hue="co2",
        s=100,
        edgecolor="black",
        alpha=0.7,
        cmap="viridis",
        ax=ax,
    )
    p.domain.add_domain(ax)
    ax.set_title("Average CO2 Concentration in Paris")
    plt.savefig(
        fig_path,
        metadata=get_metadata("Average CO2 concentration spatially across Paris."),
    )
    plt.close(fig)


def plot_co2_instruments_map(fig_path: str | Path):
    """Plot CO2 measurement instruments on a map with basemap."""
    co2 = p.tracers.get_co2_measurements()

    fig, ax = plt.subplots(figsize=(12, 8))
    co2.mean("time", keep_attrs=True).sortby("instrument").plot.scatter(
        x="x",
        y="y",
        hue="instrument",
        s=100,
        edgecolor="black",
        cmap="viridis",
        ax=ax,
    )
    p.domain.add_domain(ax, legend=True)
    p.domain.add_basemap(ax=ax, provider="CartoDB")
    ax.set_title("Instruments for CO2 mixing ratio measurements in Paris")
    plt.savefig(
        fig_path,
        metadata=get_metadata("CO2 measurement instruments on map with basemap."),
    )
    plt.close(fig)


def plot_co2_data_availability(fig_path_2023: str | Path, fig_path_2024: str | Path):
    """Plot CO2 data availability for 2023 and 2024."""
    co2 = p.tracers.get_co2_measurements()
    years = ["2023", "2024"]
    file_paths = [fig_path_2023, fig_path_2024]

    for year, file_path in zip(years, file_paths):
        fig, ax = plt.subplots(figsize=(12, 6))
        data_availability = co2.sel(time=year).notnull().mean("time")
        data_availability.co2.attrs = {
            "long_name": "Mean CO2 data availability",
            "units": "%",
        }
        data_availability.plot.scatter(
            x="x",
            y="y",
            hue="co2",
            s=100,
            edgecolor="black",
            alpha=0.7,
            cmap="viridis",
            ax=ax,
        )
        p.domain.add_domain(ax)
        ax.set_title(f"Data available in Paris for {year}")
        plt.savefig(
            file_path,
            metadata=get_metadata(f"CO2 data availability in Paris for {year}."),
        )
        plt.close(fig)


def plot_picarro_co2_violin(fig_path: str | Path, year: str = "2023"):
    """Plot violin plot of CO2 concentrations by Picarro station."""

    co2 = p.tracers.get_co2_measurements()
    co2 = co2.sel(time=year)

    # Filter for Picarro instruments only
    picarro_co2 = co2.where(co2.instrument == "Picarro", drop=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create violin plot
    violin_data = [
        picarro_co2["co2"].sel(station=station).dropna("time").values
        for station in picarro_co2.station.values
    ]
    ax.violinplot(
        violin_data,
        positions=range(len(violin_data)),
        showmeans=True,
        showmedians=True,
    )

    # Add count labels above each violin
    for i, data in enumerate(violin_data):
        count = len(data)
        y_max = ax.get_ylim()[1]
        ax.text(
            i,
            y_max * 0.98,
            f"n={count}",
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    ax.set_xticks(range(len(violin_data)))
    ax.set_xticklabels(picarro_co2["station"].values, rotation=45, ha="right")
    ax.set_xlabel("Station")
    ax.set_ylabel("CO2 Concentration (ppm)")
    ax.set_title(f"Picarro CO2 Concentration Distribution by Station for {year}")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"CO2 concentration distribution by Picarro station for {year}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_co2_and_meteo_stations_map(fig_path: str | Path):
    """Plot CO2 and meteorological measurement stations on a map.

    Creates a map showing the locations of CO2 measurement stations (colored by
    instrument type) and meteorological stations, with labels for Picarro stations.
    Includes basemap, domain boundaries, and a size bar.

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    """
    # Load data
    co2 = p.tracers.get_co2_measurements()
    co2 = co2.sortby("instrument", ascending=False)
    meteo = p.meteo.get_meteo_measurements()
    meteo = meteo.sel(station=list(CONFIG["matching"]["stations"].keys()))

    # Define colors and markers for different instrument types
    colors = {
        "K96": "orange",
        "HPP": "orange",
        "Meteo": "DodgerBlue",
        "Picarro": "maroon",
    }
    markers = {
        "Picarro": "o",
        "K96": "o",
        "HPP": "o",
        "Meteo": "o",
    }

    # Define legend labels
    legends = {
        "CRDS CO$_2$": (colors["Picarro"], markers["Picarro"]),
        "NDIR CO$_2$": (colors["K96"], markers["K96"]),
        "Wind Measurements": (colors["Meteo"], markers["Meteo"]),
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each instrument type
    for instrument in colors.keys():
        if instrument == "Meteo":
            meteo.plot.scatter(
                x="x",
                y="y",
                s=100,
                c=colors["Meteo"],
                edgecolor="k",
                marker=markers["Meteo"],
                ax=ax,
                zorder=3,
            )
        else:
            co2.sel(station=co2.instrument == instrument).plot.scatter(
                x="x",
                y="y",
                s=100,
                c=colors[instrument],
                edgecolor="k",
                marker=markers[instrument],
                ax=ax,
                zorder=3,
            )
            # Add labels for Picarro stations
            if instrument == "Picarro":
                stations = co2.sel(station=co2.instrument == instrument).station
                station_list = [s.split("_")[0] for s in stations.values]
                station_coords = {
                    s: (x, y)
                    for s, x, y in zip(
                        station_list, stations.x.values, stations.y.values
                    )
                }
                for s in station_coords:
                    ax.text(
                        station_coords[s][0] + 1000,
                        station_coords[s][1],
                        s,
                        fontsize=9,
                        ha="left",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                        ),
                    )

    # Add map features
    p.domain.add_domain(ax, legend=True)
    p.domain.add_basemap(ax=ax, provider="CartoDB")
    p.domain.add_size_bar(ax)

    # Create custom legend
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="k",
            markersize=10,
            label=instrument,
        )
        for instrument, (color, marker) in legends.items()
    ]
    ax.legend(handles=handles, title="Measurement Type", loc="upper right")

    plt.savefig(
        fig_path,
        metadata=get_metadata(
            "Map of CO2 and meteorological measurement stations by instrument type."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
