from pathlib import Path

import matplotlib.pyplot as plt

import paris_2025 as p
from paris_2025.plotting.common import get_metadata


def plot_co2_concentration_heatmap(fig_path: str | Path):
    """Plot CO2 concentration availability as a heatmap."""
    co2 = p.tracers.get_co2_measurements()

    fig, ax = plt.subplots(figsize=(12, 6))
    co2.co2.plot.imshow(ax=ax)
    ax.set_title("CO2 Concentration available in Paris")
    plt.xticks(rotation=90)
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
