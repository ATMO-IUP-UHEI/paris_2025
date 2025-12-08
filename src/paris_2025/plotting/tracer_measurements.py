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
