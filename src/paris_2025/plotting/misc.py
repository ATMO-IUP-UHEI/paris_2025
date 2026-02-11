"""Miscellaneous plotting functions."""

from pathlib import Path

import matplotlib.pyplot as plt

import paris_2025 as p
from paris_2025.plotting.common import get_metadata


def plot_temperature_anomaly_with_co2(
    fig_path: str | Path,
    time_period: str | slice = slice("2023-09-15", "2023-10"),
    co2_station: str = "JUS_30",
    temp_station: str = "TOUR EIFFEL",
):
    """Plot temperature anomaly heatmap with CO2 and temperature comparison.

    Creates a two-panel figure showing:
    - Temperature anomaly across all meteorological stations
    - CO2 concentration at a reference station with temperature anomaly
      at a specific station on a twin axis

    Parameters
    ----------
    fig_path : str | Path
        Path to save the output figure
    time_period : str | slice, optional
        Time period for the analysis. Default is slice("2023-09-15", "2023-10")
    co2_station : str, optional
        Station code for CO2 measurements. Default is "JUS_30"
    temp_station : str, optional
        Station name for temperature comparison. Default is "TOUR EIFFEL"
    """
    # Load data
    co2_data = p.tracers.get_co2_measurements()
    co2_station_data = co2_data.sel(station=co2_station, time=time_period).co2

    temperature = p.meteo.get_meteo_measurements().sel(time=time_period).temperature
    # Drop stations with missing temperature data
    temperature = temperature.dropna("station")

    # Calculate temperature anomaly
    temp_anomaly = temperature - temperature.mean("station")

    # Create figure
    gridspec = {"width_ratios": [1, 0.02]}
    fig, axs = plt.subplots(
        2, 2, figsize=(18, 6), gridspec_kw=gridspec, sharex="col"
    )

    # Top panel: Temperature anomaly heatmap
    temp_anomaly.plot(
        x="time",
        ax=axs[0, 0],
        cbar_kwargs={"label": "Temperature anomaly [°C]", "cax": axs[0, 1]},
    )

    # Bottom panel: CO2 with temperature on twin axis
    co2_station_data.plot(x="time", ax=axs[1, 0])
    twin_ax = axs[1, 0].twinx()
    temp_station_anomaly = temp_anomaly.sel(station=temp_station)
    temp_station_anomaly.plot(ax=twin_ax, c="red", label="Temperature anomaly")
    twin_ax.set_ylim(-5, 15)
    twin_ax.set_ylabel("Temperature anomaly [°C]", color="red")
    twin_ax.tick_params(axis="y", labelcolor="red")
    twin_ax.set_title("")

    axs[1, 0].set_title(
        f"CO2 at {co2_station} and temperature anomaly at {temp_station}"
    )
    axs[1, 0].set_ylabel("CO2 [ppm]")
    axs[1, 0].grid(axis="x")

    # Remove empty subplot
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(
        fig_path,
        metadata=get_metadata(
            f"Temperature anomaly heatmap with CO2 at {co2_station} "
            f"and temperature at {temp_station}."
        ),
        bbox_inches="tight",
    )
    plt.close(fig)
