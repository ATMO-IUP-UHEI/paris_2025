import geopandas as gpd
import shapely
import contextily as ctx
import matplotlib.pyplot as plt
from paris_2025.config import load_config

CONFIG = load_config()


def checking_domain(domain_name: str, x, y):
    """
    Checks if the given coordinates (x, y) are within the specified domain.
    Parameters
    ----------
    domain_name : str
        The name of the domain to check against, e.g., "gramm" or "gral".
    x : array-like
        The x-coordinates to check.
    y : array-like
        The y-coordinates to check.

    Returns
    -------
    array-like
        A boolean array indicating whether each coordinate pair (x, y) is within the
        domain.
    """
    bbox = CONFIG["domain"][domain_name]["bbox"]
    in_domain_x = (bbox["x0"] <= x) & (x <= bbox["x1"])
    in_domain_y = (bbox["y0"] <= y) & (y <= bbox["y1"])
    return in_domain_x & in_domain_y


def get_domain_as_geopandas():
    """
    Returns the domain as a GeoDataFrame with GRAL and GRAMM bounding boxes.
    """
    gramm_bbox = CONFIG["domain"]["gramm"]["bbox"]
    gral_bbox = CONFIG["domain"]["gral"]["bbox"]
    gramm_width = gramm_bbox["x1"] - gramm_bbox["x0"]
    gramm_height = gramm_bbox["y1"] - gramm_bbox["y0"]
    gral_width = gral_bbox["x1"] - gral_bbox["x0"]
    gral_height = gral_bbox["y1"] - gral_bbox["y0"]
    gdf = gpd.GeoDataFrame(
        {
            "label": [
                f"GRAL ({gral_width*1e-3:.0f}x{gral_height*1e-3:.0f} km)",
                f"GRAMM ({gramm_width*1e-3:.0f}x{gramm_height*1e-3:.0f} km)",
            ],
            "geometry": [
                shapely.geometry.box(*gral_bbox.values()),
                shapely.geometry.box(*gramm_bbox.values()),
            ],
        },
        crs=CONFIG["domain"]["crs"],
    )
    return gdf


def get_centroid_of_domain(domain_name: str):
    """
    Returns the centroid coordinates of the specified domain.

    Parameters
    ----------
    domain_name : str
        The name of the domain, e.g., "gramm" or "gral".

    Returns
    -------
    centroid_x, centroid_y : tuple
        The centroid coordinates (x, y) of the specified domain.
    """

    bbox = CONFIG["domain"][domain_name]["bbox"]
    centroid_x = (bbox["x0"] + bbox["x1"]) / 2
    centroid_y = (bbox["y0"] + bbox["y1"]) / 2
    return centroid_x, centroid_y


def add_basemap(ax, zoom=12, provider=None):
    """
    Adds a basemap to the given axes using contextily.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to which the basemap will be added.
    zoom : int, optional
        The zoom level for the basemap, by default 12.
    provider : str
        The provider for the basemap, by default None. If None, uses Esri WorldImagery.
        If "CartoDB", uses CartoDB PositronNoLabels. If "OpenStreetMap", uses
        OpenStreetMap Mapnik.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the added basemap.
    """
    if provider is None:
        provider = ctx.providers.Esri.WorldImagery  # type: ignore
    elif provider == "CartoDB":
        provider = ctx.providers.CartoDB.PositronNoLabels  # type: ignore
    elif provider == "OpenStreetMap":
        provider = ctx.providers.OpenStreetMap.Mapnik  # type: ignore
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    ctx.add_basemap(
        ax,
        crs=CONFIG["domain"]["crs"],
        source=provider,
        zoom=zoom,  # type: ignore
    )
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    return ax


def add_domain(ax: plt.Axes) -> None:  # type: ignore
    """
    Adds the domain bounding boxes and labels to the given axes.
    """
    gdf = get_domain_as_geopandas()
    colors = ["midnightblue", "royalblue"]
    textbuffer = 500
    # Get the fontsize for the axes labels
    fontsize = ax.xaxis.get_ticklabels()[0].get_fontsize()
    # Plot GRAL and GRAMM boxes separately for different colors
    for idx, color in zip([0, 1], colors):
        # gdf.iloc[[idx]].plot(
        #     column="label",
        #     legend=False,
        #     facecolor="none",
        #     linestyle="-",
        #     linewidth=3,
        #     edgecolor=color,
        #     ax=ax,
        #     categorical=True,
        # )
        x0, y0, x1, y1 = gdf.geometry.iloc[idx].bounds  # type: ignore
        plt.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            color=color,
            linewidth=2,
        )
        # Add label to geometries in same color
        row = gdf.iloc[idx]
        ax.text(
            row.geometry.bounds[0] + textbuffer,
            row.geometry.bounds[1] + textbuffer,
            row.label,
            fontsize=fontsize,
            ha="left",
            va="bottom",
            color=color,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=0.5,
                boxstyle="round,pad=0.1",
            ),
        )
    # Add buffer to axes limits for better visualization
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xbuffer = 5000
    ybuffer = 2000
    ax.set_xlim(xlim[0] - xbuffer, xlim[1] + xbuffer)
    ax.set_ylim(ylim[0] - ybuffer, ylim[1] + ybuffer)
