from os import environ

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely

from paris_2025.config import CONFIG


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
        try:
            # Check if the environment variable is set
            cartodb_api_key = environ["CARTODB_API_KEY"]
            provider = ctx.providers.CartoDB.PositronNoLabels(
                api_key=cartodb_api_key
            )  # type: ignore
            provider["url"] = provider["url"] + "?key={api_key}"
        except KeyError as e:
            raise ValueError(
                "CARTODB_API_KEY environment variable is not set. Please set it to use "
                "CartoDB provider."
            ) from e
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
    # ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    return ax


def _format_degrees(value: float, positive: str, negative: str, step: float) -> str:
    """Format a degree value with a hemisphere suffix and step-appropriate precision.

    The precision is the smallest that still resolves the tick spacing, so a
    0.25\u00b0 step is labelled "2.25\u00b0E" and a 1\u00b0 step "2\u00b0E".
    """
    decimals = 0
    while decimals < 6 and abs(round(step, decimals) - step) > 1e-9:
        decimals += 1
    suffix = positive if value >= 0 else negative
    return f"{abs(value):.{decimals}f}\u00b0{suffix}"


def add_latlon_ticks(
    ax: plt.Axes,  # type: ignore
    crs: str = CONFIG["domain"]["crs"],
    n_ticks: int = 4,
    grid: bool = False,
    grid_kwargs: dict | None = None,
) -> None:
    """
    Label the axes of a projected map with geographic (lat/lon) ticks.

    The axes stay in the projected CRS (metres), so this can be called after
    ``add_basemap``, ``add_domain`` or any GeoDataFrame plot without changing
    the data coordinates. Meridians and parallels are not exactly axis-parallel
    in a projected CRS, so each tick is placed where its graticule line crosses
    the axis it is labelled on: longitudes at the bottom spine, latitudes at
    the left spine.

    Call this *after* all data has been plotted and the limits are final — the
    ticks are computed from the current ``xlim``/``ylim``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to label, with data in ``crs`` coordinates.
    crs : str, optional
        The projected CRS of the axes, by default the domain CRS from CONFIG.
    n_ticks : int, optional
        Approximate number of ticks per axis, by default 4.
    grid : bool, optional
        If True, draw the graticule (curved in the projected CRS) and switch
        off the axes' own grid, by default False. Matplotlib's grid draws
        straight lines through the tick positions, which would show up as a
        second, rectangular grid next to the curved graticule.
    grid_kwargs : dict, optional
        Extra keyword arguments for the graticule lines, e.g.
        ``dict(color="white", linewidth=0.5)``.

    Notes
    -----
    The axis labels are cleared: they usually name the projected coordinate
    (metres), which no longer describes the ticks. Set your own labels after
    this call if you want them.
    """
    from matplotlib.ticker import MaxNLocator
    from pyproj import Transformer

    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Sample the whole frame, not just the corners: the lon/lat range of a
    # rotated projected frame is wider than its corner values suggest.
    edge = np.linspace(0, 1, 20)
    frame_x = np.concatenate(
        [
            xlim[0] + edge * (xlim[1] - xlim[0]),
            xlim[0] + edge * (xlim[1] - xlim[0]),
            np.full(edge.size, xlim[0]),
            np.full(edge.size, xlim[1]),
        ]
    )
    frame_y = np.concatenate(
        [
            np.full(edge.size, ylim[0]),
            np.full(edge.size, ylim[1]),
            ylim[0] + edge * (ylim[1] - ylim[0]),
            ylim[0] + edge * (ylim[1] - ylim[0]),
        ]
    )
    frame_lon, frame_lat = to_lonlat.transform(frame_x, frame_y)

    locator = MaxNLocator(nbins=n_ticks, steps=[1, 2, 2.5, 5, 10])
    lon_ticks = locator.tick_values(frame_lon.min(), frame_lon.max())
    lat_ticks = locator.tick_values(frame_lat.min(), frame_lat.max())
    lon_step = lon_ticks[1] - lon_ticks[0]
    lat_step = lat_ticks[1] - lat_ticks[0]

    # Densified graticule lines spanning the visible lon/lat range.
    lon_line = np.linspace(frame_lon.min(), frame_lon.max(), 100)
    lat_line = np.linspace(frame_lat.min(), frame_lat.max(), 100)

    grid_style = {"color": "grey", "linewidth": 0.5, "alpha": 0.5, "zorder": 2}
    grid_style.update(grid_kwargs or {})
    if grid:
        # Matplotlib's own grid would draw straight lines through the same
        # ticks, doubling the graticule with a rectangular one.
        ax.grid(False)

    xticks, xlabels = [], []
    for lon in lon_ticks:
        # Where does this meridian cross the bottom spine?
        mx, my = to_proj.transform(np.full(lat_line.size, lon), lat_line)
        x = np.interp(ylim[0], my, mx)
        if xlim[0] <= x <= xlim[1]:
            xticks.append(x)
            xlabels.append(_format_degrees(lon, "E", "W", lon_step))
        if grid:
            ax.plot(mx, my, **grid_style)

    yticks, ylabels = [], []
    for lat in lat_ticks:
        # Where does this parallel cross the left spine?
        px, py = to_proj.transform(lon_line, np.full(lon_line.size, lat))
        y = np.interp(xlim[0], px, py)
        if ylim[0] <= y <= ylim[1]:
            yticks.append(y)
            ylabels.append(_format_degrees(lat, "N", "S", lat_step))
        if grid:
            ax.plot(px, py, **grid_style)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    # The axis labels describe projected metres, which the ticks no longer show.
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Graticule lines extend beyond the frame; keep the original extent.
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def add_domain(ax: plt.Axes, legend=False) -> None:  # type: ignore
    """
    Adds the domain bounding boxes and labels to the given axes.
    """
    from paris_2025.plotting import DOMAIN_COLORS

    gdf = get_domain_as_geopandas()
    textbuffer = 500
    # Get the fontsize for the axes labels
    fontsize = ax.xaxis.get_ticklabels()[0].get_fontsize()
    # Plot GRAL and GRAMM boxes separately for different colors
    for idx in [0, 1]:
        row = gdf.iloc[idx]
        color = DOMAIN_COLORS[row.label.split()[0]]
        x0, y0, x1, y1 = gdf.geometry.iloc[idx].bounds  # type: ignore
        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            color=color,
            linewidth=2,
        )
        if not legend:
            # Add label to geometries in same color
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
    if legend:
        # Create a legend for the domain boxes
        handles = [
            plt.Line2D([0], [0], color=DOMAIN_COLORS["GRAMM"], lw=2, label="GRAMM"),  # type: ignore
            plt.Line2D([0], [0], color=DOMAIN_COLORS["GRAL"], lw=2, label="GRAL"),  # type: ignore
        ]
        legend = plt.legend(
            handles=handles,
            loc="upper left",
            fontsize=fontsize,
            frameon=False,
            ncol=2,
        )
        ax.add_artist(legend)
    # Add buffer to axes limits for better visualization
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xbuffer = 5000
    ybuffer = 2000
    ax.set_xlim(xlim[0] - xbuffer, xlim[1] + xbuffer)
    ax.set_ylim(ylim[0] - ybuffer, ylim[1] + ybuffer)
    ax.set_aspect("equal")


def add_size_bar(ax: plt.Axes) -> None:  # type: ignore
    """
    Adds a scale bar to the given axes.
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    scalebar = AnchoredSizeBar(
        ax.transData,
        5000,  # length in meters
        "5 km",
        loc="lower right",
        pad=0.1,
        color="black",
        frameon=False,
        size_vertical=100,
    )
    ax.add_artist(scalebar)
