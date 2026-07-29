import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.io import shapereader
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from shapely import Geometry

def add_features(
    ax: plt.Axes,
    gridlines: bool = True,
    land: bool = False,
    ocean: bool = False,
    states: bool = True,
    counties: bool = True
):
    """
    Adds common map features and gridlines to a Cartopy GeoAxes object.

    Args:
        ax: The matplotlib GeoAxes object to modify.
        gridlines: If True, adds gridlines with labels on the left and bottom.
        land: If True, adds a land feature.
        ocean: If True, adds an ocean feature.
        states: If True, adds state borders.
        counties: If True, adds U.S. county borders.
    """
    if land:
        ax.add_feature(cfeat.LAND)

    if ocean:
        ax.add_feature(cfeat.OCEAN)

    if states:
        ax.add_feature(cfeat.STATES)

    if counties:
        # Get the path to the Natural Earth shapefile for US counties
        counties_shp = shapereader.natural_earth(
            resolution="10m",
            category="cultural",
            name="admin_2_counties"
        )
        rdr = shapereader.Reader(counties_shp)
        try:
            # Create a feature from the shapefile geometries
            counties_feature = cfeat.ShapelyFeature(
                rdr.geometries(),
                ccrs.PlateCarree(),
                facecolor='none',
                edgecolor='gray'
            )
            ax.add_feature(counties_feature)
        finally:
            rdr.close()

    if gridlines:
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False


def add_geometry(ax: plt.Axes, geo: Geometry, crs, **feature_kws):
    feature = cfeat.ShapelyFeature(geo, crs, **feature_kws)
    ax.add_feature(feature)


def add_fire_perimeter(ax: plt.Axes, geojson_file: os.PathLike, fire_name: str, **feature_kws):
    df = gpd.read_file(geojson_file)
    ii = df.name.str.lower() == fire_name.lower()
    df = df.loc[ii, :].sort_values('dateTime')
    geo = df.iloc[-1]['geometry']
    if df.crs.to_epsg() == 4326:
        crs = ccrs.PlateCarree()
    else:
        crs = ccrs.CRS(df.crs.to_wkt())
    add_geometry(ax, geo, crs, **feature_kws)


def plot_box(loncorn, latcorn, ax=None, **style):
    ax = ax or plt.gca()
    x = [*loncorn, loncorn[0]]
    y = [*latcorn, latcorn[0]]
    ax.plot(x, y, **style)
