import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.io import shapereader
import matplotlib.pyplot as plt

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
        # Create a feature from the shapefile geometries
        counties_feature = cfeat.ShapelyFeature(
            shapereader.Reader(counties_shp).geometries(),
            ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='gray'
        )
        ax.add_feature(counties_feature)

    if gridlines:
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
