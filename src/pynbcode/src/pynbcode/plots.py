import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.io import shapereader
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import re

from typing import Optional, Tuple


STD_CO_ENH = 0.03 # mol/m2

class PlumeMask:
    def __init__(self, enhancement_file: str, enhancement_group: str = 'Time', co_enhancement_threshold: float = STD_CO_ENH) -> None:
        with Dataset(enhancement_file) as ds:
            self.dates = pd.to_datetime(ds['date'][:])
            self.lon = ds['grid_lon'][:]
            self.lat = ds['grid_lat'][:]
            co_enh = ds[enhancement_group]['co_enhancement'][:]
            self.mask = co_enh < co_enhancement_threshold

    def apply_mask(self, values, val_lon, val_lat, val_date, fill_val=np.nan):
        if not np.allclose(self.lon, val_lon) or not np.allclose(self.lat, val_lat):
            raise NotImplementedError('Grid lat/lon is inconsistent with mask lat/lon')

        idate = np.flatnonzero(self.dates == val_date)[0]
        mask_arr = self.mask[idate, :, :]
        values[mask_arr] = fill_val



def plot_gridded_data(
    grid_file: str,
    variable: str,
    level_idx: Optional[int] = None,
    vscale: float = 1.0,
    cblabel: str = "",
    extent: Optional[Tuple[float, float, float, float]] = None,
    plume_mask: Optional['PlumeMask'] = None,
    ax: Optional[plt.Axes] = None,
    **pcolormesh_kw
):
    """
    Plots gridded data from a netCDF file on a map.

    Args:
        grid_file: Path to the netCDF grid file.
        variable: Name of the variable to plot within the netCDF file.
        level_idx: For 3D variables, the index of the vertical level to plot.
        vscale: A scaling factor to apply to the data values.
        cblabel: Label for the colorbar. If empty, the variable name is used.
        extent: A tuple (lon_min, lon_max, lat_min, lat_max) to define the
                plotting area.
        plume_mask: An optional PlumeMask object to apply to the data.
        ax: A matplotlib Axes object to plot on. If None, a new one is created.
        **pcolormesh_kw: Additional keyword arguments passed to pcolormesh.
    """
    with Dataset(grid_file, 'r') as ds:
        lon = ds.variables["grid_lon"][:]
        lat = ds.variables["grid_lat"][:]

        var = ds.variables[variable]
        is_3d = var.ndim == 3

        if is_3d and level_idx is None:
            raise ValueError(f"'{variable}' is 3D, a level_idx must be provided.")
        elif is_3d:
            values = var[level_idx, :, :]
        else:
            values = var[:, :]

        if plume_mask is not None:
            # Assumes the date is in 'yyyymmdd' format in the filename
            match = re.search(r"\d{8}", grid_file)
            if match:
                date_str = match.group(0)
                inferred_date = pd.to_datetime(date_str)
                # Assumes apply_mask modifies 'values' or returns a modified copy
                plume_mask.apply_mask(values, lon, lat, inferred_date)

        if extent:
            lon_inds = (lon >= extent[0]) & (lon <= extent[1])
            lat_inds = (lat >= extent[2]) & (lat <= extent[3])
            lon = lon[lon_inds]
            lat = lat[lat_inds]
            # Use np.ix_ to correctly index the 2D array from 1D boolean masks
            values = values[np.ix_(lat_inds, lon_inds)]

        values = vscale * values

        if ax is None:
            _, ax = plt.subplots(
                subplot_kw={"projection": ccrs.PlateCarree()}
            )

        # The '.T' transposes the array, matching the Julia 'values''
        mesh = ax.pcolormesh(lon, lat, values, **pcolormesh_kw)

        # Use the variable name as a fallback for the colorbar label
        final_cblabel = cblabel or variable
        plt.colorbar(mesh, label=final_cblabel, ax=ax)

        add_features(ax) # Assumes this function is defined elsewhere

        return ax


def subset_to_extent(lon, lat, values, extent):
    xx = (lon >= extent[0]) & (lon <= extent[1])
    yy = (lat >= extent[2]) & (lat <= extent[3])
    # assumes that we read values from a file that put lat first
    return values[np.ix_(yy, xx)]


def subset_dict_to_extent(data: dict, extent):
    xx = (data['lon'] >= extent[0]) & (data['lon'] <= extent[1])
    yy = (data['lat'] >= extent[2]) & (data['lat'] <= extent[3])
    data['lon'] = data['lon'][xx]
    data['lat'] = data['lat'][yy]
    data['values'] = data['values'][np.ix_(yy, xx)]
    return data



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
