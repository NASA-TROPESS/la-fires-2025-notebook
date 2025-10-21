import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.io import shapereader
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import re

import jllutils.stats as jstats

from typing import Optional, Tuple, Union


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


def binned_box_plots(df: pd.DataFrame, xcol: str, ycol: str, bin_edges: Union[int, float, np.ndarray], add_hexbin: bool = True,
                     fit_values: bool = True, xerr_col: Optional[str] = None, fit_bins: bool = True, yerr_col: Optional[str] = None,
                     hexbin_kws=None, value_fit_plot_kws=None, bin_fit_plot_kws=None, fit_model='york', ax=None):
    if isinstance(bin_edges, (int, float)):
        top_edge = np.floor(df[xcol].max() / bin_edges + 2) * bin_edges
        bin_edges = np.arange(0, top_edge, bin_edges)

    bin_x = []
    bin_y = []
    for edge, bin_df in df.groupby(pd.cut(df[xcol], bin_edges), observed=True):
        bin_x.append(edge)
        bin_y.append(bin_df[ycol].to_numpy())

    x = np.array([0.5*(b.left + b.right) for b in bin_x])
    width = np.mean([0.67*(b.right - b.left) for b in bin_x])

    ax = ax or plt.gca()
    ax.boxplot(bin_y, positions=x, widths=width)
    ax.set_xticklabels(bin_x)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    if fit_values:
        xerr = df[xerr_col].to_numpy() if xerr_col else np.ones(df.shape[0])
        yerr = df[yerr_col].to_numpy() if yerr_col else np.ones(df.shape[0])
        values_fit = jstats.PolyFitModel(x=df[xcol].to_numpy(), xerr=xerr, y=df[ycol].to_numpy(), yerr=yerr, model=fit_model)
        value_fit_plot_kws = dict() if value_fit_plot_kws is None else value_fit_plot_kws
        value_fit_plot_kws.setdefault('color', 'tab:blue')
        value_fit_plot_kws.setdefault('label', f'York fit, indiv. days: ${_to_sci_notation(values_fit.yint)} + {_to_sci_notation(values_fit.slope)}x$')
        values_fit.plot_fit(ax, **value_fit_plot_kws)
    if fit_bins:
        bin_meds = np.array([np.median(y) for y in bin_y])
        bin_stds = np.array([np.std(y, ddof=1) for y in bin_y])
        bins_fit = jstats.PolyFitModel(x=x, xerr=np.ones(x.size), y=bin_meds, yerr=bin_stds, nans='drop', model=fit_model)
        bin_fit_plot_kws = dict() if bin_fit_plot_kws is None else bin_fit_plot_kws
        bin_fit_plot_kws.setdefault('color', 'tab:orange')
        bin_fit_plot_kws.setdefault('label', f'York fit, bin medians: ${_to_sci_notation(bins_fit.yint)} + {_to_sci_notation(bins_fit.slope)}x$')
        bins_fit.plot_fit(ax, **bin_fit_plot_kws)
    ax.grid(True)

    hexbin_kws = dict() if hexbin_kws is None else hexbin_kws
    hexbin_kws.setdefault('mincnt', 1)
    hexbin_kws.setdefault('gridsize', 30)
    hexbin_kws.setdefault('bins', 'log')
    hexbin_kws.setdefault('cmap', 'pink_r')
    h = ax.hexbin(df[xcol].to_numpy(), df[ycol].to_numpy(), **hexbin_kws)
    plt.colorbar(h)

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)



def er_vs_frp_box_plots(fire_df: pd.DataFrame, bin_edges: Union[int, float, np.ndarray] = 25, frp_column: str = 'mean_frp', show_anderson_fit=True, ax=None):
    """Plot NO2:CO enhancement ratios vs. fire radiative power as a series of boxplots

    This aims to reproduce Anderson et al. (2023) Fig. 3c (doi.org/10.1029/2023GL105811).

    Parameters
    ----------
    fire_df:
        A dataframe containing daily values of FRP in the column defined by ``frp_column`` and enhancement
        ratios in the column named "er". Usually this will be one returned by
        :func:`pynbcode.er_frp_analysis.match_frps_to_ers_many_files`.

    bin_edges
        Used to define the FRP bins. If an integer or float, then this defines the WIDTH of the FRP bins,
        not the number as is typical for binning functions. In this case, the first bin will have
        a lower edge of 0 and the last bin will be set to include the maximum FRP value. If this is
        an array instead, it must define the actual bin edges.

    frp_column
        Which column in the dataframe to use for the FRP values.

    show_anderson_fit
        By default, the fit reported by Anderson et al. (2023) is plotted for comparison. Set this keyword
        to ``False`` to hide that.

    ax
        Axes to plot into

    """
    if isinstance(bin_edges, (int, float)):
        top_edge = np.floor(fire_df[frp_column].max() / bin_edges + 2) * bin_edges
        bin_edges = np.arange(0, top_edge, bin_edges)

    bin_x = []
    bin_y = []
    for edge, bin_df in fire_df.groupby(pd.cut(fire_df[frp_column], bin_edges), observed=True):
        bin_x.append(edge)
        bin_y.append(bin_df.er.to_numpy())

    x = np.array([0.5*(b.left + b.right) for b in bin_x])
    bin_meds = np.array([np.median(y) for y in bin_y])
    bin_stds = np.array([np.std(y, ddof=1) for y in bin_y])
    fit_days = jstats.PolyFitModel(x=fire_df[frp_column].to_numpy(), xerr=np.ones(fire_df.shape[0]), y=fire_df.er.to_numpy(), yerr=fire_df.er_unc.to_numpy())
    fit_bins = jstats.PolyFitModel(x=x, xerr=np.ones(x.size), y=bin_meds, yerr=bin_stds, nans='drop')

    width = np.mean([0.67*(b.right - b.left) for b in bin_x])

    ax = ax or plt.gca()
    ax.boxplot(bin_y, positions=x, widths=width)
    ax.set_xticklabels(bin_x)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    fit_days.plot_fit(ax, color='tab:blue', label=f'York fit, indiv. days: ${_to_sci_notation(fit_days.yint)} + {_to_sci_notation(fit_days.slope)}x$')
    fit_bins.plot_fit(ax, color='tab:orange', label=f'York fit, bin medians: ${_to_sci_notation(fit_bins.yint)} + {_to_sci_notation(fit_bins.slope)}x$')
    ax.grid(True)

    h = ax.hexbin(fire_df[frp_column].to_numpy(), fire_df.er.to_numpy(), mincnt=1, gridsize=30, bins='log', cmap='pink_r')
    plt.colorbar(h, label='# points')

    if show_anderson_fit:
        ax.plot(bin_edges, 1.04e-5*bin_edges + 1.58e-4, color='r', ls=':', label='Fit from Anderson et al.: $(1.58 \\times 10^{-4}) + (1.04 \\times 10^{-5}) x$')

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))
    ax.set_xlabel('FRP bin (MW)')
    ax.set_ylabel('NO$_2$:CO enhancement ratio (mol mol$^{-1}$)')


def _to_sci_notation(v, decimals=2, exp_cutoff=2) -> str:
    ex = int(np.floor(np.log10(np.abs(v))))
    if np.abs(ex) <= exp_cutoff:
        return str(np.round(v, decimals=decimals))
    mantissa = np.round(v / 10**ex, decimals=decimals)
    # This is used in a format call, so the curly braces need escaped twice
    return f'({mantissa} \\times 10^{{{{{ex}}}}})'


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
