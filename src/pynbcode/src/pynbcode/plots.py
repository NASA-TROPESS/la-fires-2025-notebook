import cartopy.crs as ccrs
from loguru import logger
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import re
from shapely.geometry import Point, Polygon

import jllutils.plots as jplt
import jllutils.stats as jstats

from .enhancement_ratios import load_enhancements_for_date
from .er_frp_analysis import FrpProvider
from .plot_utils import add_features
# Import for convenience re-export
from .frp.frp_plots import *
from .tropess.tropess_plots import *
from .utils import PlumeMask

from typing import Optional, Tuple, Union


def plot_gridded_data(
    grid_file: str,
    variable: str,
    level_idx: Optional[int] = None,
    vscale: float = 1.0,
    cblabel: str = "",
    extent: Optional[Tuple[float, float, float, float]] = None,
    plume_mask: Optional[PlumeMask] = None,
    feature_kws: dict = dict(),
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

        mesh = ax.pcolormesh(lon, lat, values, **pcolormesh_kw)

        # Use the variable name as a fallback for the colorbar label
        final_cblabel = cblabel or variable
        extend = jplt.cbextend(values, vmin=pcolormesh_kw.get('vmin', None), vmax=pcolormesh_kw.get('vmax', None))
        plt.colorbar(mesh, label=final_cblabel, extend=extend, ax=ax)

        add_features(ax, **feature_kws)

        return ax



def plot_gridded_enhancement(
    grid_file: str,
    date: str | pd.Timestamp,
    variable: str,
    vscale: float = 1.0,
    cblabel: str = '{var} ({units})',
    extent: Optional[Tuple[float, float, float, float]] = None,
    plume_mask: Optional['PlumeMask'] = None,
    feature_kws: dict = dict(),
    show_mean_er: bool = True,
    mean_er_text_kw: Optional[dict] = None, 
    ax: Optional[plt.Axes] = None,
    **pcolormesh_kw
):

    enh_dict = load_enhancements_for_date(grid_file=grid_file, date=date, variable=variable, plume_mask=plume_mask, extent=extent)
    enh_dict['values'] = vscale * enh_dict['values']
    units = _units_mathmode(enh_dict['units'])

    if ax is None:
        _, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

    mesh = ax.pcolormesh(enh_dict['lon'], enh_dict['lat'], enh_dict['values'], **pcolormesh_kw)

    if show_mean_er:
        mean_er = np.nanmean(enh_dict['values'])
        std_er = np.nanstd(enh_dict['values'], ddof=1)
        mean_er_text_kw = mean_er_text_kw or dict()
        mean_er_text_kw.setdefault('x', 0.98)
        mean_er_text_kw.setdefault('y', 0.96)
        mean_er_text_kw.setdefault('ha', 'right')
        mean_er_text_kw.setdefault('va', 'top')
        mean_er_text_kw.setdefault('transform', ax.transAxes)
        mean_er_text_kw.setdefault('backgroundcolor', (1.0, 1.0, 1.0, 0.75))
        ax.text(s=f'Mean ER: {mean_er:.1f} $\\pm$ {std_er:.1f} {units}', **mean_er_text_kw)

    final_cblabel = cblabel.format(var=variable, units=units)
    extend = jplt.cbextend(enh_dict['values'], vmin=pcolormesh_kw.get('vmin', None), vmax=pcolormesh_kw.get('vmax', None))
    plt.colorbar(mesh, label=final_cblabel, extend=extend, ax=ax)

    add_features(ax, **feature_kws)

    return ax



def binned_box_plots(df: pd.DataFrame, xcol: str, ycol: str, bin_edges: Union[int, float, np.ndarray], add_hexbin: bool = False, add_points: bool = False,
                     fit_values: bool = True, xerr_col: Optional[str] = None, fit_bins: bool = False, yerr_col: Optional[str] = None,
                     use_point=False, bin_plot_kws=None, hexbin_kws=None, point_kws=None, value_fit_plot_kws=None, bin_fit_plot_kws=None, set_bin_xticks=True, fit_model='sk',
                     fit_x=None, ax=None):
    if isinstance(bin_edges, (int, float)):
        top_edge = np.floor(df[xcol].max() / bin_edges + 2) * bin_edges
        bin_edges = np.arange(0, top_edge, bin_edges)

    bin_x = []
    bin_y = []
    for edge, bin_df in df.groupby(pd.cut(df[xcol], bin_edges), observed=True):
        bin_x.append(edge)
        bin_y.append(bin_df[ycol].to_numpy())

    x = np.array([0.5*(b.left + b.right) for b in bin_x])
    if fit_x is None:
        fit_x = np.array([0.5*(bin_edges[i] + bin_edges[i+1]) for i in range(np.size(bin_edges)-1)])
    width = np.mean([0.67*(b.right - b.left) for b in bin_x])
    bin_meds = np.array([np.median(y) for y in bin_y])
    bin_stds = np.array([np.std(y, ddof=1) for y in bin_y])

    ax = ax or plt.gca()
    bin_plot_kws = dict() if bin_plot_kws is None else bin_plot_kws.copy()
    if use_point:
        bin_plot_kws.setdefault('color', 'k')
        bin_plot_kws.setdefault('marker', 'o')
        bin_plot_kws.setdefault('ls', 'none')
        bin_plot_kws.setdefault('ecolor', 'k')
        ax.errorbar(x, bin_meds, yerr=bin_stds, **bin_plot_kws)
    else:
        ax.boxplot(bin_y, positions=x, widths=width, **bin_plot_kws)

    if set_bin_xticks:
        ax.set_xticks(x)
        ax.set_xticklabels(bin_x)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    if fit_values:
        xerr = df[xerr_col].to_numpy() if xerr_col else np.ones(df.shape[0])
        yerr = df[yerr_col].to_numpy() if yerr_col else np.ones(df.shape[0])
        logger.debug(f'Fitting individual values with "{fit_model}" polyfit model')
        values_fit = jstats.PolyFitModel(x=df[xcol].to_numpy(), xerr=xerr, y=df[ycol].to_numpy(), yerr=yerr, model=fit_model, nans='drop')
        value_fit_plot_kws = dict() if value_fit_plot_kws is None else value_fit_plot_kws
        value_fit_plot_kws.setdefault('color', 'tab:red')
        value_fit_plot_kws.setdefault('label', f'Fit to indiv. points: ${_to_sci_notation(values_fit.yint)} + {_to_sci_notation(values_fit.slope)}x$')
        fit_y = values_fit.predict(fit_x)
        ax.plot(fit_x, fit_y, **value_fit_plot_kws)
    if fit_bins:
        logger.debug(f'Fitting bins with "{fit_model}" polyfit model')
        bins_fit = jstats.PolyFitModel(x=x, xerr=np.ones(x.size), y=bin_meds, yerr=bin_stds, model=fit_model, nans='drop')
        bin_fit_plot_kws = dict() if bin_fit_plot_kws is None else bin_fit_plot_kws
        bin_fit_plot_kws.setdefault('color', 'tab:orange')
        bin_fit_plot_kws.setdefault('label', f'Fit to bin medians: ${_to_sci_notation(bins_fit.yint)} + {_to_sci_notation(bins_fit.slope)}x$')
        fit_y = bins_fit.predict(fit_x)
        ax.plot(fit_x, fit_y, **bin_fit_plot_kws)
    ax.grid(True)

    if add_hexbin:
        hexbin_kws = dict() if hexbin_kws is None else hexbin_kws.copy()
        hexbin_kws.setdefault('mincnt', 1)
        hexbin_kws.setdefault('gridsize', 30)
        hexbin_kws.setdefault('bins', 'log')
        hexbin_kws.setdefault('cmap', 'pink_r')
        h = ax.hexbin(df[xcol].to_numpy(), df[ycol].to_numpy(), **hexbin_kws)
        plt.colorbar(h, label='# points')
    if add_points:
        point_kws = dict() if point_kws is None else point_kws.copy()
        point_kws.setdefault('color', 'royalblue')
        point_kws.setdefault('ecolor', 'lightskyblue')
        point_kws.setdefault('ls', 'none')
        point_kws.setdefault('marker', '.')
        point_kws.setdefault('zorder', 1.5)
        point_kws.setdefault('alpha', 0.25)
        ax.errorbar(x=df[xcol].to_numpy(), y=df[ycol].to_numpy(), yerr=df[yerr_col].to_numpy(), **point_kws)


    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    return bin_edges



def er_vs_frp_box_plots(fire_df: pd.DataFrame, bin_edges: Union[int, float, np.ndarray] = 25, frp_column: str = 'mean_frp', show_anderson_fit=True, ax=None, **inner_kws):
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
    ax = ax or plt.gca()
    bin_edges = binned_box_plots(
        df=fire_df,
        xcol=frp_column,
        ycol='er',
        yerr_col='er_unc',
        bin_edges=bin_edges,
        ax=ax,
        **inner_kws
    )

    if show_anderson_fit:
        fit_x = inner_kws.get('fit_x', bin_edges)
        ax.plot(fit_x, 1.04e-5*fit_x + 1.58e-4, color='r', ls=':', label='Fit from Anderson et al.: $(1.58 \\times 10^{-4}) + (1.04 \\times 10^{-5}) x$')

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))
    ax.set_xlabel('FRP bin (MW)')
    ax.set_ylabel('NO$_2$:CO enhancement ratio (mol mol$^{-1}$)')


def fire_fit_plot(fire_df, frp_col: str, er_col: str = 'er', er_unc_col: str = 'er_unc', fit_x=None, point_kws=None, value_fit_plot_kws=None, ax=None):
    ax = ax or plt.gca()
    x = fire_df[frp_col].to_numpy()
    y = fire_df[er_col].to_numpy()
    yerr = fire_df[er_unc_col].to_numpy()

    point_kws = dict() if point_kws is None else point_kws.copy()
    point_kws.setdefault('color', 'royalblue')
    point_kws.setdefault('ecolor', 'lightskyblue')
    point_kws.setdefault('ls', 'none')
    point_kws.setdefault('marker', '.')
    point_kws.setdefault('zorder', 1.5)
    point_kws.setdefault('alpha', 0.25)
    ax.errorbar(x=x, y=y, yerr=yerr, **point_kws)

    xlabel = frp_col.replace('_', ' ').capitalize().replace('frp', 'FRP')
    ax.set_xlabel(f'{xlabel} (MW)')
    ax.set_ylabel('NO$_2$:CO enhancement ratio (mol/mol)')

    values_fit = jstats.PolyFitModel(x=x, y=y, yerr=yerr, model='sk', nans='drop')
    value_fit_plot_kws = dict() if value_fit_plot_kws is None else value_fit_plot_kws
    value_fit_plot_kws.setdefault('color', 'tab:red')
    value_fit_plot_kws.setdefault('label', f'Fit: ${_to_sci_notation(values_fit.yint)} + {_to_sci_notation(values_fit.slope)}x$')
    if fit_x is None:
        x1, x2 = ax.get_xlim()
        fit_x = np.linspace(x1, x2)
    fit_y = values_fit.predict(fit_x)
    ax.plot(fit_x, fit_y, **value_fit_plot_kws)
    ax.grid(True)


def _to_sci_notation(v, decimals=2, exp_cutoff=2) -> str:
    ex = int(np.floor(np.log10(np.abs(v))))
    if np.abs(ex) <= exp_cutoff:
        return str(np.round(v, decimals=decimals))
    mantissa = np.round(v / 10**ex, decimals=decimals)
    # This is used in a format call, so the curly braces need escaped twice
    return f'({mantissa} \\times 10^{{{{{ex}}}}})'


def _units_mathmode(units):
    return re.sub(r'\^([+\-]?\d+)', r'$^{\1}$', units)


def plot_frp_timeseries(
    provider: FrpProvider,
    tz: str = '-08:00',
    time_bin: str = 'h',
    bounds: Polygon | None = None,
    min_time: pd.Timestamp | str | None = None,
    max_time: pd.Timestamp | str | None = None,
    ax: plt.Axes = None,
    **plot_kws
):
    fires_df = provider.detection_df()
    local_times = fires_df.datetime.dt.tz_convert(tz).dt.tz_localize(None)
    xx = np.ones(fires_df.shape[0], dtype=bool)
    if bounds is not None:
        xx &= fires_df.apply(lambda r: bounds.contains(Point(r.longitude, r.latitude)), axis=1)
    if min_time is not None:
        xx &= local_times >= min_time
    if max_time is not None:
        xx &= local_times <= max_time
    fires_df = fires_df[xx]


    local_time_bins = local_times.dt.round(time_bin)
    mean_df = fires_df.groupby(local_time_bins)[['frp']].sum()

    if ax is None:
        ax = plt.gca()

    plot_kws.setdefault('marker', 'o')
    plot_kws.setdefault('color', 'black')
    h = ax.plot(mean_df.index, mean_df.frp, **plot_kws)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M %b %d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.grid(True)
    ax.set_ylabel('Summed FRP (MW)')
    return h
