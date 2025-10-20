import jllutils.stats as jstats
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os
import pandas as pd

from typing import Sequence, Union


def load_enhancments(nc_file: os.PathLike) -> dict:
    """Load an enhancement file into a dictionary
    """
    with Dataset(nc_file) as ds:
        out_dict = dict()
        for (varname, var) in ds.variables.items():
            out_dict[varname] = var[:]

        out_dict['enhancements'] = dict()
        for (grpname, grp) in ds.groups.items():
            subdata = dict()
            for (varname, var) in grp.variables.items():
                subdata[varname] = var[:]
            out_dict['enhancements'][grpname.lower()] = subdata
    out_dict['date'] = pd.to_datetime(out_dict['date'])
    return out_dict



def calculate_er_as_fit(enh_dict: dict, subscene_extent: Sequence[float], idate: Union[int, pd.Timestamp, str], min_co_enh=None, enh_group='time', model='robust', debug_plot=False):
    if not isinstance(idate, int):
        idate = np.flatnonzero(enh_dict['date'] == idate).item()

    xx_lon = (enh_dict['grid_lon'] >= subscene_extent[0]) & (enh_dict['grid_lon'] <= subscene_extent[1])
    xx_lat = (enh_dict['grid_lat'] >= subscene_extent[2]) & (enh_dict['grid_lat'] <= subscene_extent[3])
    co = enh_dict['co_vcds'][idate][np.ix_(xx_lat, xx_lon)].filled(np.nan).ravel()
    no2 = enh_dict['no2_trop_vcds'][idate][np.ix_(xx_lat, xx_lon)].filled(np.nan).ravel()

    xx = np.isfinite(co) & np.isfinite(no2)
    if min_co_enh is not None:
        delta_co = enh_dict['enhancements'][enh_group]['co_enhancement'][idate][np.ix_(xx_lat, xx_lon)].filled(np.nan).ravel()
        xx = xx & (delta_co > min_co_enh)

    if np.sum(xx) == 0:
        return {'er': np.nan, 'er_unc': np.nan, 'intercept': np.nan, 'intercept_unc': np.nan}

    fit = jstats.PolyFitModel(x=co[xx], y=no2[xx], model=model)

    if debug_plot:
        _, ax = plt.subplots()
        ax.plot(co[xx], no2[xx], ls='none', marker='.', color='k')
        fit.plot_fit(ax, ls='--', color='r')
        ax.legend()

    return {'er': fit.slope, 'er_unc': fit.coeff_errors[1], 'intercept': fit.yint, 'intercept_unc': fit.coeff_errors[0]}
