import cartopy.crs as ccrs
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

from .tropess_data import load_muses_cols, TROPESS_COL_INDS
from ..plot_utils import add_features

__all__ = ['plot_muses_columns']

def plot_muses_columns(
    muses_root: os.PathLike,
    species: str,
    date: datetime,
    extents: list[float],
    column: str = 'total',
    good_only: bool = False,
    day_only: bool = True,
    col_avg: bool = False,
    step: int = 0,
    vscale: float = 1.0,
    cblabel: str ='',
    ax=None,
    **scatter_kws,
) -> None:
    data = load_muses_cols(muses_root, species, date, step=step)
    ii = _make_filter_index(data, extents=extents, good_only=good_only, day_only=day_only)

    icol = TROPESS_COL_INDS[column]
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    cols = data['columns'][:, icol]
    if col_avg:
        cols = cols / data['air_columns'][:, icol]

    scatter_kws.setdefault('s', 384)
    scatter_kws.setdefault('edgecolor', 'black')
    h = ax.scatter(data['lon'][ii], data['lat'][ii], c=cols[ii] * vscale, **scatter_kws)
    plt.colorbar(h, ax=ax, label=cblabel)
    add_features(ax, land=True, ocean=True)


def _make_filter_index(data: dict, extents: list[float], good_only: bool, day_only: bool) -> np.ndarray:
    x1, x2, y1, y2 = extents
    ii = (data['lon'] >= x1) & (data['lon'] <= x2) & (data['lat'] >= y1) & (data['lat'] <= y2)
    if good_only:
        ii &= data['quality'] == 1
    if day_only:
        ii &= data['day'] == 1
    return ii
