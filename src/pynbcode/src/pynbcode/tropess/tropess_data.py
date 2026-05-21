from datetime import datetime
from netCDF4 import Dataset
import os
import numpy as np
from pathlib import Path

TROPESS_COL_INDS = {'total': 0, 'trop': 1, 'ut': 2, 'lt': 3, 'strat': 4}

def load_muses_cols(muses_root_dir: os.PathLike, species: str, date: datetime, step: int = 0) -> dict[str, np.ma.MaskedArray]:
    file = Path(muses_root_dir) / f'{date:%Y}' / f'{date:%m}' / f'{date:%d}' / 'L2_Products_Lite' / f'CRIS-JPSS-1_L2-{species}-{step}_{date:%Y_%m_%d}_F02_1.23_Litev01_Day_Night.nc'
    with Dataset(file) as ds:
        return {
            'lat': ds['Latitude'][:],
            'lon': ds['Longitude'][:],
            'quality': ds['Quality'][:],
            'land': ds['LandFlag'][:],
            'day': ds['DayNightFlag'][:],
            'x': ds['Species'][:],
            'p': ds['Pressure'][:],
            'columns': ds['Retrieval/Column'][:,:],
            'air_columns': ds['Retrieval/Column_Air'][:,:],
            'aks': ds['AveragingKernel'][:,:,:]
        }


def load_tropess_cols(tropess_dir: os.PathLike, species: str, date: datetime) -> dict[str, np.ma.MaskedArray]:
    pattern = f'TROPESS_*_{species}_{date:%Y%m%d}*.nc'
    files = list(Path(tropess_dir).glob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(f'Found {len(files)} matching "{pattern}" in {tropess_dir}: {files}')
    file = files[0]
    with Dataset(file) as ds:
        out_data = {
            'lat': ds['latitude'][:],
            'lon': ds['longitude'][:],
            'quality': np.ma.masked_all(ds.dimensions['target'].size),
            'day': ds['geophysical/day_night_flag'][:],
            # For this work, I only use total and tropospheric columns, so I'm not going to mess
            # with the other three.
            'columns': np.ma.masked_all([ds.dimensions['target'].size, 2]),
            'air_columns': np.ma.masked_all([ds.dimensions['target'].size, 2]),
        }

        if 'x' in ds.variables.keys():
            out_data['x'] = ds['x'][:]
            out_data['p'] = ds['pressure'][:]

        out_data['air_columns'][:,0] = ds['col_dry_air'][:]
        if 'col' in ds.variables.keys():
            out_data['columns'][:,0] = ds['col'][:]
        if 'col_t' in ds.variables.keys():
            out_data['columns'][:,1] = ds['col_t'][:]

            # We don't seem to include partial columns of air, so we'd need to
            # weight the total air column by the fraction of air in the subcolumn
            trop_pres = ds['trop_pres'][:]
            surf_pres = ds['surf_pres'][:]
            trop_frac = (surf_pres - trop_pres) / surf_pres
            out_data['air_columns'][:,1] = out_data['air_columns'][:,0] * trop_frac

    return out_data


def get_muses_col_values(data, column: str, col_avg: bool):
    icol = TROPESS_COL_INDS[column]

    cols = data['columns'][:, icol]
    if col_avg:
        cols = cols / data['air_columns'][:, icol]

    cols = cols.filled(np.nan)
    cols[cols < -990.0] = np.nan
    return cols


def make_filter_index(data: dict, extents: list[float], good_only: bool, day_only: bool) -> np.ndarray:
    x1, x2, y1, y2 = extents
    ii = (data['lon'] >= x1) & (data['lon'] <= x2) & (data['lat'] >= y1) & (data['lat'] <= y2)
    if good_only:
        ii &= data['quality'] == 1
    if day_only:
        ii &= data['day'] == 1
    return ii
