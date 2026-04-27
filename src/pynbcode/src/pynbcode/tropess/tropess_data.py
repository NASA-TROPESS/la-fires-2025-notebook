from datetime import datetime
from netCDF4 import Dataset
import os
import numpy as np
from pathlib import Path

TROPESS_COL_INDS = {'total': 0, 'trop': 1, 'ut': 2, 'lt': 3, 'strat': 4}
MOLEC_CM2_TO_MOL_M2 = 1e4 / 6.022e23

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
