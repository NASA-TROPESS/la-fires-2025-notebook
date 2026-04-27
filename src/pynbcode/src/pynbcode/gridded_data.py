from netCDF4 import Dataset
import os
import pandas as pd
from pathlib import Path

from .utils import PlumeMask, subset_dict_to_extent

from typing import Tuple


def load_variable_for_date(
    quantity: str,
    scene: str,
    date: pd.Timestamp | str,
    variable: str,
    data_dir: os.PathLike = Path('static-data/tropomi/gridded/'),
    plume_mask: PlumeMask | None = None,
    extent: Tuple[float, float, float, float] | None = None,
) -> dict:
    date = pd.to_datetime(date)
    file = Path(data_dir) / quantity / f'gridded_{quantity}_{scene}_{date:%Y%m%d}.nc'
    with Dataset(file) as ds:
        data = {
            'lon': ds['grid_lon'][:],
            'lat': ds['grid_lat'][:],
            'values': ds[variable][:],
        }

        if plume_mask is not None:
            plume_mask.apply_mask(
                values=data['values'],
                val_lon=data['lon'],
                val_lat=data['lat'],
                val_date=date,
            )

        if extent is not None:
            data = subset_dict_to_extent(data, extent)

        return data
