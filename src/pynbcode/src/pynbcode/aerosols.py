import os
import pandas as pd

from .constants import STATIC_DATA_DIR
from .gridded_data import load_variable_for_date
from .utils import PlumeMask, subset_dict_to_extent

from typing import Tuple

def load_aerosol_variable_for_date(
    quantity: str,
    scene: str,
    date: pd.Timestamp | str,
    variable: str,
    data_dir: os.PathLike = STATIC_DATA_DIR / 'tropomi' / 'gridded',
    plume_mask: PlumeMask | None = None,
    extent: Tuple[float, float, float, float] | None = None,
    wavelength_index: int = 2, # index = 2 is the 500 nm values
) -> dict:
    data = load_variable_for_date(
        quantity=quantity,
        scene=scene,
        date=date,
        variable=variable,
        data_dir=data_dir,
        plume_mask=None,
        extent=None,
    )
    if data['values'].ndim == 3:
        data['values'] = data['values'][wavelength_index]
    elif data['values'].ndim != 2:
        raise NotImplementedError(f'Expected aerosol value array to be 2D or 3D, got {data["values"].ndim}D')

    # This must be done after we reduce the variable to 2D
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
