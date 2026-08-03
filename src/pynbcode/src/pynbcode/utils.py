from .constants import STD_CO_ENH, STATIC_DATA_DIR
from enum import IntEnum
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import re
from shapely.geometry import Point, Polygon

from typing import Tuple

class ConfidenceLevel(IntEnum):
    """Represents a confidence level for fire data.

    Usually nominal or high confidence data are useful.
    """
    LOW = 0
    NOMINAL = 1
    HIGH = 2

    @classmethod
    def from_char(cls, c):
        if c in 'lL':
            return cls.LOW
        if c in 'nN':
            return cls.NOMINAL
        if c in 'hH':
            return cls.HIGH

        raise ValueError(f'unknown character for fire confidence: {c}')


class PlumeMask:
    def __init__(self, dates, grid_lon, grid_lat, mask) -> None:
        self.dates = dates
        self.lon = grid_lon
        self.lat = grid_lat
        self.mask = mask

    @classmethod
    def from_enhancement(cls, enhancement_file: str, enhancement_group: str = 'Time', co_enhancement_threshold: float = STD_CO_ENH):
        with Dataset(enhancement_file) as ds:
            dates = pd.to_datetime(ds['date'][:])
            lon = ds['grid_lon'][:]
            lat = ds['grid_lat'][:]
            co_enh = ds[enhancement_group]['co_enhancement'][:]
            mask = co_enh < co_enhancement_threshold
            return cls(dates=dates, grid_lon=lon, grid_lat=lat, mask=mask)

    @classmethod
    def from_floodfill(cls, scene_name: str, subscene_name: str, plume_dir: Path = STATIC_DATA_DIR / 'tropomi' / 'plume'):
        plume_files = sorted(plume_dir.glob(f'plumeinfo_{scene_name}_{subscene_name}_????????.nc'))
        dates = []
        lon = None
        lat = None
        mask = []

        for file in plume_files:
            with Dataset(file) as ds:
                this_date = pd.to_datetime(re.search(r'_(\d{8}).nc', file.name).group(1))
                this_lon = ds['grid_lon'][:]
                this_lat = ds['grid_lat'][:]
                this_mask = ds['mask'][:].filled(0).astype(bool)

            if lon is None:
                lon = this_lon
            elif not np.allclose(lon, this_lon):
                raise ValueError('longitudes are inconsistent across files')

            if lat is None:
                lat = this_lat
            elif not np.allclose(lat, this_lat):
                raise ValueError('latitudes are inconsistent across files')

            dates.append(this_date)
            # The floodfill masks are 1 in the plume, we need 1 out of the plume
            mask.append(~this_mask[np.newaxis])

        dates = pd.DatetimeIndex(dates)
        mask = np.concatenate(mask, axis=0)
        return cls(dates=dates, grid_lon=lon, grid_lat=lat, mask=mask)


    def apply_mask(self, values, val_lon, val_lat, val_date, fill_val=np.nan):
        ix1 = np.flatnonzero(np.isclose(self.lon[0], val_lon)).item()
        ix2 = ix1 + self.lon.size
        iy1 = np.flatnonzero(np.isclose(self.lat[0], val_lat)).item()
        iy2 = iy1 + self.lat.size
        if not np.allclose(self.lon, val_lon[ix1:ix2]) or not np.allclose(self.lat, val_lat[iy1:iy2]):
            raise NotImplementedError('Could not match mask lat/lon to a subset of the value lat/lon')

        idate = np.flatnonzero(self.dates == val_date)[0]
        mask_arr = np.ones(values.shape, dtype=bool)
        mask_arr[iy1:iy2, ix1:ix2] = self.mask[idate, :, :]
        values[mask_arr] = fill_val


def extents_to_polygon(extents: Tuple[float, float, float, float]) -> Polygon:
    x1, x2, y1, y2 = extents
    return Polygon([
        (x1, y1),
        (x1, y2),
        (x2, y2),
        (x2, y1),
        (x1, y1)
    ])


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


def in_poly(lon, lat, poly: Polygon):
    is_in = np.zeros(lon.shape, dtype=bool)
    for i, (x, y) in enumerate(zip(lon, lat)):
        is_in[i] = poly.contains(Point(x, y))
    return is_in


def bbox_from_width_and_angle(base_lat, base_lon, angle, width, depth, as_poly=False):
    coords = np.array([[0.0, depth, depth, 0.0], [width/2.0, width/2.0, -width/2.0, -width/2.0]])
    a = np.deg2rad(angle)
    r_mat = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    coords = r_mat @ coords
    loncorn = base_lon + coords[0,:]
    latcorn = base_lat + coords[1,:]
    if as_poly:
        return Polygon([(x, y) for x, y in zip(loncorn, latcorn)])
    else:
        return loncorn, latcorn


def default_upwind_box(as_poly=False):
    return bbox_from_width_and_angle(base_lat=34.3, base_lon=-117.9, angle=30.0, width=0.5, depth=0.75, as_poly=as_poly)


def iter_subscene_dates(scenes: list, from_fire_start: bool, verbose: bool = False):
    nscene = len(scenes)
    for iscene, scene in enumerate(scenes, start=1):
        nsub = len(scene['subscene'])
        for isub, subscene in enumerate(scene['subscene'], start=1):
            start = subscene['fire_start'] if from_fire_start else scene['start']
            dates = pd.date_range(start, scene['end'])
            ndate = len(dates)
            for idate, date in enumerate(dates, start=1):
                if verbose:
                    print(f'Now on date {date:%Y-%m-%d} ({idate}/{ndate}) from subscene {subscene["name"]} ({isub}/{nsub}) from scene {scene["name"]} ({iscene}/{nscene})')
                yield scene['name'], subscene['name'], date, subscene['extent']
