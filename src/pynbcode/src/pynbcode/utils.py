from .constants import STD_CO_ENH
from enum import IntEnum
from netCDF4 import Dataset
import numpy as np
import pandas as pd
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
