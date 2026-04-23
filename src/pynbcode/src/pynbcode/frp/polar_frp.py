from pyhdf.SD import SD
import pandas as pd
from pathlib import Path

from ..utils import ConfidenceLevel


def assign_overpass_index(df: pd.DataFrame, time_column: str ='datetime'):
    df = df.sort_values(time_column).reset_index(drop=True)
    df['overpass_index'] = -1
    opi = 0
    for irow, row in df.iterrows():
        if irow > 0 and (row[time_column] - df.loc[irow-1,time_column]) > pd.Timedelta(6, unit='h'):
            opi += 1
        df.loc[irow,'overpass_index'] = opi
    return df


def group_modis_granules_by_overpass(files: list[Path]):
    # According to https://www.earthdata.nasa.gov/data/instruments/modis,
    # the names follow M?D14.Ayyyymmdd.hhmm.vvv.*.hdf with yyyymmdd being
    # the acquisition date and hhmm the acquisition time - presumably in UTC.
    file_list = []
    for file in files:
        datestr, timestr = file.name.split('.')[1:3]
        dtstr = f'{datestr[1:]}{timestr}+0000'
        dt = pd.to_datetime(dtstr, format='%Y%j%H%M%z')
        file_list.append({'file': file, 'datetime': dt, 'overpass_index': -999})

    file_df = pd.DataFrame(file_list)
    return assign_overpass_index(file_df)


def load_modis_fires_by_overpass(file_df: list[Path] | pd.DataFrame):
    if not isinstance(file_df, pd.DataFrame):
        file_df = group_modis_granules_by_overpass(file_df)

    fire_dfs = []
    for _, row in file_df.iterrows():
        dset = SD(str(row.file))

        # Check if there were any active fires
        dimnames, dimlens, _, _ = dset.datasets()['FP_confidence']
        dims = {k: v for k, v in zip(dimnames, dimlens)}
        if dims['number_of_active_fires'] > 0:
            lon = dset.select('FP_longitude').get()
            lat = dset.select('FP_latitude').get()
            conf = dset.select('FP_confidence').get()
            conf_level = [_modis_percent_to_conf_level(c) for c in conf]
            fire_dfs.append(pd.DataFrame({
                'acquisition_time': row.datetime,
                'lon': lon,
                'lat': lat,
                'confidence_percent': conf,
                'confidence_level': conf_level,
                'overpass_index': row.overpass_index
            }))
        else:
            print(f'{row.file} has no fires')

    return pd.concat(fire_dfs)


def _modis_percent_to_conf_level(percent: float) -> ConfidenceLevel:
    if 0 <= percent < 30:
        return ConfidenceLevel.LOW
    if 30 <= percent < 80:
        return ConfidenceLevel.NOMINAL
    if 80 <= percent <= 100:
        return ConfidenceLevel.HIGH
    raise ValueError(f'confidence percent "{percent}" out of range')
