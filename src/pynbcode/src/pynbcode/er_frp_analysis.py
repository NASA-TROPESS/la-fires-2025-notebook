from abc import ABC, abstractmethod
import json
from loguru import logger
from netCDF4 import Dataset
import numpy as np
import os
import pandas as pd
from pathlib import Path
import xarray as xr

from . import enhancement_ratios
from .constants import STD_CO_ENH
from .frp import polar_frp
from .utils import ConfidenceLevel


from typing import Sequence, Tuple, Union




class FrpMultiSceneProvider(ABC):
    """Base class for providers of fire radiative power for a collection of subscenes
    """
    @abstractmethod
    def daily_frps_for_subscene(self, scene_name: str, subscene: dict, local_hour_limits: Tuple[float, float], tz='-08:00') -> pd.DataFrame:
        """Compute daily statistics for a particular fire scene and subscene

        Parameters
        ----------
        scene_name
            The name of the scene to compute daily FRP values for, as defined in the download configuration.

        subscene
            A dictionary defining the subscene within the larger scene to compute FRP values over, from the download configuration.
            This should usually have keys "fire_start" (a string defining the first date of the fire) and "extent" (a 4-tuple defining
            the min lon, max lon, min lat, and max lat for the subscene).

        local_hour_limits
            Limits on the local hours to include in the calculation. For example, ``(9, 15)`` will only include data between 9:00a and 3:00p
            local.

        tz
            Offset from UTC to use when converting UTC time to local. Used to limit by local hour and group by date.

        Returns
        -------
        pd.DataFrame
            A dataframe, indexed by date, with FRP statistics for this scene/subscene. The specific statistics may vary
            depending on the concrete implementation, but it is recommened to include columns "mean_frp" as the average
            of FRP data within the subscene, "total_frp" as the sum of FRP data in the subscene, and "n_frp" as the number
            of FRP observations within the subscene.
        """
        pass

class FrpProvider(ABC):
    """Base class for providers of fire radiative power for a single subscene
    """
    @abstractmethod
    def daily_frps(self, subscene_extent: Tuple[float, float, float, float], local_hour_limits: Tuple[float, float], tz='-08:00') -> pd.DataFrame:
        """
        subscene_extent
            A 4-tuple defining the min lon, max lon, min lat, and max lat for the subscene.

        local_hour_limits
            Limits on the local hours to include in the calculation. For example, ``(9, 15)`` will only include data between 9:00a and 3:00p
            local.

        tz
            Offset from UTC to use when converting UTC time to local. Used to limit by local hour and group by date.

        Returns
        -------
        pd.DataFrame
            A dataframe, indexed by date, with FRP statistics for this scene/subscene. The specific statistics may vary
            depending on the concrete implementation, but it is recommened to include columns "mean_frp" as the average
            of FRP data within the subscene, "total_frp" as the sum of FRP data in the subscene, "mean_areal_frp"/"total_areal_frp"
            as the previous two normalized by pixel or fire area, and "n_frp" as the number of FRP observations within the subscene.
        """
        pass

    @abstractmethod
    def detection_df(self) -> pd.DataFrame:
        """Return a data frame with the datetime, latitude, longitude, and FRP of individual fire detections
        """
        pass


class ViirsFrpProvider(FrpProvider, FrpMultiSceneProvider):
    """An FRP provider for VIIRS data contained in a FIRMS JSON file.

    It takes a path to the JSON file to load as its single argument.
    FIRMS JSONs can be requested from https://firms.modaps.eosdis.nasa.gov/download/list.php.

    Note that this class implements both the :class:`FrpProvider` and
    :class:`FrpMultiSceneProvider` interfaces, since it assumes that the
    FRP data from multiple fires are contained in the same JSON file.
    """
    def __init__(self, json_file, min_confidence=ConfidenceLevel.NOMINAL) -> None:
        self.df = self.load_viirs_json(json_file)
        self._min_confidence_level = min_confidence

    def daily_frps(self, subscene_extent: Tuple[float, float, float, float], local_hour_limits: Tuple[float, float], tz='-08:00') -> pd.DataFrame:
        local_hours = _dti_to_local_hours(self.df.index, tz)
        xx_lon = (self.df.longitude >= subscene_extent[0]) & (self.df.longitude <= subscene_extent[1])
        xx_lat = (self.df.latitude >= subscene_extent[2]) & (self.df.latitude <= subscene_extent[3])
        xx_hr = (local_hours >= local_hour_limits[0]) & (local_hours <= local_hour_limits[1])
        xx_conf = self.df.confidence >= self._min_confidence_level

        xx = xx_lon & xx_lat & xx_hr & xx_conf
        subset_frp = self.df.loc[xx, 'frp']
        subset_area = self.df.loc[xx, 'area']
        subset_areal_frp = subset_frp / subset_area

        logger.debug(f'{(xx_lon & xx_lat).sum()/self.df.shape[0]:.2%} VIIRS obs. retained by lat/lon bounds')
        logger.debug(f'{xx_hr.sum()/self.df.shape[0]:.2%} VIIRS obs. retained by local hour bounds')
        logger.debug(f'{xx_conf.sum()/self.df.shape[0]:.2%} VIIRS obs. retained by confidence level')
        logger.debug(f'{subset_frp.size/self.df.shape[0]:.2%} VIIRS obs. retained overall')

        dates = subset_frp.index.tz_convert(tz).date
        grouped_frp = subset_frp.groupby(dates)
        grouped_areal_frp = subset_areal_frp.groupby(dates)
        return pd.DataFrame({
            'mean_frp': grouped_frp.mean(),
            'mean_areal_frp': grouped_areal_frp.mean(),
            'total_frp': grouped_frp.sum(),
            'total_areal_frp': grouped_areal_frp.sum(),
            'n_frp': grouped_frp.count()
        })

    def detection_df(self) -> pd.DataFrame:
        df = self.df.reset_index()[['datetime', 'latitude', 'longitude', 'frp']]
        return polar_frp.assign_overpass_index(df)

    def daily_frps_for_subscene(self, scene_name: str, subscene: dict, local_hour_limits: Tuple[float, float], tz='-08:00') -> pd.DataFrame:
        # Since the VIIRS JSON files aren't broken up by scenes, simply subsetting by time and location is all we need to do to 
        # get the FRPs for that subscene.
        return self.daily_frps(subscene_extent=subscene['extent'], local_hour_limits=local_hour_limits, tz=tz)

    @staticmethod
    def load_viirs_json(json_file) -> pd.DataFrame:
        with open(json_file) as f:
            json_list = json.load(f)

        for json_entry in json_list:
            datestr = json_entry.pop('acq_date')
            timestr = json_entry.pop('acq_time')
            json_entry['datetime'] = pd.to_datetime(f'{datestr} {timestr}', format='%Y-%m-%d %H%M', utc=True)
            json_entry['confidence'] = int(ConfidenceLevel.from_char(json_entry['confidence']))
            json_entry['is_day'] = json_entry.pop('daynight') == 'D'
            json_entry['area'] = json_entry['scan'] * json_entry['track']

        return pd.DataFrame(json_list).set_index('datetime')


class ViirsMultiJsonFrpProvider(ViirsFrpProvider):
    """An FRP provider for VIIRS data spread across multiple FIRMS JSON files.

    This takes paths to all of the JSON files to load as its positional arguments.
    FIRMS JSONs can be requested from https://firms.modaps.eosdis.nasa.gov/download/list.php.

    Note that this class implements both the :class:`FrpProvider` and
    :class:`FrpMultiSceneProvider` interfaces, since it assumes that the
    FRP data from multiple fires are contained in the same JSON file.

    It is important that the files passed in do not contain duplicate data;
    this class will do nothing to de-duplicate data present in multiple files.
    So, if you requested separate JSON files for individual fires that might
    overlap in time and space, passing those files to this class can easily
    result in those data being multiply counted in the output statistics.
    """
    def __init__(self, *json_files, min_confidence=ConfidenceLevel.NOMINAL) -> None:
        dfs = []
        for jf in json_files:
            dfs.append(self.load_viirs_json(jf))
        self.df = pd.concat(dfs)
        self._min_confidence_level = min_confidence


class ModisFrpProvider(FrpProvider):
    def __init__(self, source_dir: os.PathLike, min_confidence: ConfidenceLevel = ConfidenceLevel.NOMINAL):
        hdf_files = sorted(Path(source_dir).glob('*.hdf'))
        file_df = polar_frp.group_modis_granules_by_overpass(hdf_files)
        self.df = polar_frp.load_modis_fires_by_overpass(file_df)
        self.df = self.df[self.df.confidence_level >= min_confidence]

    def daily_frps(self, subscene_extent: Tuple[float, float, float, float], local_hour_limits: Tuple[float, float], tz='-08:00') -> pd.DataFrame:
        """
        subscene_extent
            A 4-tuple defining the min lon, max lon, min lat, and max lat for the subscene.

        local_hour_limits
            Limits on the local hours to include in the calculation. For example, ``(9, 15)`` will only include data between 9:00a and 3:00p
            local.

        tz
            Offset from UTC to use when converting UTC time to local. Used to limit by local hour and group by date.

        Returns
        -------
        pd.DataFrame
            A dataframe, indexed by date, with FRP statistics for this scene/subscene. The specific statistics may vary
            depending on the concrete implementation, but it is recommened to include columns "mean_frp" as the average
            of FRP data within the subscene, "total_frp" as the sum of FRP data in the subscene, "mean_areal_frp"/"total_areal_frp"
            as the previous two normalized by pixel or fire area, and "n_frp" as the number of FRP observations within the subscene.
        """
        raise NotImplementedError('Have not done the daily FRPs yet for MODIS, since it is not used in the ER:FRP analysis')

    def detection_df(self) -> pd.DataFrame:
        """Return a data frame with the datetime, latitude, longitude, and FRP of individual fire detections
        """
        df_out = self.df.rename(columns={'acquisition_time': 'datetime', 'lon': 'longitude', 'lat': 'latitude'})
        df_out = df_out[['datetime', 'longitude', 'latitude', 'overpass_index']].copy()
        df_out['frp'] = np.nan
        return df_out



class GoesFrpProvider(FrpProvider):
    def __init__(self, goes_file: os.PathLike) -> None:
        self.ds = self.load_goes(goes_file)

    @classmethod
    def from_scene(cls, scene_name, subscene_name, goes_summary_dir: os.PathLike) -> 'GoesFrpProvider':
        goes_file = Path(goes_summary_dir) / f'{scene_name}-{subscene_name}-goes_summary.nc'
        return cls(goes_file)

    def daily_frps(self, subscene_extent: Tuple[float, float, float, float], local_hour_limits: Tuple[float, float], tz='-08:00') -> pd.DataFrame:
        local_hours = _dti_to_local_hours(pd.to_datetime(self.ds.time).tz_localize('utc'), tz)
        t1, t2 = local_hour_limits
        x1, x2, y1, y2 = subscene_extent
        tt = (local_hours >= t1) & (local_hours <= t2)
        xx = (self.ds.lon >= x1) & (self.ds.lon <= x2)
        yy = (self.ds.lat >= y1) & (self.ds.lat <= y2)
        subscene_frp = self.ds.frp.sel(time=tt, pixel=xx & yy)
        subscene_area = self.ds.area.sel(time=tt, pixel=xx & yy)
        subscene_areal_frp = subscene_frp / subscene_area

        # Get the average values for the pixels for each day, then sum/average that.
        # This is more comparable to VIIRS, especially for summing, since otherwise
        # the greater number of GOES time slices will mess up the sum
        dates = pd.to_datetime(subscene_frp.time).tz_localize('utc').tz_convert('-08:00').date
        dates = xr.DataArray(dates, dims=['time'], coords=[subscene_frp.time])

        data = dict()

        if subscene_frp.size == 0:
            data['mean_frp'] = np.nan
            data['total_frp'] = np.nan
            data['n_frp'] = 0
            index = pd.to_datetime([pd.NaT])
        else:
            grouped_frp = subscene_frp.groupby(dates).mean()
            data['mean_frp'] = grouped_frp.mean(dim='pixel').to_numpy()
            data['total_frp'] = grouped_frp.sum(dim='pixel').to_numpy()
            data['n_frp'] = grouped_frp.count(dim='pixel').to_numpy()
            index = pd.to_datetime(grouped_frp.group)

        if subscene_areal_frp.size == 0:
            data['mean_areal_frp'] = np.nan
            data['total_areal_frp'] = np.nan
        else:
            grouped_areal_frp = subscene_areal_frp.groupby(dates).mean()
            data['mean_areal_frp'] = grouped_areal_frp.mean(dim='pixel').to_numpy()
            data['total_areal_frp'] = grouped_areal_frp.sum(dim='pixel').to_numpy()

        return pd.DataFrame(data, index=index)

    def detection_df(self) -> pd.DataFrame:
        times = np.broadcast_to(self.ds.time.to_numpy(), self.ds.frp.shape)
        lat = np.broadcast_to(self.ds.lat.to_numpy()[:, np.newaxis], self.ds.frp.shape)
        lon = np.broadcast_to(self.ds.lon.to_numpy()[:, np.newaxis], self.ds.frp.shape)
        frp = self.ds.frp.to_numpy()

        ii = np.isfinite(frp)
        times = pd.to_datetime(times[ii]).tz_localize('utc')
        lat = lat[ii]
        lon = lon[ii]
        frp = frp[ii]
        return pd.DataFrame({'datetime': times, 'latitude': lat, 'longitude': lon, 'frp': frp})

    @staticmethod
    def load_goes(goes_file, allowed_mask_values=np.array([10,12,13,14,30,32,33,34])):
        with xr.open_dataset(goes_file) as ds:
            lats = ds['lat']
            lons = ds['lon']
            frp = ds['Power']
            mask = ds['Mask']
            area = ds['Area'] * 1e-6  # convert from square meters to square kilometers


        # Verify that lat and lon are consistent in time
        lat_check = np.unique(lats.diff(dim='time'))
        assert np.allclose(lat_check[np.isfinite(lat_check)], 0.0)
        lon_check = np.unique(lons.diff(dim='time'))
        assert np.allclose(lon_check[np.isfinite(lon_check)], 0.0)

        lats = lats.isel(time=0)
        lons = lons.isel(time=0)

        # Remove data we aren't retaining, then limit to pixels with at least
        # one actual FRP value
        mm = np.isin(mask, allowed_mask_values)
        frp.data[~mm] = np.nan

        xx = ~(np.isnan(frp).all(dim='time'))
        frp = frp.sel(pixel=xx)
        area = area.sel(pixel=xx)
        lats = lats.sel(pixel=xx)
        lons = lons.sel(pixel=xx)

        return xr.Dataset({'frp': frp, 'area': area, 'lat': lats, 'lon': lons})


class GoesMultiSceneFrpProvider(FrpMultiSceneProvider):
    def __init__(self, goes_summary_dir: os.PathLike) -> None:
        self.goes_summary_dir = goes_summary_dir
        self.subscene_providers = dict()

    def daily_frps_for_subscene(self, scene_name: str, subscene: dict, local_hour_limits: Tuple[float, float], tz='-08:00') -> pd.DataFrame:
        provider_key = f'{scene_name}-{subscene["name"]}'
        if provider_key not in self.subscene_providers:
            self.subscene_providers[provider_key] = GoesFrpProvider.from_scene(scene_name, subscene['name'], self.goes_summary_dir)
        return self.subscene_providers[provider_key].daily_frps(
            subscene_extent=subscene['extent'],
            local_hour_limits=local_hour_limits,
            tz=tz
        )


class ErMultiSceneProvider(ABC):
    """Base class for enhancement ratio providers that span a range of subscenes
    """
    @abstractmethod
    def daily_ers_for_subscene(self, scene_name: str, subscene: dict, tz='-08:00') -> pd.DataFrame:
        """Compute daily statistics for a particular fire scene and subscene

        Parameters
        ----------
        scene_name
            The name of the scene to compute daily ER values for, as defined in the download configuration.

        subscene
            A dictionary defining the subscene within the larger scene to compute ER values over, from the download configuration.
            This should usually have keys "fire_start" (a string defining the first date of the fire) and "extent" (a 4-tuple defining
            the min lon, max lon, min lat, and max lat for the subscene).

        tz
            Offset from UTC to use when converting UTC time to local. Used to group by date.

        Returns
        -------
        pd.DataFrame
            A dataframe, indexed by date, with FRP statistics for this scene/subscene. The specific statistics may vary
            depending on the concrete implementation, but it is recommened to include columns "er" with the ER values
            and "er_unc" with the uncertainties on those values.
        """
        pass


class SlopeErMultiSceneProvider(ErMultiSceneProvider):
    """An enhancement ratio provider that computes ERs for scenes as slopes of NO2 vs. CO columns.

    Parameters
    ----------
    scenes
        The list of scenes from the download configuration or a list of scene names.

    enhancement_dir
        The directory where the enhancement files from the ``AnalysisDriver.jl`` package are
        stored.
    """
    def __init__(self, scenes: Sequence[Union[str, dict]], enhancement_dir: os.PathLike) -> None:
        self.sub_providers = dict()
        for scene in scenes:
            if isinstance(scene, dict):
                scene = scene['name']
            self.sub_providers[scene] = SlopeErProvider.from_scene(scene, enhancement_dir)

    def daily_ers_for_subscene(self, scene_name: str, subscene: dict, tz='-08:00') -> pd.DataFrame:
        provider = self.sub_providers[scene_name]
        return provider.daily_ers(subscene_extent=subscene['extent'], tz=tz)


class BackgroundErMultiSceneProvider(ErMultiSceneProvider):
    def __init__(self, scenes: Sequence[Union[str, dict]], enhancement_dir: os.PathLike, enh_group: str, min_co_enh: float = STD_CO_ENH) -> None:
        self.sub_providers = dict()
        for scene in scenes:
            if isinstance(scene, dict):
                scene = scene['name']
            self.sub_providers[scene] = BackgroundErProvider.from_scene(scene, enhancement_dir, enh_group, min_co_enh=min_co_enh)

    def daily_ers_for_subscene(self, scene_name: str, subscene: dict, tz='-08:00') -> pd.DataFrame:
        provider = self.sub_providers[scene_name]
        return provider.daily_ers(subscene_extent=subscene['extent'], tz=tz)


class ErProvider(ABC):
    """Base class for enhancement ratio providers that operate for a single subscene.
    """
    @abstractmethod
    def daily_ers(self, subscene_extent: Tuple[float, float, float, float], tz='-08:00') -> pd.DataFrame:
        """
        Parameters
        ----------
        subscene_extent
            A 4-tuple defining the min lon, max lon, min lat, and max lat for the subscene.

        tz
            Offset from UTC to use when converting UTC time to local. Used to group by date.

        Returns
        -------
        pd.DataFrame
            A dataframe, indexed by date, with FRP statistics for this scene/subscene. The specific statistics may vary
            depending on the concrete implementation, but it is recommened to include columns "er" with the ER values
            and "er_unc" with the uncertainties on those values.
        """
        pass



class SlopeErProvider(ErProvider):
    """An enhancement ratio provider that computes ERs for a scene as slopes of NO2 vs. CO columns.

    Parameters
    ----------
    enh_nc_file
        An enhancement netCDF file generated by the ``AnalysisDriver.jl`` package.

    model
        Which fitting model to use for the slopes. The default, "sk", matches the fitting function
        used in Anderson et al.
    """
    def __init__(self, enh_nc_file: os.PathLike, model='sk') -> None:
        self.data = enhancement_ratios.load_enhancments(enh_nc_file)
        self.model = model

    def daily_ers(self, subscene_extent: Tuple[float, float, float, float], tz='-08:00') -> pd.DataFrame:
        dates = self.data['date']
        ers = np.full(dates.shape, np.nan)
        er_uncs = np.full(dates.shape, np.nan)
        for idate in range(dates.size):
            fit = enhancement_ratios.calculate_er_as_fit(self.data, subscene_extent, idate, model=self.model)
            ers[idate] = fit['er']
            er_uncs[idate] = fit['er_unc']
        return pd.DataFrame({'er': ers, 'er_unc': er_uncs}, index=dates)

    @classmethod
    def from_scene(cls, scene_name: str, enhancement_dir: os.PathLike) -> 'SlopeErProvider':
        """Convenience method to create an instance of this class from the scene name and the directory
        where the ``AnalysisDriver.jl`` enhancement files are stored. This assumes that the files are
        named following the standard ``AnalysisDriver.jl`` pattern.
        """
        nc_file = Path(enhancement_dir) / f'enhancement_{scene_name}.nc'
        return cls(nc_file)


class BackgroundErProvider(ErProvider):
    def __init__(self, enh_nc_file: os.PathLike, enh_group: str, min_co_enh: float = STD_CO_ENH):
        self._min_co_enh = min_co_enh
        """ # This keeps throwing a segfault, blame xarray. Do this the old fashioned way.
        with xr.open_datatree(enh_nc_file) as ds:
            dates = pd.to_datetime(ds['date'])
            grp = ds[enh_group]
            er = grp['enhancement_ratio'].assign_coords({'date': dates})
            co_enh = grp['co_enhancement'].assign_coords({'date': dates})
        """
        with Dataset(enh_nc_file) as ds:
            dates = pd.to_datetime(ds['date'][:])
            grid_lon = ds['grid_lon'][:]
            grid_lat = ds['grid_lat'][:]

            grp = ds.groups[enh_group]
            er = grp['enhancement_ratio'][:]
            er = xr.DataArray(er, dims=['date', 'grid_lat', 'grid_lon'], coords=[dates, grid_lat, grid_lon])
            co_enh = grp['co_enhancement'][:]
            co_enh = xr.DataArray(er, dims=['date', 'grid_lat', 'grid_lon'], coords=[dates, grid_lat, grid_lon])
        self.data = xr.Dataset({'er': er, 'co_enh': co_enh})

    def daily_ers(self, subscene_extent: Tuple[float, float, float, float], tz='-08:00') -> pd.DataFrame:
        xx_lon = (self.data.grid_lon >= subscene_extent[0]) & (self.data.grid_lon <= subscene_extent[1])
        xx_lat = (self.data.grid_lat >= subscene_extent[2]) & (self.data.grid_lat <= subscene_extent[3])
        subset_ds = self.data.isel(grid_lon=xx_lon, grid_lat=xx_lat)

        dates = []
        ers = []
        er_uncs = []
        for date, date_ds in subset_ds.groupby('date'):
            is_plume = date_ds.co_enh.data > self._min_co_enh
            plume_ers = date_ds.er.data[is_plume]
            dates.append(pd.to_datetime(date))
            ers.append(np.nanmean(plume_ers))
            er_uncs.append(np.nanstd(plume_ers, ddof=1))

        return pd.DataFrame({'er': ers, 'er_unc': er_uncs}, index=dates)

    @classmethod
    def from_scene(cls, scene_name: str, enhancement_dir: os.PathLike, enh_group: str, min_co_enh: float = STD_CO_ENH) -> 'SlopeErProvider':
        """Convenience method to create an instance of this class from the scene name and the directory
        where the ``AnalysisDriver.jl`` enhancement files are stored. This assumes that the files are
        named following the standard ``AnalysisDriver.jl`` pattern.
        """
        nc_file = Path(enhancement_dir) / f'enhancement_{scene_name}.nc'
        return cls(nc_file, enh_group, min_co_enh=min_co_enh)


def match_frps_to_ers(frp: FrpMultiSceneProvider, er: ErMultiSceneProvider, scene_name: str, subscene: dict, local_hour_limits: Tuple[float, float], tz='-08:00') -> pd.DataFrame:
    """Load daily FRP and ER values from their respective providers for one subscene and match the two datasets by date.
    """
    frp_df = frp.daily_frps_for_subscene(scene_name=scene_name, subscene=subscene, local_hour_limits=local_hour_limits, tz=tz)
    er_df = er.daily_ers_for_subscene(scene_name=scene_name, subscene=subscene, tz=tz)
    return pd.merge(frp_df, er_df, left_index=True, right_index=True)


def match_frps_to_ers_many_files(frp: FrpMultiSceneProvider, er: ErMultiSceneProvider, scenes: Sequence[dict], local_hour_limits: Tuple[float, float], tz='-08:00') -> pd.DataFrame:
    """Load daily FRP and ER values from their respective providers, match by date, and concatenate multiple subscenes' data into a single dataframe.

    Parameters
    ----------
    frp
        Provider for the FRP data

    er
        Provider for the ER data

    scenes
        The list of dictionaries defining the scenes from the download config.

    local_hour_limits
        Limits on the local hours to include in the calculation. For example, ``(9, 15)`` will only include data between 9:00a and 3:00p
        local.

    tz
        Offset from UTC to use when converting UTC time to local. Used to limit by local hour and group by date.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the FRP and ER data by date, as well as the scene and subscene names. Note that only data from
        dates greater than or equal to the fire start date will be included.
    """
    dfs = []
    nscene = len(scenes)
    for iscene, scene in enumerate(scenes, start=1):
        logger.debug(f'Matching for scene {iscene} of {nscene}: {scene["name"]}')
        for subscene in scene['subscene']:
            subscene_df = match_frps_to_ers(frp, er, scene_name=scene['name'], subscene=subscene, local_hour_limits=local_hour_limits, tz=tz)
            # Since we have VIIRS for 2020 and 2021 fire seasons without any differentiation by scene, we want to be sure that
            # the analysis covers the right period. We will have data before the fire start for the time backgrounding.
            # The end date check may be redundant since the ER data should usually end when the scene does, but better
            # to be safe.
            dd = (subscene_df.index >= subscene['fire_start']) & (subscene_df.index <= scene['end'])
            subscene_df = subscene_df.loc[dd, :]
            subscene_df['scene'] = scene['name']
            subscene_df['subscene'] = subscene['name']
            subscene_df['fire_start'] = pd.to_datetime(subscene['fire_start'])
            dfs.append(subscene_df)
    return pd.concat(dfs)


def _dti_to_local_hours(index: pd.DatetimeIndex, tz: str) -> pd.Index:
    index = index.tz_convert(tz)
    return index.hour.astype(float) + index.minute.astype(float) / 60
