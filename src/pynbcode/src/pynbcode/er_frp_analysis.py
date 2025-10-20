from abc import ABC, abstractmethod
from enum import IntEnum
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path

from . import enhancement_ratios

from typing import Sequence, Tuple, Union


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
            of FRP data within the subscene, "total_frp" as the sum of FRP data in the subscene, and "n_frp" as the number
            of FRP observations within the subscene.
        """
        pass

    @staticmethod
    def dti_to_local_hours(index: pd.DatetimeIndex, tz: str) -> pd.Index:
        index = index.tz_convert(tz)
        return index.hour.astype(float) + index.minute.astype(float) / 60


class ViirsFrpProvider(FrpProvider, FrpMultiSceneProvider):
    """An FRP provider for VIIRS data contained in a FIRMS JSON file.

    It takes a path to the JSON file to load as its single argument.
    FIRMS JSONs can be requested from https://firms.modaps.eosdis.nasa.gov/download/list.php.

    Note that this class implements both the :class:`FrpProvider` and
    :class:`FrpMultiSceneProvider` interfaces, since it assumes that the
    FRP data from multiple fires are contained in the same JSON file.
    """
    def __init__(self, json_file) -> None:
        self.df = self.load_viirs_json(json_file)

    def daily_frps(self, subscene_extent: Tuple[float, float, float, float], local_hour_limits: Tuple[float, float], tz='-08:00') -> pd.DataFrame:
        local_hours = self.dti_to_local_hours(self.df.index, tz)
        xx_lon = (self.df.longitude >= subscene_extent[0]) & (self.df.longitude <= subscene_extent[1])
        xx_lat = (self.df.latitude >= subscene_extent[2]) & (self.df.latitude <= subscene_extent[3])
        xx_hr = (local_hours >= local_hour_limits[0]) & (local_hours <= local_hour_limits[1])

        subset_frp = self.df.loc[xx_lon & xx_lat & xx_hr, 'frp']

        dates = subset_frp.index.tz_convert(tz).date
        grouped_frp = subset_frp.groupby(dates)
        return pd.DataFrame({'mean_frp': grouped_frp.mean(), 'total_frp': grouped_frp.sum(), 'n_frp': grouped_frp.count()})

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
    def __init__(self, *json_files) -> None:
        dfs = []
        for jf in json_files:
            dfs.append(self.load_viirs_json(jf))
        self.df = pd.concat(dfs)



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
    """
    def __init__(self, enh_nc_file: os.PathLike) -> None:
        self.data = enhancement_ratios.load_enhancments(enh_nc_file)

    def daily_ers(self, subscene_extent: Tuple[float, float, float, float], tz='-08:00') -> pd.DataFrame:
        dates = self.data['date']
        ers = np.full(dates.shape, np.nan)
        er_uncs = np.full(dates.shape, np.nan)
        for idate in range(dates.size):
            fit = enhancement_ratios.calculate_er_as_fit(self.data, subscene_extent, idate)
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
        print(f'Matching for scene {iscene} of {nscene}: {scene["name"]}')
        for subscene in scene['subscene']:
            subscene_df = match_frps_to_ers(frp, er, scene_name=scene['name'], subscene=subscene, local_hour_limits=local_hour_limits, tz=tz)
            dd = subscene_df.index >= subscene['fire_start']
            subscene_df = subscene_df.loc[dd, :]
            subscene_df['scene'] = scene['name']
            subscene_df['subscene'] = subscene['name']
            subscene_df['fire_start'] = pd.to_datetime(subscene['fire_start'])
            dfs.append(subscene_df)
    return pd.concat(dfs)
