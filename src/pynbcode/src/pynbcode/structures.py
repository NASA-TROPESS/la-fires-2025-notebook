import cartopy.crs as ccrs
from enum import IntEnum
from jllutils import miscutils
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from shapely import Polygon
import shapely.plotting

from .constants import ALTADENA_EXTENT, STATIC_DATA_DIR


from typing import Tuple


class DamageLevel(IntEnum):
    NO_DAMAGE = 0
    AFFECTED = 1
    MINOR = 2
    MAJOR = 3
    DESTROYED = 4
    INACCESSIBLE = -1


def load_structures_damage_df(structures_file=STATIC_DATA_DIR / 'fire-progression/structures.geojson'):
    label_map = {
        'No Damage': DamageLevel.NO_DAMAGE,
        'Affected (>0-10%)': DamageLevel.AFFECTED,
        'Minor (10-25%)': DamageLevel.MINOR,
        'Major (25-50%)': DamageLevel.MAJOR,
        'Destroyed (>50%)': DamageLevel.DESTROYED,
        'Inaccessible': DamageLevel.INACCESSIBLE
    }

    df = gpd.read_file(structures_file)
    df['DAMAGE_LEVEL'] = df.DAMAGE.map(lambda s: label_map[s])
    return df


def get_last_fire_size(fire_name: str, perimeter_file=STATIC_DATA_DIR / 'fire-progression/fire-perimeters.geojson'):
    df = gpd.read_file(perimeter_file)
    nn = df.name.str.lower() == fire_name.lower()
    df = df.loc[nn].sort_values('dateTime')
    return df.iloc[-1]


def subset_structures(
    structures_df: gpd.GeoDataFrame,
    incident_name: str | list[str] | None = None,
    min_damage: DamageLevel | None = None,
    max_damage: DamageLevel | None = None,
):
    xx = np.ones(structures_df.shape[0], dtype=bool)
    if incident_name is not None:
        incident_set = set([incident_name.lower()]) if isinstance(incident_name, str) else set(n.lower() for n in incident_name)
        xx &= structures_df.INCIDENTNAME.str.lower().isin(incident_set)
    if min_damage is not None:
        xx &= structures_df.DAMAGE_LEVEL >= min_damage
    if max_damage is not None:
        xx &= structures_df.DAMAGE_LEVEL <= max_damage

    return structures_df.loc[xx, :]


def make_structure_damage_2d_hist(structure_gdf: gpd.GeoDataFrame, extent=None) -> (np.ndarray, np.ndarray, np.ndarray):
    # Use different numbers of bins to make it more obvious if the transpose is
    # screwed up.
    if extent is None:
        lon_bins = 51
        lat_bins = 50
    else:
        lon_bins = np.linspace(extent[0], extent[1], num=51)
        lat_bins = np.linspace(extent[2], extent[3], num=50)

    lon = structure_gdf.LONGITUDE.to_numpy()
    lat = structure_gdf.LATITUDE.to_numpy()

    return np.histogram2d(lon, lat, [lon_bins, lat_bins])


def plot_fire_detections_over_fire(fires_df: pd.DataFrame, structures_gdf: gpd.GeoDataFrame,
                                   perimeters_gdf: gpd.GeoDataFrame, fire_name: str, 
                                   extent: Tuple[float, float, float, float], tz: str = '-08:00',
                                   ax: plt.Axes = None, **plot_kws):
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    plot_structures_hist(structures_gdf, extent=extent, ax=ax)
    plot_final_perimeter(perimeters_gdf, fire_name, ax=ax)

    plot_kws.setdefault('marker', '.')
    plot_kws.setdefault('color', 'red')
    plot_kws.setdefault('ls', 'none')
    ax.plot(fires_df.lon, fires_df.lat, **plot_kws)

    tmin = fires_df.datetime.min().tz_convert(tz)
    tmax = fires_df.datetime.max().tz_convert(tz)
    ax.set_title(f'{tmin} to\n{tmax}', fontsize=10)
    gl = ax.gridlines(draw_labels=True)
    gl.right_labels = False
    gl.top_labels = False


def plot_structures_hist(structures_gdf, extent=None, add_colorbar=True, ax=None, **pcolor_kws):
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    pcolor_kws.setdefault('vmin', 1)
    pcolor_kws.setdefault('cmap', 'bone_r')
    structures_gdf = structures_gdf[structures_gdf.DAMAGE_LEVEL == DamageLevel.DESTROYED]
    n, lons, lats = make_structure_damage_2d_hist(structures_gdf, extent=extent)
    n = n.astype(float)
    n[n < pcolor_kws['vmin']] = np.nan

    h = ax.pcolormesh(lons, lats, n.T, **pcolor_kws)
    if add_colorbar:
        plt.colorbar(h, ax=ax, label='# structure destroyed')
    return h


def altadena_structure_perimeter(structure_dmg: gpd.GeoDataFrame):
    h, x, y = make_structure_damage_2d_hist(structure_dmg, extent=ALTADENA_EXTENT)
    m = miscutils.flood_fill_at_coords(h, x, y, -118.15, 34.195, 20)
    return miscutils.extract_polygon_from_mask(m, x, y)


def plot_altadena_structure_perimeter(structure_dmg: gpd.GeoDataFrame, ax=None, **poly_kws):
    coords = altadena_structure_perimeter(structure_dmg)
    assert len(coords) == 1, 'Expected only a single polygon for the structure perimeter'
    poly = Polygon(coords[0])
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    poly_kws.setdefault('add_points', False)
    poly_kws.setdefault('facecolor', 'gray')
    poly_kws.setdefault('edgecolor', 'black')
    shapely.plotting.plot_polygon(poly, transform=ccrs.PlateCarree(), ax=ax, **poly_kws)

    poly_kws.pop('add_points')
    rect = Rectangle((0,0), 0, 0, label='Altadena structures', **poly_kws)
    return rect


def plot_final_perimeter(perimeters_gdf, name, ax=None, **poly_kws):
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    sub_gdf = perimeters_gdf[perimeters_gdf.name.str.lower() == name.lower()].sort_values('acres', ascending=False)
    row = sub_gdf.iloc[0:1]
    poly_kws.setdefault('facecolor', 'red')
    poly_kws.setdefault('edgecolor', 'red')
    poly_kws.setdefault('alpha', 0.1)
    row.plot(ax=ax, transform=ccrs.PlateCarree(), **poly_kws)
    rect = Rectangle((0,0), 0, 0, **poly_kws, label=f'{name.capitalize()} fire final perimeter')
    return rect
