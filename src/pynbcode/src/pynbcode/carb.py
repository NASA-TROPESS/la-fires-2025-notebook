import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pathlib import Path


def read_carb_csv(csv_file):
    df = pd.read_csv(csv_file, low_memory=False)
    # These files have some extra text at the end, it seems to put "Quality Flag Definition" into the site column,
    # so find that line and get rid of that tail.
    i = np.flatnonzero(df.site.str.strip() == 'Quality Flag Definition')[0]
    df = df.iloc[:i]
    df['date'] = pd.to_datetime(df['date']) + pd.to_timedelta(df.start_hour.astype(int), unit='h')
    df['start_hour'] = df.start_hour.astype(int)
    varname = df.variable.unique().item()
    df.rename(columns={'value': varname}, inplace=True)
    df.drop(columns=['variable'], inplace=True)
    return df


def merge_carb_csvs(no_csvs, no2_csvs, minimal=True):
    no_df = pd.concat([read_carb_csv(f) for f in no_csvs])
    no2_df = pd.concat([read_carb_csv(f) for f in no2_csvs])
    if minimal:
        no_df = no_df[['site', 'date', 'latitude', 'longitude', 'name', 'NO']]
        no2_df = no2_df[['site', 'date', 'NO2']]
    return pd.merge(no_df, no2_df, on=['site', 'date'], suffixes=('_no', '_no2'))


def load_carb_no_no2(carb_path: os.PathLike):
    carb_path = Path(carb_path)
    return merge_carb_csvs(
        sorted(carb_path.glob('NO_*.csv')),
        sorted(carb_path.glob('NO2_*.csv'))
    )


def plot_carb_ratio_vs_nox(carb_df, for_hour=None, extent=None, ax=None):
    if for_hour is not None:
        carb_df = carb_df.loc[carb_df.date.dt.hour == for_hour]

    total_nox = carb_df.NO + carb_df.NO2
    ratios = carb_df.NO2 / carb_df.NO
    xx = np.isfinite(ratios)

    ax = ax or plt.gca()
    h = ax.hexbin(total_nox[xx], ratios[xx], mincnt=1, bins='log', extent=extent, cmap='Greys')
    plt.colorbar(h, ax=ax, label='# observations')
    ax.set_xlabel('NO$_x$ (ppm)')
    ax.set_ylabel('NO$_2$:NO ratio' if for_hour is None else f'NO$_2$:NO ratio at {for_hour}:00')
    ax.grid(True)


def plot_carb_ratio_vs_nox_subset(carb_df, time_extent, spatial_extent, ax=None, **plot_kws):

    t = carb_df.date.dt.tz_localize('-08:00')
    carb_df = carb_df[(t >= time_extent[0]) & (t <= time_extent[1])]
    x1, x2, y1, y2 = spatial_extent
    xxyy = (carb_df.longitude >= x1) & (carb_df.longitude <= x2) & (carb_df.latitude >= y1) & (carb_df.latitude <= y2)
    carb_df = carb_df[xxyy]

    total_nox = carb_df.NO + carb_df.NO2
    ratios = carb_df.NO2 / carb_df.NO
    xx = np.isfinite(ratios)

    ax = ax or plt.gca()
    defaults = {'ls': 'none', 'marker': '.', 'color': 'r'}
    for k, v in defaults.items():
        plot_kws.setdefault(k, v)
    ax.plot(total_nox[xx], ratios[xx], **plot_kws)
