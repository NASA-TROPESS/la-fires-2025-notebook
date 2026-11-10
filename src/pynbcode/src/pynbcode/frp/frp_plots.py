import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_goes_frp_by_hour_df(goes_provider, scenes, hours=np.arange(8, 18)):
    hour_dfs = dict()

    for i in range(hours.size-1):
        time_lims = hours[i:i+2]
        dfs = []
        for scene in scenes:
            for subscene in scene['subscene']:
                subscene_df = goes_provider.daily_frps_for_subscene(scene['name'], subscene, time_lims).copy()
                dd = (subscene_df.index >= subscene['fire_start']) & (subscene_df.index <= scene['end'])
                subscene_df.loc[dd, 'scene'] = scene['name']
                subscene_df.loc[dd, 'subscene'] = subscene['name']
                subscene_df.loc[dd, 'fire_start'] = pd.to_datetime(subscene['fire_start'])
                dfs.append(subscene_df.loc[dd, :])

        hour_dfs[(hours[i], hours[i+1])] = pd.concat(dfs)

    return [(tuple(int(h) for h in k), v) for k, v in hour_dfs.items()]


def frp_by_hour_violins(frp_by_hour_list, column='mean_frp', logy=True, extrema=True, ax=None):
    x = []
    y = []
    ticks = set()
    for hours, frp_df in frp_by_hour_list:
        ticks.update(hours)
        this_y = frp_df[column].dropna().to_numpy()
        if this_y.size > 0:
            x.append(sum(hours)/2)
            y.append(this_y)

    ax = ax or plt.gca()
    ax.violinplot(y, x, showmeans=True, showextrema=extrema)
    if logy:
        ax.set_yscale('log')
    ax.set_xticks(sorted(ticks))
    ax.grid(True)
    ax.set_xlabel('Local hour')
    ax.set_ylabel(column)


def frp_by_hour_per_date(frp_by_hour_list, column='mean_frp', ax=None):
    ax = ax or plt.gca()
    combo_df = _concat_df_by_hour(frp_by_hour_list)
    for date in combo_df.index.get_level_values('date').unique():
        date_frp = combo_df.loc[date, column]
        date_frp = _series_to_hour_midpoint(date_frp)
        ax.plot(date_frp, marker='o', label=date.strftime('%b %d'))

def _concat_df_by_hour(frp_by_hour_list):
    new_dfs = []
    for hours, df in frp_by_hour_list:
        if df.shape[0] > 0:
            df = df.copy()
            df.index = pd.MultiIndex.from_product([df.index, [hours]], names=['date', 'hours'])
            new_dfs.append(df)
    combo_df = pd.concat(new_dfs)
    return combo_df

def _series_to_hour_midpoint(s):
    x = [(h1 + h2)/2 for h1, h2 in s.index]
    return pd.Series(s.to_numpy(), index=x)
