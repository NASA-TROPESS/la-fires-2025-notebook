from netCDF4 import Dataset
import numpy as np
import pandas as pd

def load_tropomi_no2_priors(priors_file):
    with Dataset(priors_file) as ds:
        time_array = ds['date'][:].astype(int)
        times = pd.to_datetime([pd.Timestamp(*row) for row in time_array])
        amid = ds['hyam'][:].reshape(1, -1, 1, 1)
        bmid = ds['hybm'][:].reshape(1, -1, 1, 1)
        ps = ds['ps'][:][:, np.newaxis, :, :]
        pmid = amid + bmid * ps
        lat = ds['lat'][:]
        lon = ds['lon'][:]
        no2 = ds['no2'][:]

        return {
            'time': times,
            'pres': pmid,
            'lat': lat,
            'lon': lon,
            'no2': no2
        }
