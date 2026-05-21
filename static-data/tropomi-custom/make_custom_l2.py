from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
import jllutils.geography as jgeo
from netCDF4 import Dataset, num2date
import numpy as np
from pathlib import Path
import pandas as pd
from pynbcode.tropomi.priors import load_tropomi_no2_priors
from pynbcode.tropess.tropess_data import load_muses_cols, load_tropess_cols
from scipy.interpolate import LinearNDInterpolator

STATIC_DATA_DIR = Path(__file__).parent.parent
TROPOMI_PRIORS_DIR = STATIC_DATA_DIR / 'tropomi' / 'l2' / 'no2_prior'
TROPESS_DATA_DIR = STATIC_DATA_DIR / 'tropess'
TROPESS_STD_DATA_DIR = STATIC_DATA_DIR / 'tropess-daac' / 'standard'
LA_CUSTOM_RET_EXTENT = [-120.0, -116.0, 32.0, 36.0]

VARIABLES = {
    'delta_time': 'PRODUCT/delta_time',
    'lat': 'PRODUCT/latitude',
    'lat_bounds': '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds',
    'lon': 'PRODUCT/longitude',
    'lon_bounds': '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds',
    'qa': 'PRODUCT/qa_value',
    'no2_trop': 'PRODUCT/nitrogendioxide_tropospheric_column',
    'no2_trop_err': 'PRODUCT/nitrogendioxide_tropospheric_column_precision',
    'ak': 'PRODUCT/averaging_kernel',
    'amf_trop': 'PRODUCT/air_mass_factor_troposphere',
    'amf_total': 'PRODUCT/air_mass_factor_total',
    'tropopause_index': 'PRODUCT/tm5_tropopause_layer_index',
    'surf_p': 'PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure'
}

FIXED_VARIABLES = {
    'tm5_a': 'PRODUCT/tm5_constant_a',
    'tm5_b': 'PRODUCT/tm5_constant_b'
}

NEW_VARIABLES = {
    'profile_shape': 'PRODUCT/SUPPORT_DATA/INPUT_DATA/profile_shape',
    'pres_edges': 'PRODUCT/pressure_edges',
}

NEW_VARIABLE_DIMENSIONS = {
    'profile_shape': ('time', 'scanline', 'ground_pixel', 'layer'),
    'pres_edges': ('time', 'scanline', 'ground_pixel', 'layer', 'vertices')
}

VARIABLE_ATTRIBUTE_OVERRIDES = {
    'profile_shape': dict(description='Modified profile shape, a pressure-weighted TROPESS CO profile interpolated to the TROPOMI pressure levels'),
    'no2_trop': dict(note='Recalculated using the custom profile shape'),
    'amf_trop': dict(note='Recalculated using the custom profile shape'),
    'amf_total': dict(note='Recalculated using the custom profile shape; the total NO2 column however has NOT been recalculated'),
    'pres_edges': dict(description='Pressure edges calculated from the TM5 constants and surface pressure', units='Pa'),
}

def main():
    clargs = parse_args()
    if clargs.pdb:
        import pdb
        pdb.set_trace()

    dates = pd.date_range(clargs.start_date, clargs.end_date, freq='D')
    for date in dates:
        print(f'Processing {date:%Y-%m-%d}')
        tropomi_l2_data, tropomi_l2_files = load_tropomi_l2_data(date, LA_CUSTOM_RET_EXTENT)
        profile_interp = ComboLatLonPresInterp.for_date(date)
        tropomi_fire_data = compute_fire_l2_no2(tropomi_l2_data, profile_interp)
        tropomi_l2_data.update(tropomi_fire_data)
        write_custom_tropomi_l2_file(
            date=date,
            l2_data=tropomi_l2_data,
            template_file=tropomi_l2_files[0],
            remove_existing=clargs.remove_existing,
        )


def parse_args():
    p = ArgumentParser('Generate custom TROPOMI NO2 files for LA fires')
    p.add_argument('start_date', default='2025-01-01', nargs='?', help='First date to process, default = %(default)s')
    p.add_argument('end_date', default='2025-01-11', nargs='?', help='Last date to process, default = %(default)s')
    p.add_argument('--remove-existing', action='store_true', help='Remove existing files for dates processed')
    p.add_argument('--pdb', action='store_true', help='Launch Python debugger')
    return p.parse_args()


class LatLonPresInterp(ABC):
    @abstractmethod
    def __call__(self, out_lon: float, out_lat: float, out_pres: np.ndarray, out_time: pd.Timestamp):
        pass


class TropomiPriorLatLonPresInterp:
    """Interpolate the gridded TROPOMI priors to new longitudes, latitudes, times, and pressure levels."""
    def __init__(self, time: pd.DatetimeIndex, lon: np.ndarray, lat: np.ndarray, pres: np.ndarray, values: np.ndarray):
        assert np.all(np.diff(lon) > 0)
        assert np.all(np.diff(lat) > 0)
        assert np.all(np.diff(time.to_julian_date()) > 0)

        self.time = time
        self.lon = lon
        self.lat = lat
        self.pres = pres
        self.values = values

    @classmethod
    def for_date(cls, date: datetime, priors_root=TROPOMI_PRIORS_DIR):
        next_date = date + pd.Timedelta(days=1)
        pattern = f'S5P_OPER_AUX_CTMANA_{date:%Y%m%d}T000000_{next_date:%Y%m%d}T000000_*.nc'
        possible_files = list(Path(priors_root).glob(pattern))
        if len(possible_files) != 1:
            raise FileNotFoundError(f'No unique priors file found for {date}, list of files matching the glob was: {possible_files}')

        data = load_tropomi_no2_priors(possible_files[0])
        return cls(time=data['time'], lon=data['lon'], lat=data['lat'], pres=data['pres'], values=data['no2'])

    def __call__(self, out_lon: float, out_lat: float, out_pres: np.ndarray, out_time: pd.Timestamp):
        if out_lon < self.lon.min() or out_lat > self.lon.max():
            raise ValueError('out_lon is out of domain')
        if out_lat < self.lat.min() or out_lat > self.lat.max():
            raise ValueError('out_lat is out of domain')
        if out_time < self.time.min() or out_time > self.time.max():
            raise ValueError('out_time is out of domain')

        ilon = np.flatnonzero(self.lon < out_lon).max()
        ilat = np.flatnonzero(self.lat < out_lat).max()
        itime = np.flatnonzero(self.time < out_time).max()

        profiles = np.full([8, out_pres.size], np.nan)
        coords = np.full([8, 3], np.nan)


        iprof = -1
        for jlon in [ilon, ilon+1]:
            this_lon = self.lon[jlon]
            for jlat in [ilat, ilat+1]:
                this_lat = self.lat[jlat]
                for jtime in [itime, itime+1]:
                    iprof += 1
                    this_time = self.time[jtime]


                    # Ensure we are pressure sorted so np.interp can extrapolate correctly.
                    p = self.pres[jtime, :, jlat, jlon]
                    x = self.values[jtime, :, jlat, jlon]
                    ii = np.argsort(p)
                    p = p[ii]
                    x = x[ii]
                    coords[iprof, 0] = this_lon
                    coords[iprof, 1] = this_lat
                    coords[iprof, 2] = this_time.to_julian_date()
                    profiles[iprof] = np.interp(np.log(out_pres), np.log(p), x)

        final_profile = np.full(out_pres.size, np.nan)
        out_jtime = out_time.to_julian_date()
        for ilev in range(out_pres.size):
            interp = LinearNDInterpolator(coords, profiles[:,ilev])
            final_profile[ilev] = interp(out_lon, out_lat, out_jtime)
        return final_profile


class TropessLatLonPresInterp:
    """Interpolate TROPESS posterior VMR profiles to new latitude, longitude, and pressure levels."""
    def __init__(self, lon, lat, pres, values):
        self.lon = lon
        self.lat = lat
        self.pres = pres
        self.values = values

    @classmethod
    def for_date(cls, species, date, step=0, tropess_dir=None, standard_tropess: bool = True):
        if standard_tropess:
            data_dir = tropess_dir or TROPESS_STD_DATA_DIR
            data = load_tropess_cols(data_dir, species, date)
        else:
            data_dir = tropess_dir or TROPESS_DATA_DIR
            data = load_muses_cols(data_dir, species, date, step=step)
        # TROPESS uses hPa, convert to Pa
        return cls(lon=data['lon'], lat=data['lat'], pres=data['p'] * 1e2, values=data['x'])

    @classmethod
    def _fill_fills(cls, this_z):
        x = np.arange(this_z.size)
        is_good = this_z > -990
        return np.interp(x, x[is_good], this_z[is_good])


    def __call__(self, out_lon, out_lat, out_pres, out_time, n_nodes=10):
        # Limit to the closest profiles to try to speed up the computation
        all_distances = jgeo.great_circle_distance(out_lon, out_lat, self.lon, self.lat)
        ii_dist = np.argsort(all_distances)[:n_nodes]
        new_profiles = np.full([n_nodes, out_pres.size], np.nan)

        # Ignore time - we should have cut down to the daytime values anyway
        coords = np.stack([self.lon[ii_dist], self.lat[ii_dist]], axis=1)

        for inew, iprof in enumerate(ii_dist):
            if np.all(self.pres[iprof] < 0) or np.all(self.values[iprof] < 0):
                continue
            p = self._fill_fills(self.pres[iprof])
            x = self._fill_fills(self.values[iprof])

            # Ensure we are pressure sorted so np.interp can extrapolate correctly.
            ii = np.argsort(p)
            p = p[ii]
            x = x[ii]

            # Use numpy interpolation because it does outside fill values as constant
            # value extrapolation
            new_profiles[inew] = np.interp(np.log(out_pres), np.log(p), x)

        is_all_bad = np.all(np.isnan(new_profiles), axis=1)
        new_profiles = new_profiles[~is_all_bad]
        coords = coords[~is_all_bad]
        final_profile = np.full(out_pres.size, np.nan)
        for ilev in range(out_pres.size):
            interp = LinearNDInterpolator(coords, new_profiles[:,ilev])
            final_profile[ilev] = interp(out_lon, out_lat)
        return final_profile


class ComboLatLonPresInterp:
    """Interpolate a mixture of the TROPESS CO posteriors and TROPOMI NO2 priors to new coordinates."""
    def __init__(self, tropess_interp: TropessLatLonPresInterp, tropomi_interp: TropomiPriorLatLonPresInterp):
        self.tropess_interp = tropess_interp
        self.tropomi_interp = tropomi_interp

    @classmethod
    def for_date(cls, date):
        tropess = TropessLatLonPresInterp.for_date('CO', date)
        tropomi = TropomiPriorLatLonPresInterp.for_date(date)
        return cls(tropess_interp=tropess, tropomi_interp=tropomi)

    def __call__(self, out_lon, out_lat, out_pres, out_time):
        tropess_profile = self.tropess_interp(
            out_lon=out_lon, out_lat=out_lat, out_pres=out_pres, out_time=out_time
        )
        tropess_profile /= tropess_profile.max()

        tropomi_profile = self.tropomi_interp(
            out_lon=out_lon, out_lat=out_lat, out_pres=out_pres, out_time=out_time
        )
        tropomi_profile /= tropomi_profile.max()

        return 0.5 * (tropess_profile + tropomi_profile)



def load_tropess_profiles(species, date, step=0):
    file = STATIC_DATA_DIR / 'tropess' / f'{date:%Y}' / f'{date:%m}' / f'{date:%d}' / 'L2_Products_Lite' / f'CRIS-JPSS-1_L2-{species}-{step}_{date:%Y_%m_%d}_F02_1.23_Litev01_Day_Night.nc'
    with Dataset(file) as ds:
        return {
            'lat': ds['Latitude'][:],
            'lon': ds['Longitude'][:],
            'quality': ds['Quality'][:],
            'land': ds['LandFlag'][:],
            'day': ds['DayNightFlag'][:],
            'x': ds['Species'][:],
            'p': ds['Pressure'][:],
        }


def load_tropomi_l2_data(date, extents):
    files = sorted((STATIC_DATA_DIR / 'tropomi' / 'l2' / 'no2').glob(f'*NO2____{date:%Y%m%d}T*.nc'))

    data = {k: [] for k in VARIABLES}
    fixed_data = {k: None for k in FIXED_VARIABLES}
    x1, x2, y1, y2 = extents

    delta_time_units = None
    for f in files:
        with Dataset(f) as ds:
            this_data = dict()
            for key, var in VARIABLES.items():
                this_data[key] = ds[var][:]

            for key, var in FIXED_VARIABLES.items():
                if fixed_data[key] is None:
                    fixed_data[key] = ds[var][:]
                elif not np.ma.allclose(fixed_data[key], ds[var][:]):
                    raise ValueError(f'{var} is not consistent across files')

            # Special case, it only has dimensions time and scanline, so we'll need
            # to broadcast it to be able to index it like we do the others.
            if 'delta_time' in this_data:
                delta_time = this_data['delta_time'][:,:,np.newaxis]
                assert delta_time.mask is np.False_  # if it were to have masked elements, we need to broadcast the mask separately
                this_data['delta_time'] = np.broadcast_to(delta_time, this_data['lon'].shape, subok=True)
                delta_time_units = ds[VARIABLES['delta_time']].units


        xx = (this_data['lon'] >= x1) & (this_data['lon'] <= x2)
        yy = (this_data['lat'] >= y1) & (this_data['lat'] <= y2)
        ii = xx & yy

        # As long as `val` has the same three leading dimensions as
        # lon and lat, this will select those slices where lon and lat
        # are in the extents.
        for key, val in this_data.items():
            data[key].append(val[ii])

    for key, vals in data.items():
        data[key] = np.concat(vals, axis=0)

    data.update(fixed_data)

    data['pres_edges'] = _compute_tropomi_pres_edges(
        data['surf_p'],
        data['tm5_a'],
        data['tm5_b'],
    )

    time_strings = [num2date(dt, delta_time_units).isoformat() for dt in data['delta_time']]
    data['time'] = pd.to_datetime(time_strings)

    return data, files


def _compute_tropomi_pres_edges(surf_p, a, b):
    if surf_p.ndim != 1:
        raise NotImplementedError('surf_p is not 1D')
    s = [surf_p.size] + list(a.shape)
    edges = np.ma.masked_all(s)
    surf_p = surf_p[:,np.newaxis]
    for i in [0, 1]:
        ai = a[:,i][np.newaxis, :]
        bi = b[:,i][np.newaxis, :]
        edges[:, :, i] = surf_p * bi + ai
    return edges


def compute_pres_weighted_prof(lon: float, lat: float, pres_edges: np.ma.MaskedArray, time: pd.Timestamp, profile_interp: LatLonPresInterp):
    pres_centers = np.ma.mean(pres_edges, axis=1)
    tropess_x = profile_interp(lon, lat, pres_centers, time)

    pres_thicknesses = np.abs(np.ma.diff(pres_edges, axis=1)).squeeze()
    pres_weights = pres_thicknesses / np.ma.sum(pres_thicknesses)
    return tropess_x * pres_weights


def compute_fire_amf(
    lon: float,
    lat: float,
    time: pd.Timestamp,
    tropomi_pres_edges: np.ndarray,
    tropomi_ak: np.ndarray,
    tropomi_amf_tot: float,
    tropomi_i_tropopause: int,
    profile_interp: LatLonPresInterp
):
    new_x = compute_pres_weighted_prof(lon=lon, lat=lat, pres_edges=tropomi_pres_edges, time=time, profile_interp=profile_interp)
    tropomi_sw = tropomi_ak * tropomi_amf_tot
    fire_amf_tot = np.ma.sum(tropomi_sw * new_x) / np.ma.sum(new_x)
    # The ATBD says the the tropospheric AMF includes the tropopause level.
    # If we assume the index in the file is zero-based, that means that we
    # need to add 1 to work with Python's indexing. FWIW, the CF conventions
    # use zero-based indexing in many places.
    # ATBD: https://sentiwiki.copernicus.eu/__attachments/1673595/S5P-KNMI-L2-0005-RP%20-%20Sentinel-5P%20TROPOMI%20ATBD%20NO2%20data%20products%202024%20-%202.8.0.pdf?inst-v=e9b0dc9b-a734-4403-9001-41ebb5a2cae9
    # See sect. 6.4 (pages 35-37)
    i_trop = tropomi_i_tropopause + 1
    fire_amf_trop = np.ma.sum(tropomi_sw[:i_trop] * new_x[:i_trop]) / np.ma.sum(new_x[:i_trop])
    return fire_amf_trop.item(), fire_amf_tot.item(), new_x


def compute_fire_l2_no2(tropomi_l2_data: dict, profile_interp: LatLonPresInterp):
    fire_data = {
        'no2_trop': np.ma.masked_all_like(tropomi_l2_data['no2_trop']),
        'amf_trop': np.ma.masked_all_like(tropomi_l2_data['amf_trop']),
        'amf_total': np.ma.masked_all_like(tropomi_l2_data['amf_trop']),
        'profile_shape': np.ma.masked_all_like(tropomi_l2_data['ak']),
    }

    n = tropomi_l2_data['lat'].size
    for i in range(n):
        print(f'\rProcessing {i+1}/{n}', end='')
        amf_trop_fire, amf_tot_fire, new_prof_shape = compute_fire_amf(
            lon=tropomi_l2_data['lon'][i],
            lat=tropomi_l2_data['lat'][i],
            time=tropomi_l2_data['time'][i],
            tropomi_pres_edges=tropomi_l2_data['pres_edges'][i],
            tropomi_ak=tropomi_l2_data['ak'][i],
            tropomi_amf_tot=tropomi_l2_data['amf_total'][i],
            tropomi_i_tropopause=tropomi_l2_data['tropopause_index'][i],
            profile_interp=profile_interp
        )

        fire_data['amf_trop'][i] = amf_trop_fire
        fire_data['amf_total'][i] = amf_tot_fire
        fire_data['profile_shape'][i] = new_prof_shape 
        fire_data['no2_trop'][i] = tropomi_l2_data['no2_trop'][i] * tropomi_l2_data['amf_trop'][i] / amf_trop_fire
    print('')

    for k, arr in fire_data.items():
        fire_data[k] = np.ma.masked_invalid(arr)
    return fire_data


def write_custom_tropomi_l2_file(date: pd.Timestamp, l2_data: dict, template_file: Path, remove_existing: bool = False):
    all_vars = VARIABLES.copy()
    all_vars.update(FIXED_VARIABLES)
    all_vars.update(NEW_VARIABLES)

    dest_dir = STATIC_DATA_DIR / 'tropomi-custom' / 'l2' / 'no2'
    if remove_existing:
        glob_existing = f'S5P_CUST_L2__NO2____{date:%Y%m%d}T000000_{date:%Y%m%d}T000000_XXXXX_YY_ZZZZZZ_*.nc'
        for extant in dest_dir.glob(glob_existing):
            extant.unlink()
            print(f'  Removed existing file {extant}')

    now = pd.Timestamp.now('utc')
    filename = f'S5P_CUST_L2__NO2____{date:%Y%m%d}T000000_{date:%Y%m%d}T000000_XXXXX_YY_ZZZZZZ_{now:%Y%m%dT%H%M%S}.nc'
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
    file = dest_dir / filename
    with Dataset(file, 'w') as ds, Dataset(template_file) as temp:
        ds.createGroup('PRODUCT')

        for key, arr in l2_data.items():
            if key not in all_vars:
                print(f'Warning: skipping output of variable "{key}" (no variable path defined)')
                continue

            varpath = all_vars[key]
            try:
                template_var = temp[varpath]
            except IndexError:
                template_dims = NEW_VARIABLE_DIMENSIONS[key]
                template_attrs = dict()
            else:
                template_dims = template_var.dimensions
                template_attrs = template_var.__dict__
            arr = _expand_custom_data(arr, template_dims)

            for dim, n in zip(template_dims, arr.shape):
                if dim not in ds['PRODUCT'].dimensions:
                    ds.groups['PRODUCT'].createDimension(dim, n)
                elif ds['PRODUCT'].dimensions[dim].size != n:
                    m = ds['PRODUCT'].dimensions[dim].size
                    raise ValueError(f'Dimension {dim} has inconsistent lengths ({n} vs {m}')

            # Let the netCDF package handle type conversions and scaling/offset itself.
            # Otherwise, the scaling/offset will get double applied, or applied after
            # the conversion to integer type for example.
            var = ds.createVariable(varpath, template_var.dtype, template_dims)
            new_attrs = VARIABLE_ATTRIBUTE_OVERRIDES.get(key, dict())
            template_attrs.update(new_attrs)
            # Always override _FillValue, since that will cause the masked values to be
            # the incorrect value to be identified as fills if the original and new
            # values don't match
            template_attrs['_FillValue'] = var.get_fill_value()
            var.setncatts(template_attrs)
            var[:] = arr

    print(f'  - Wrote {file}')


def _expand_custom_data(arr: np.ndarray, dims: list[str]) -> np.ndarray:
    # In the custom data in memory, we unraveled the time, scanline, and ground_pixel
    # dimensions to be a single dimension for simplicity. To match the standard L2 file
    # format, we reinsert time and ground_pixel as singleton dimensions. 
    for dim in ['time', 'ground_pixel']:
        try:
            i = dims.index(dim)
        except ValueError:
            pass
        else:
            arr = np.expand_dims(arr, axis=i)
    return arr


if __name__ == '__main__':
    main()
