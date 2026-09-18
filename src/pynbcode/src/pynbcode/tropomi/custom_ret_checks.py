from netCDF4 import Dataset
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re

from ..constants import STATIC_DATA_DIR


def match_custom_sw_to_orig(aerosol_custom_l2_file: os.PathLike, extent: tuple[float, float, float, float],
                            std_l2_dir: Path = STATIC_DATA_DIR / 'tropomi' / 'l2' / 'no2') -> dict:
    """Return a dictionary with data needed to compare custom scattering weights to standard ones

    Parameters
    ----------
    aerosol_custom_l2_file
        A path to the custom L2 file under the repository's :path:`static-data` directory
        that was run with custom scattering weights imposed. This must have the variable
        ``PRODUCT/SUPPORT_DATA/INPUT_DATA/scattering_weights``.

    extent
        The longitude and latitude limits that we want to match in, given as
        ``[lonmin, lonmax, latmin, latmax]``.

    std_l2_dir
        Path to the directory containing the standard TROPOMI L2 NO2 files.

    Returns
    -------
    matched_data
        A dictionary with keys:

        - "lat", "lon": pixel lat/lon centers
        - "latcorn", "loncorn": pixel lat/lon corners
        - "std_sw": scattering weights from the standard TROPOMI files,
           computed as AK * AMF_total.
        - "cust_aer_sw": scattering weights from the aerosol test case file.
        - "cust_prior_shape": shape factor from the Gaussian plume test case
          the used the standard scattering weights.
        - "cust_aer_shape": same, but from the test case with custom scattering
          weights.
        - "std_no2": troposphere NO2 VCD from the standard product.
        - "cust_prior_no2": trop. NO2 VCD from the Gaussian plume test using
          standard scattering weights.
        - "cust_aer_no2": trop. NO2 VCD from the Gaussian plume test using
          custom scattering weights.
        - "std_amf": troposphere AMF from the standard product.
        - "cust_prior_amf": trop. AMF from the Gaussian plume test using
          standard scattering weights.
        - "cust_aer_amf": trop. AMF from the Gaussian plume test using
          custom scattering weights.
        - "pres": pressure levels, in Pa.

        All values will be reshaped so that the first dimension iterates over
        all pixels. For values with >1 value per pixel (level or corner), those
        will be along the second dimension.
    """
    # First, find the original L2 files for the date of our custom file
    custom_datestr = Path(aerosol_custom_l2_file).name[20:28]

    prior_custom_l2_parts = list(Path(aerosol_custom_l2_file).parts[:-1])
    ipath = prior_custom_l2_parts.index('gaussian-aerosol')
    prior_custom_l2_parts[ipath] = 'gaussian'
    prior_custom_l2_parts[ipath+1] = re.sub(r'-[0-9.]+sw$', '', prior_custom_l2_parts[3])
    prior_custom_l2_file = list(Path(*prior_custom_l2_parts).glob(f'S5P_CUST_L2__NO2____{custom_datestr}*.nc'))
    assert len(prior_custom_l2_file) == 1, 'Expected only one file with custom priors'
    prior_custom_l2_file = prior_custom_l2_file[0]

    std_l2_files = Path(std_l2_dir).glob(f'S5P_????_L2__NO2____{custom_datestr}*.nc')

    # Load the data in the extent from each of the L2 files
    std_lat = []
    std_lon = []
    std_sw = []
    std_no2 = []
    std_amf_trop = []
    x1, x2, y1, y2 = extent
    for file in std_l2_files:
        with Dataset(file) as ds:
            this_lat = ds['PRODUCT/latitude'][:]
            this_lon = ds['PRODUCT/longitude'][:]
            this_ak = ds['PRODUCT/averaging_kernel'][:]
            this_amf_tot = ds['PRODUCT/air_mass_factor_total'][:]
            this_amf_trop = ds['PRODUCT/air_mass_factor_troposphere'][:]
            this_no2 = ds['PRODUCT/nitrogendioxide_tropospheric_column'][:]

            ii = (this_lon >= x1) & (this_lon <= x2) & (this_lat >= y1) & (this_lat <= y2)
            if ii.sum() > 0:
                std_lat.append(this_lat[ii])
                std_lon.append(this_lon[ii])
                std_sw.append(this_ak[ii] * this_amf_tot[ii][:,np.newaxis])
                std_amf_trop.append(this_amf_trop[ii])
                std_no2.append(this_no2[ii])

    std_lat = np.concatenate(std_lat, axis=0)
    std_lon = np.concatenate(std_lon, axis=0)
    std_sw = np.concatenate(std_sw, axis=0)
    std_amf_trop = np.concatenate(std_amf_trop, axis=0)
    std_no2 = np.concatenate(std_no2, axis=0)

    # Load the NO2 and AMF from the custom file that only changed the prior
    with Dataset(prior_custom_l2_file) as ds:
        cust_prior_lat = ds['PRODUCT/latitude'][:]
        cust_prior_lon = ds['PRODUCT/longitude'][:]
        cust_prior_amf_trop = ds['PRODUCT/air_mass_factor_troposphere'][:]
        cust_prior_no2 = ds['PRODUCT/nitrogendioxide_tropospheric_column'][:]
        cust_prior_shape = ds['PRODUCT/SUPPORT_DATA/INPUT_DATA/profile_shape'][:]

        ii = (cust_prior_lon >= x1) & (cust_prior_lon <= x2) & (cust_prior_lat >= y1) & (cust_prior_lat <= y2)
        cust_prior_lat = cust_prior_lat[ii]
        cust_prior_lon = cust_prior_lon[ii]
        cust_prior_amf_trop = cust_prior_amf_trop[ii]
        cust_prior_no2 = cust_prior_no2[ii]
        cust_prior_shape = cust_prior_shape[ii]

    # Now load the scattering weights and lat/lon from the custom file with modified scattering weights.
    # Restrict to the extent as well.
    with Dataset(aerosol_custom_l2_file) as ds:
        cust_lat = ds['PRODUCT/latitude'][:]
        cust_lon = ds['PRODUCT/longitude'][:]
        cust_latcorn = ds['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][:]
        cust_loncorn = ds['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][:]
        cust_sw = ds['PRODUCT/SUPPORT_DATA/INPUT_DATA/scattering_weights'][:]
        cust_amf_trop = ds['PRODUCT/air_mass_factor_troposphere'][:]
        cust_no2 = ds['PRODUCT/nitrogendioxide_tropospheric_column'][:]
        cust_pres = ds['PRODUCT/pressure_edges'][:]
        cust_shape = ds['PRODUCT/SUPPORT_DATA/INPUT_DATA/profile_shape'][:]

        ii = (cust_lon >= x1) & (cust_lon <= x2) & (cust_lat >= y1) & (cust_lat <= y2)
        cust_lat = cust_lat[ii]
        cust_lon = cust_lon[ii]
        cust_latcorn = cust_latcorn[ii]
        cust_loncorn = cust_loncorn[ii]
        cust_amf_trop = cust_amf_trop[ii]
        cust_no2 = cust_no2[ii]
        cust_sw = cust_sw[ii]
        cust_shape = cust_shape[ii]
        cust_pres = cust_pres[ii]
        # convert edges to centers
        cust_pres = 0.5*(cust_pres[:,:,0] + cust_pres[:,:,1])

    # Finally, match things up. TROPOMI pixels should be separated by more than 0.01 deg,
    # so we can use their lat/lon * 100 as integers to match on.
    std_df = pd.DataFrame({'lon_int': (std_lon*100).astype(int), 'lat_int': (std_lat*100).astype(int), 'lin_idx_std': np.arange(std_lat.size)})
    cust_df = pd.DataFrame({'lon_int': (cust_lon*100).astype(int), 'lat_int': (cust_lat*100).astype(int), 'lin_idx_cust': np.arange(cust_lat.size)})
    cust_prior_df = pd.DataFrame({'lon_int': (cust_prior_lon*100).astype(int), 'lat_int': (cust_prior_lat*100).astype(int), 'lin_idx_prior': np.arange(cust_prior_lat.size)})
    merge_df = pd.merge(cust_df, std_df, on=['lat_int', 'lon_int'], how='left')
    merge_df = pd.merge(merge_df, cust_prior_df, on=['lat_int', 'lon_int'], how='left')

    npix = merge_df.shape[0]
    nlev = cust_sw.shape[1]
    output = {
        'lat': np.full(npix, np.nan),
        'lon': np.full(npix, np.nan),
        'latcorn': np.full([npix, 4], np.nan),
        'loncorn': np.full([npix, 4], np.nan),
        'std_sw': np.full([npix, nlev], np.nan),
        'cust_aer_sw': np.full([npix, nlev], np.nan),
        'cust_prior_shape': np.full([npix, nlev], np.nan),
        'cust_aer_shape': np.full([npix, nlev], np.nan),
        'std_no2': np.full(npix, np.nan),
        'cust_prior_no2': np.full(npix, np.nan),
        'cust_aer_no2': np.full(npix, np.nan),
        'std_amf': np.full(npix, np.nan),
        'cust_prior_amf': np.full(npix, np.nan),
        'cust_aer_amf': np.full(npix, np.nan),
        'pres': np.full([npix, nlev], np.nan)
    }

    for _, row in merge_df.iterrows():
        i_std = row['lin_idx_std']
        i_cust = row['lin_idx_cust']
        i_prior = row['lin_idx_prior']
        output['lat'][i_cust] = cust_lat[i_cust]
        output['lon'][i_cust] = cust_lon[i_cust]
        output['latcorn'][i_cust] = cust_latcorn[i_cust]
        output['loncorn'][i_cust] = cust_loncorn[i_cust]
        output['std_sw'][i_cust] = std_sw[i_std]
        output['cust_aer_sw'][i_cust] = cust_sw[i_cust]
        output['cust_prior_shape'][i_cust] = cust_prior_shape[i_prior]
        output['cust_aer_shape'][i_cust] = cust_shape[i_cust]
        output['std_no2'][i_cust] = std_no2[i_std]
        output['cust_prior_no2'][i_cust] = cust_prior_no2[i_prior]
        output['cust_aer_no2'][i_cust] = cust_no2[i_cust]
        output['std_amf'][i_cust] = std_amf_trop[i_std]
        output['cust_prior_amf'][i_cust] = cust_prior_amf_trop[i_prior]
        output['cust_aer_amf'][i_cust] = cust_amf_trop[i_cust]
        output['pres'][i_cust] = cust_pres[i_cust]
    return output
