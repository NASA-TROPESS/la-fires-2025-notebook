#!/usr/bin/env bash

# lrwxrwxrwx@    - laughner 17 Nov 09:17 -- carb -> /tb/sandbox17/laughner/la-fires-2025/static-data/carb
# .rw-r--r--@ 5.3k laughner  9 Nov 15:27 -N download-config-anderson-local.toml
# .rw-r--r--@ 5.4k laughner 29 Jul 09:13 -- download-config-anderson.toml
# .rw-r--r--@ 9.5k laughner  5 Nov 14:08 -N download-config-goes-only.toml
# .rw-r--r--@ 3.2k laughner 17 Nov 09:08 -- download-config-la.toml
# .rw-r--r--@ 4.2k laughner 17 Nov 09:08 -- download-config-small.toml
# lrwxrwxrwx@    - laughner 17 Nov 09:08 -- download-config-tropess.toml -> download-config-la.toml
# .rw-r--r--@ 8.3k laughner 17 Nov 09:07 -- download-config.toml
# drwxr-xr-x@    - laughner 28 Oct 09:57 -N fire-progression
# lrwxrwxrwx@    - laughner 23 Jul  2025 -- goes -> /tb/sandbox17/laughner/la-fires-2025/static-data/goes
# .rw-r--r--@ 168k laughner 10 Nov 11:13 -N json-update.log
# lrwxrwxrwx@    - laughner 17 Nov 09:59 -- land-cover -> /tb/sandbox17/laughner/la-fires-2025/static-data/land-cover
# lrwxrwxrwx@    - laughner 17 Nov 09:14 -- modis -> /tb/sandbox17/laughner/la-fires-2025/static-data/modis
# .rwxr-xr-x@  288 laughner 10 Jun 09:13 -N run-gridding-anderson.jl
# .rwxr-xr-x@  458 laughner 17 Nov 09:08 -- run-gridding.jl
# lrwxrwxrwx@    - laughner 17 Nov 09:17 -- scaqmd -> /tb/sandbox17/laughner/la-fires-2025/static-data/scaqmd
# drwxr-xr-x@    - laughner 23 May  2025 -- shapefiles
# lrwxrwxrwx@    - laughner 17 Nov 09:58 -- topography -> /tb/sandbox17/laughner/la-fires-2025/static-data/topography
# lrwxrwxrwx@    - laughner 16 Jul 16:45 -- tropess-daac -> /tb/sandbox17/laughner/la-fires-2025/static-data/tropess-daac
# lrwxrwxrwx@    - laughner 24 May  2025 -- tropomi -> /tb/sandbox17/laughner/la-fires-2025/static-data/tropomi
# drwxr-xr-x@    - laughner 17 Nov 10:17 -N tropomi-custom
# lrwxrwxrwx@    - laughner 17 Nov 09:14 -- viirs -> /tb/sandbox17/laughner/la-fires-2025/static-data/viirs
# lrwxrwxrwx@    - laughner 17 Nov 10:17 -- wrf-chem -> /tb/sandbox17/laughner/la-fires-2025/static-data/wrf-chem

# For most directories, we can directly compress them, dereferencing so that if they
# are links themselves, we follow those links.
# Also do not include download scripts (get.sh)
MAIN_DIRS="carb fire-progression goes land-cover modis scaqmd shapefiles topography tropess-daac viirs wrf-chem"
## for d in $MAIN_DIRS; do
##     printf "\n==========\nCompressing ${d}\n=========\n\n"
##     tar -czvf ${d}.tar.gz \
##         --dereference \
##         --exclude __pycache__ \
##         --exclude .DS_Store \
##         --exclude get.sh \
##         ${d}
## done


# For the TROPOMI standard data, we don't want to archive the L2 data (since it can be
# downloaded and isn't needed for the NB) except for the few files that we use for examples
# Likewise, the gridded subdirectory is very large on the cluster, and we only need the
# files for the 8th and 9th during the LA fires
## printf "\n==========\nCompressing tropomi\n=========\n\n"
## tar -cvf tropomi.tar \
##     --dereference \
##     --exclude 'tropomi/l2/*' \
##     --exclude 'tropomi/gridded/*' \
##     --exclude __pycache__ \
##     --exclude .DS_Store \
##     tropomi 
## 
## tar -rvf tropomi.tar \
##     tropomi/l2/no2/S5P_OFFL_L2__NO2____20250109*.nc \
##     tropomi/l2/no2_prior/S5P_OPER_AUX_CTMANA_20250108T000000_20250109T000000_20250116T133501.nc \
##     tropomi/l2/no2_prior/S5P_OPER_AUX_CTMANA_20250109T000000_20250110T000000_20250117T133721.nc \
##     tropomi/gridded/*/gridded_*_eaton_and_palisades_20250108.nc \
##     tropomi/gridded/*/gridded_*_eaton_and_palisades_20250109.nc 

gzip tropomi.tar

# The for the custom, we need to dereference because there are links to the 
# different scenarios, but we don't want to dereference the co links under the
# gridded subdirectory. We don't need most of the gridded or l2 files anyway, 
# so no sense in including them.
printf "\n==========\nCompressing tropomi-custom\n=========\n\n"
tar -cvf tropomi-custom.tar \
    --dereference \
    --exclude 'tropomi-custom/*/l2/*' \
    --exclude 'tropomi-custom/*/gridded/*' \
    --exclude __pycache__ \
    --exclude .DS_Store \
    tropomi-custom

tar -rvf tropomi-custom.tar \
    tropomi-custom/gaussian-aerosol/560hPa-by-50hPa-25ppb-0.01sw/l2/no2/S5P_CUST_L2__NO2____20250109T000000_20250109T000000_XXXXX_YY_ZZZZZZ_20260731T173416.nc \
    tropomi-custom/gaussian/560hPa-by-50hPa-25ppb/l2/no2/S5P_CUST_L2__NO2____20250109T000000_20250109T000000_XXXXX_YY_ZZZZZZ_20260717T034832.nc

gzip tropomi-custom.tar
