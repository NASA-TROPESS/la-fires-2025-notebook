#!/usr/bin/env bash
TAR_FILE=la-fires-data.tar

# Archive everything except the L2 data (since that can be downloaded elsewhere)
# and the TROPESS directory (which has the internal MUSES format that I switched away from).
# Also skip Python bytecode caches and the TEMPO symlink, which points outside this repo.
tar -cvf $TAR_FILE \
    --exclude ./static-data/tropomi/l2 \
    --exclude ./static-data/tropess \
    --exclude ./static-data/tempo \
    --exclude __pycache__ \
    ./static-data/

# Include the Jan 8th and 9th data needed for examples of the custom priors
tar -rvf $TAR_FILE \
    ./static-data/tropomi/l2/no2/S5P_OFFL_L2__NO2____20250109*.nc \
    ./static-data/tropomi/l2/no2_prior/S5P_OPER_AUX_CTMANA_20250108T000000_20250109T000000_20250116T133501.nc \
    ./static-data/tropomi/l2/no2_prior/S5P_OPER_AUX_CTMANA_20250109T000000_20250110T000000_20250117T133721.nc

gzip $TAR_FILE
