from pathlib import Path

STATIC_DATA_DIR = Path(__file__).parents[4] / 'static-data/'

# The standard CO enhancement used to distinguish plumes from background
STD_CO_ENH = 0.03 # mol/m2
LA_EXTENT = [-119.0, -117.0, 33.5, 34.5]
EATON_EXTENT = [-118.2, -118, 34.15, 34.255]
ALTADENA_EXTENT = [-118.2, -118.0, 34.15, 34.25]
MOLEC_CM2_TO_MOL_M2 = 1e4 / 6.022e23
