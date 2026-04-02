cd $(dirname $0)

PYTHON=$(realpath "../../.mamba-nb-env/bin/python3")
$PYTHON make_custom_l2.py $@
