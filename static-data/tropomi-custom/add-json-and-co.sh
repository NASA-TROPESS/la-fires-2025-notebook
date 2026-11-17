#!/usr/bin/env bash

cd $(dirname $0)
src_l2_no2=$(realpath ../tropomi/l2/no2/)
src_grid_co=$(realpath ../tropomi/gridded/co)
echo "src_l2_no2 = $src_l2_no2"
echo "src_grid_co = $src_grid_co"
for d in $@; do
    pushd "$d/" > /dev/null
    echo "In $(pwd)"
    rm -v l2/no2/*.json
    cp -v "$src_l2_no2"/*.json ./l2/no2/
    mkdir -p gridded
    ln -sv "$src_grid_co" gridded/co
    popd > /dev/null
    echo ""
done
