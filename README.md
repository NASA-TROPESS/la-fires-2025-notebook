# Support code for "Quantifying Oxidized Reactive Nitrogen Emissions from Wildland-Urban Interface Fires..."

## Quickstart

1. Have [pixi](https://pixi.prefix.dev/latest/) installed
2. Clone this repo with the `--recurse-submodules` flag
3. Download the tarballs from [zenodo](https://doi.org/10.5281/zenodo.24326385)
   and unpack each under the `static-data` subdirectory
4. In the top of this repo, run `pixi run jupyter`, then copy and paste the link it emits
   into your browser to access its jupyter-lab instance.
5. Open the `EST LA Fires.ipynb` notebook to find how all figures from the paper were produced.

## Reproducing the summary data

Much of the data unpacked under `static-data` is the result of summarizing L2 satellite data.
These steps should be reproducible using the `run-gridding.jl` script under `static-data`.
It takes one of the `download-config*.toml` files as its positional argument; running it with
`download-config-la.toml`, `download-config-anderson.toml`, and `download-config-tropess.toml`
should reproduce the summarization steps.
