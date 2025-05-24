#!/Users/laughner/.juliaup/bin/julia +1.11.4
ENV["JULIA_CONDAPKG_BACKEND"] = "NULL"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(@__DIR__, "../.mambaenv/bin/python3")
using Pkg
Pkg.activate(joinpath(@__DIR__, "../env"))
import AnalysisDriver
AnalysisDriver.end2end_driver("./download-config.toml")
