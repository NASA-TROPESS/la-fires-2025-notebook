#!/Users/laughner/.juliaup/bin/julia +1.11.4
ENV["JULIA_CONDAPKG_BACKEND"] = "NULL"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(@__DIR__, "../.mambaenv/bin/python3")
using Pkg
Pkg.activate(joinpath(@__DIR__, "../env"))
config_file = "./download-config.toml"
if length(ARGS) > 0
    config_file = ARGS[1]
end
println(config_file)
if !isfile(config_file)
    error("$config_file does not exist")
end
import AnalysisDriver
AnalysisDriver.end2end_driver(config_file)
