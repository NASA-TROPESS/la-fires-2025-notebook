#!/usr/bin/env julia +1.11.4
ENV["JULIA_CONDAPKG_BACKEND"] = "NULL"
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(@__DIR__, "../../.mambaenv/bin/python3")
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../env/"))

using AnalysisDriver
using Dates
using JSON3

function main()
    cfg = AnalysisDriver.load_config_file("../download-config-small.toml")
    scene = cfg["scene"][1]
    start = Date(scene["start"], dateformat"yyyy-mm-dd")
    stop = Date(scene["end"], dateformat"yyyy-mm-dd")
    for date in start:Day(1):stop
        grid_no2_for_date(date; scene_name=scene["name"])
    end
    AnalysisDriver.enhancements.create_enhancement_file(scene=scene, grid_dir="gridded/", enhancement_dir="enhancements/")
end

function grid_no2_for_date(date::Date; scene_name="eaton_and_palisades", grid_res=0.025, l2_dir="l2/", debug_level=0)
    l2_json_file = joinpath(l2_dir, "no2", "$(scene_name)_download_parameters.json")
    l2_scene_info = JSON3.parsefile(l2_json_file)
    l2_dir = joinpath(l2_dir, "no2")
    grid_latlim, grid_lonlim = AnalysisDriver.gridding_driver.grid_def_from_scene(l2_scene_info[:scene], grid_res)
    custom_l2_file = get_custom_file_for_date(date, l2_dir)
    dstr = Dates.format(date, dateformat"yyyymmdd")
    grid_file = joinpath("gridded", "no2", "gridded_no2_$(scene_name)_$(dstr).nc")
    config = AnalysisDriver.tropomi_no2.tropomi_no2_config(nothing)
    @info "Gridding custom NO2 for $date, from $custom_l2_file, saving as $grid_file"
    AnalysisDriver.gridding_driver.grid_scene_for_date(
        [custom_l2_file],
        config,
        grid_latlim,
        grid_lonlim,
        grid_file;
        grid_res=grid_res,
        debug_level=debug_level
    )
end

function get_custom_file_for_date(date::Date, l2_dir::String)::String
    # TROPOMI files always have the start date in the same position in the name,
    # and we mimic that with the custom files
    files = readdir(l2_dir)

    for f in files
        if !endswith(f, ".nc")
            continue
        end
        dstr = basename(f)[21:28]
        filedate = Date(dstr, dateformat"yyyymmdd")

        if filedate == date
            return joinpath(l2_dir, f)
        end
    end

    error("Could not find a file for $(date) in $(l2_dir)")
end

main()
