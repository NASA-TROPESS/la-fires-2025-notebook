module gridding_driver

using AnalysisDriver
using Dates
using JSON3
using Mustache
using OversampledGridding

function grid_no2_scene_from_config(; scene_dict, l2_dir, grid_dir, grid_res=0.025, debug_level=0)
    scene_name = scene_dict["name"]
    l2_json_file = joinpath(l2_dir, "no2", "$(scene_name)_download_parameters.json")
    prior_json_file = joinpath(l2_dir, "no2_prior", "$(scene_name)_download_parameters.json")
    output_dir = joinpath(grid_dir, "no2")
    mkpath(output_dir)
    output_template = joinpath(output_dir, "gridded_no2_$(scene_name)_{{Date}}.nc")
    grid_no2_scene(l2_json_file=l2_json_file, prior_json_file=prior_json_file, output_template=output_template, grid_res=grid_res, debug_level=debug_level)
end


function grid_no2_scene(; l2_json_file, prior_json_file, output_template, grid_res=0.025, debug_level=0)
    l2_scene_info = JSON3.parsefile(l2_json_file)
    l2_files_by_date = group_tropomi_files_by_start_date(l2_scene_info[:files], dirname(l2_json_file))

    prior_scene_info = JSON3.parsefile(prior_json_file)
    prior_files_by_date = group_tropomi_files_by_start_date(prior_scene_info[:files], dirname(prior_json_file))

    grid_latlim, grid_lonlim = grid_def_from_scene(l2_scene_info[:scene], grid_res)

    output_template = Mustache.parse(output_template)
    for (date, files) in l2_files_by_date
        prior_files = prior_files_by_date[date]
        if length(prior_files) != 1
            error("Not implemented: expected one prior file for $date, got $(length(prior_files))")
        end
        date_config = AnalysisDriver.tropomi_no2.tropomi_no2_config(prior_files[1])
        dstr = Dates.format(date, dateformat"yyyymmdd")
        output_file = Mustache.render(output_template, Dict("Date" => dstr, "Gas" => "no2", "Scene" => l2_scene_info[:scene][:name]))
        if grid_update_required(l2_json_file, [prior_files[1], files...], output_file)
            @info "Gridding NO2 for $date"
            grid_scene_for_date(files, date_config, grid_latlim, grid_lonlim, output_file; grid_res=grid_res, debug_level=debug_level)
        end
    end
end


function grid_co_scene_from_config(; scene_dict, l2_dir, grid_dir, grid_res=0.025, debug_level=0)
    scene_name = scene_dict["name"]
    l2_json_file = joinpath(l2_dir, "co", "$(scene_name)_download_parameters.json")
    output_dir = joinpath(grid_dir, "co")
    mkpath(output_dir)
    output_template = joinpath(output_dir, "gridded_co_$(scene_name)_{{Date}}.nc")
    grid_co_scene(l2_json_file=l2_json_file, output_template=output_template, grid_res=grid_res, debug_level=debug_level)
end

function grid_co_scene(; l2_json_file, output_template, grid_res=0.025, debug_level=0)
    l2_scene_info = JSON3.parsefile(l2_json_file)
    l2_files_by_date = group_tropomi_files_by_start_date(l2_scene_info[:files], dirname(l2_json_file))
    grid_latlim, grid_lonlim = grid_def_from_scene(l2_scene_info[:scene], grid_res)
    output_template = Mustache.parse(output_template)
    config = AnalysisDriver.tropomi_co.tropomi_co_config()

    for (date, files) in l2_files_by_date
        dstr = Dates.format(date, dateformat"yyyymmdd")
        output_file = Mustache.render(output_template, Dict("Date" => dstr, "Gas" => "co", "Scene" => l2_scene_info[:scene][:name]))
        if grid_update_required(l2_json_file, files, output_file)
            @info "Gridding CO for $date"
            grid_scene_for_date(files, config, grid_latlim, grid_lonlim, output_file; grid_res=grid_res, debug_level=debug_level)
        end
    end
end

function grid_scene_for_date(files, grid_config, grid_latlim, grid_lonlim, output_file; grid_res=0.025, debug_level=0)
    output_dir = dirname(output_file)
    mkpath(output_dir)
    if debug_level > 0
        mkpath(joinpath(output_dir, "debug"))
    end

    gridded_data = average_files_to_grid(
        nc_files=files,
        config=grid_config,
        grid_latlim=grid_latlim,
        grid_lonlim=grid_lonlim,
        grid_res=grid_res,
        debug_level=debug_level,
        debug_output_pattern=joinpath(output_dir, "debug", "grid_debug_{{Stem}}.nc")
    )

    write_grid_data(output_file, gridded_data)
    @info "Wrote $output_file"
end

function group_tropomi_files_by_start_date(tropomi_files, parent=nothing)
    # TROPOMI files always have the start date in the same position in the name
    files = Dict()
    for f in tropomi_files
        dstr = basename(f)[21:28]
        date = Date(dstr, dateformat"yyyymmdd")
        if !isnothing(parent)
            f = joinpath(parent, f)
        end

        if !haskey(files, date)
            files[date] = [f]
        else
            push!(files[date], f)
        end
    end
    return files
end


function grid_update_required(json_scene_file, l2_files, output_file)
    if !isfile(output_file)
        @info "Grid output file $output_file does not exist, must create"
        return true
    end

    output_mtime = mtime(output_file)
    for file in [json_scene_file, l2_files...]
        input_mtime = mtime(output_file)
        if input_mtime > output_mtime
            @info "Input file $file is newer than $output_file, must recreate output file"
            return true
        end
    end
    @info "Gridded output file $output_file is newer than all inputs, no need to recreate"
    return false
end

function grid_def_from_scene(scene_dict, grid_res)
    lat_min = scene_dict[:lat_center] - scene_dict[:lat_width] / 2.0
    lat_min = floor(lat_min / grid_res) * grid_res
    lat_max = scene_dict[:lat_center] + scene_dict[:lat_width] / 2.0
    lat_max = ceil(lat_max / grid_res) * grid_res

    lon_min = scene_dict[:lon_center] - scene_dict[:lon_width] / 2.0
    lon_min = floor(lon_min / grid_res) * grid_res
    lon_max = scene_dict[:lon_center] + scene_dict[:lon_width] / 2.0
    lon_max = ceil(lon_max / grid_res) * grid_res

    return [lat_min, lat_max], [lon_min, lon_max]
end

end
