module goes_summary

using Dates
using JSON3
using Logging
using NCDatasets
using Printf
using Statistics
using AnalysisDriver

GOES_VAR_CLABELS = Dict(
    "Temp" => "Fire temperature (K)",
    "Power" => "Fire radiative power (MW)",
    "Area" => "Fire area (m\$^2\$)",
    "DQF" => "Quality Flag",
    "Mask" => "Mask"
)

GOES_VAR_CMAPS = Dict(
    "Temp" => "inferno",
    "Power" => "plasma",
    "Area" => "turbo",
)


function summarize_goes_fire_data_driver(; scene_dict, l2_dir, summary_dir)
    # Find the JSON file for the scene as a whole
    # For each subscene, check if we need to update the GOES summary file
    # If so, create it
    mkpath(summary_dir)
    scene_name = scene_dict["name"]
    subscenes = get(scene_dict, "subscene", [])
    if length(subscenes) == 0
        @warn "No subscenes defined for scene '$(scene_name)', no GOES summary files will be produced"
        return
    end

    json_file = joinpath(l2_dir, "goes_fire", "$(scene_name)_download_parameters.json")
    if !isfile(json_file)
        @warn "JSON file $json_file does not exist, will not make GOES summary for $scene_name"
        return
    end
    for subscene in subscenes
        sub_name = subscene["name"]
        sub_extent = subscene["extent"]
        summary_file = joinpath(summary_dir, "$(scene_name)-$(sub_name)-goes_summary.nc")
        if summary_file_needs_remade(summary_file, json_file)
            @info "Creating $summary_file"
            summarize_goes_fire_data(json_file, summary_file; extent=sub_extent)
        else
            @info "$summary_file newer than all inputs, not remaking"
        end
    end
end


function summarize_goes_fire_data(json_file, output_file; extent=nothing)
    json_string = read(json_file, String)
    fire_info = JSON3.read(json_string)
    goes_files = fire_info["files"]
    goes_dir = dirname(json_file)

    n = length(goes_files)
    out_dict = nothing
    time = nothing
    lat = nothing
    lon = nothing
    data = nothing
    for (ifile, file) in enumerate(goes_files)
        if ifile % 100 == 0
            @info "--> Processing GOES file $ifile of $n"
        end
        fullfile = joinpath(goes_dir, file)
        success = false
        try
            time, lat, lon, data = load_goes_fire_data(fullfile, extent=extent)
            success = true
        catch err
            @error "Error loading $file ($err)"
        end

        if success
            @debug " -> Reading $file"
            if isnothing(out_dict)
                out_dict = init_goes_fire_summary_dict(data, n, length(lat))
            end
            add_goes_fire_summary(out_dict, ifile, time, lat, lon, data)
        end
    end

    @info "Done loading, writing GOES summary file"
    fullfile = joinpath(goes_dir, goes_files[1])
    write_goes_fire_summary(output_file, fullfile, out_dict, extent)
end


function load_goes_fire_data(file; extent=nothing)
    Dataset(file) do ds
        lat, lon = AnalysisDriver.utils.load_goes_coordinates(ds)

        time = ds["t"][1]
        data = Dict(k => ds[k][:, :]' for k in keys(GOES_VAR_CLABELS))

        if !isnothing(extent)
            xx = (lon .>= extent[1]) .& (lon .<= extent[2])
            yy = (lat .>= extent[3]) .& (lat .<= extent[4])
            lat = lat[xx.&yy]
            lon = lon[xx.&yy]
            data = Dict(k => v[xx.&yy] for (k, v) in data)
        else
            lat = lat[:]
            lon = lon[:]
            data = Dict(k => v[:] for (k, v) in data)
        end
        time, lat, lon, data
    end
end



function write_goes_fire_summary(output_file, sample_input_file, out_dict, extent)
    Dataset(output_file, "c") do ds
        ntime = length(out_dict["time"])
        npix = size(out_dict["lat"], 2)
        defDim(ds, "time", ntime)
        defDim(ds, "pixel", npix)
        if !isnothing(extent)
            ds.attrib["extent"] = extent
        end

        v = defVar(ds, "time", Float64, ("time",), fillvalue=fill_val_for_type(Float64))
        v.attrib["units"] = "seconds since 1970-01-01 00:00:00"
        v[:] = nomissing(out_dict["time"], fill_val_for_type(Float64))

        v = defVar(ds, "lat", Float32, ("time", "pixel"), fillvalue=fill_val_for_type(Float32))
        v.attrib["units"] = "degrees_north"
        v[:, :] = nomissing(out_dict["lat"], fill_val_for_type(Float32))

        v = defVar(ds, "lon", Float32, ("time", "pixel"), fillvalue=fill_val_for_type(Float32))
        v.attrib["units"] = "degrees_east"
        v[:, :] = nomissing(out_dict["lon"], fill_val_for_type(Float32))

        Dataset(sample_input_file) do in_ds
            for (varname, vararr) in out_dict
                if varname in ["time", "lat", "lon"]
                    continue
                end
                dtype = nonmissingtype(eltype(vararr))
                v = defVar(ds, varname, dtype, ("time", "pixel"), fillvalue=fill_val_for_type(dtype))
                orig_var = in_ds[varname]
                for (attname, attval) in orig_var.attrib
                    if !startswith(attname, "_")
                        v.attrib[attname] = attval
                    end
                end
                v[:, :] = nomissing(vararr, fill_val_for_type(dtype))
            end
        end

    end
end

function init_goes_fire_summary_dict(data, ntime, npix)
    summary = Dict(
        "time" => fill(fill_val_for_type(Float64), ntime),
        "lat" => fill(fill_val_for_type(Float32), (ntime, npix)),
        "lon" => fill(fill_val_for_type(Float32), (ntime, npix))
    )

    for (k, v) in data
        dtype = nonmissingtype(eltype(v))
        summary[k] = fill(fill_val_for_type(dtype), (ntime, npix))
    end
    summary
end

function add_goes_fire_summary(output_ds, itime, time, lat, lon, data)
    output_ds["time"][itime] = Millisecond(time - DateTime(1970, 1, 1)).value / 1000.0
    output_ds["lat"][itime, :] = nomissing(lat, NaN)
    output_ds["lon"][itime, :] = nomissing(lon, NaN)
    for (varname, vararr) in data
        dtype = nonmissingtype(eltype(vararr))
        vararr = nomissing(vararr, fill_val_for_type(dtype))
        output_ds[varname][itime, :] = vararr
    end
end


function summary_file_needs_remade(summary_file, json_file)
    @info "Reading JSON file $json_file"
    json_data = JSON3.read(json_file)
    input_dir = dirname(json_file)
    input_files = [joinpath(input_dir, f) for f in json_data[:files]]
    return AnalysisDriver.utils.check_mtime(summary_file, input_files)
end


function fill_val_for_type(dtype)
    try
        dtype(NaN)
    catch InexactError
        typemax(dtype)
    end
end

end
