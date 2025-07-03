module utils

"""
    check_mtime(output_file, input_files)

Compare the modification time of the `output_file` to the `input_files`
and determine if `output_file` should be recreated.

# Parameters

- `output_file`: a path to the file that would be created
- `input_files`: a list of paths to the various files that the output depends on.

# Returns

`true` if `output_file` should be recreated (because an input was newer or the
output file does not exist), `false` otherwise.
"""
function check_mtime(output_file, input_files)
    if !isfile(output_file)
        @info "Output file $output_file does not exist, must create"
        return true
    end

    output_mtime = mtime(output_file)
    for file in input_files
        input_mtime = mtime(file)
        if input_mtime > output_mtime
            @info "Input file $file is newer than $output_file, must recreate output file"
            return true
        end
    end
    @info "Output file $output_file is newer than all inputs, no need to recreate"
    return false

end

"""
    allclose(arr, val; tol=1e-8)

Return `true` if all values in `a` are within a tolerance of `v`.
By default, any NaNs in `a` or `b` will cause this to return `false`.
Set the `equal_nan` keyword to `true` to consider `a` and `b` equal
when two elements are both NaN when `a` and `b` are broadcast against
each other.
"""
function allclose(a, b; tol=1e-8, equal_nan=false)
    if !equal_nan
        return all(abs.(a .- b) .< tol)
    end

    a_is_nan = isnan.(a)
    b_is_nan = isnan.(b)
    if any(a_is_nan .!= b_is_nan)
        return false
    end

    return all(abs.(a[.!a_is_nan] .- b[.!b_is_nan]) .< tol)
end


"""
    make_grid(grid_lat, grid_lon)

Create 2D grids of latitude and longitude from 1D inputs.
"""
function make_grid(grid_lat, grid_lon)
    mesh_lat = repeat(grid_lat', length(grid_lon), 1)
    mesh_lon = repeat(grid_lon, 1, length(grid_lat))
    return mesh_lat, mesh_lon
end


"""
    subset_grid_values(extent, grid_lat, grid_lon, grid_values)

Returns `grid_values` subset to the latitudes and longitudes specified
in `extent`.

# Parameters
- `extent`: either `nothing` or a 4-element vector, `[lonmin, lonmax, latmin, latmax]`
  to subset the gridded values to.
- `grid_lat`: the 1D vector of latitudes for the grid.
- `grid_lon`: the 1D vector of longitudes for the grid.
- `grid_values`: the 2D array of gridded values to subset.

# Returns
If `extent` is `nothing`, then `grid_values` is returned directly.
Otherwise, the subset of `grid_values` falling within the defined extents
is returned.
"""
function subset_grid_values(extent, grid_lat, grid_lon, grid_values)
    if isnothing(extent)
        return grid_values
    end

    xx = (grid_lon .>= extent[1]) .& (grid_lon .<= extent[2])
    yy = (grid_lat .>= extent[3]) .& (grid_lat .<= extent[4])

    subset = grid_values[xx, yy]
    return subset
end


"""
    match_files(grid_dir)

Matches up gridded NO2 files and CO file under `grid_dir` by date.
Assumes that the files are in `no2` and `co` subdirectories of `grid_dir`.

# Returns

A dictionary with strings representing dates in YYYYMMDD format as keys
and a subdictionary with the NO2 and CO files under keys "no2" and "co".
"""
function match_files(grid_dir)
    no2_basenames = readdir(joinpath(grid_dir, "no2"))
    files = Dict()
    for no2_name in no2_basenames
        datestr = match(r"(\d{8})", no2_name)[1]
        co_name = replace(no2_name, "_no2_" => "_co_")
        tmp = Dict("no2" => joinpath(grid_dir, "no2", no2_name), "co" => joinpath(grid_dir, "co", co_name))
        if !isfile(tmp["co"])
            error("Missing expected CO gridded file, $(tmp["co"])")
        end
        files[datestr] = tmp
    end

    return files
end


"""
    match_scene(no2_file, scenes)

Given the list of `scenes` dictionaries as read from the TOML config,
return the one that includes the scene name in the NO2 file's base name.
Error if no such scene found.
"""
function match_scene(no2_file, scenes)
    bname = basename(no2_file)
    for scene in scenes
        if occursin(scene["name"], bname)
            return scene
        end
    end
    error("No scene found for NO2 file '$no2_file'")
end

end
