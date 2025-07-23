module enhancements

using AnalysisDriver
using Dates
using Mustache
using NCDatasets
using PolygonOps
using Statistics

function create_enhancement_file(; scene, grid_dir, enhancement_dir, quantile=0.1)
    mkpath(enhancement_dir)
    enh_file = joinpath(enhancement_dir, "enhancement_$(scene["name"]).nc")
    if !enh_file_needs_remade(enh_file, grid_dir, scene)
        @info "Not recreating enhancement file for $(scene["name"])"
        return
    else
        @info "Creating enhancement file for $(scene["name"])"
    end
    bck_selectors = Dict()
    bck_selectors["Time"] = make_btime_for_subscenes(grid_dir, scene)
    bck_selectors["Quantile"] = make_quant_for_subscenes(quantile, scene)
    # TODO: add in the background box as an option

    results = Dict()

    dates = nothing
    grid_lat = nothing
    grid_lon = nothing
    no2_vcds = nothing
    co_vcds = nothing

    for (key, selectors) in bck_selectors
        this_dates, this_lat, this_lon, this_no2_bck, this_co_bck = make_scene_backgrounds(grid_dir, scene, selectors)
        this_no2, this_no2_enh, this_co, this_co_enh, this_ratio = compute_enhancements(
            this_dates, this_lat, this_lon, this_no2_bck, this_co_bck, grid_dir, scene["name"]
        )
        results[key] = Dict(
            "no2_background" => this_no2_bck,
            "co_background" => this_co_bck,
            "no2_enhancement" => this_no2_enh,
            "co_enhancement" => this_co_enh,
            "enhancement_ratio" => this_ratio
        )

        if isnothing(dates)
            dates = this_dates
            grid_lat = this_lat
            grid_lon = this_lon
            no2_vcds = this_no2
            co_vcds = this_co
        elseif any(dates .!= this_dates)
            error("Inconsistent dates among background selectorrs")
        elseif !AnalysisDriver.utils.allclose(this_lat, grid_lat) || !AnalysisDriver.utils.allclose(this_lon, grid_lon)
            error("Inconsistent lats/lons among background selectors")
        elseif !AnalysisDriver.utils.allclose(no2_vcds, this_no2, equal_nan=true)
            error("Inconsistent NO2 VCDs among background selectors")
        elseif !AnalysisDriver.utils.allclose(co_vcds, this_co, equal_nan=true)
            error("Inconsistent CO VCDs among background selectors")
        end
    end

    save_enhancement_file(
        dates=dates,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        no2_vcds=no2_vcds,
        co_vcds=co_vcds,
        results=results,
        selectors=bck_selectors,
        scene=scene,
        enh_file=enh_file
    )

end


function save_enhancement_file(; dates, grid_lat, grid_lon, no2_vcds, co_vcds, results, selectors, scene, enh_file)
    Dataset(enh_file, "c") do ds
        defDim(ds, "date", length(dates))
        defDim(ds, "grid_lat", length(grid_lat))
        defDim(ds, "grid_lon", length(grid_lon))

        v = defVar(ds, "date", String, ("date",))
        v[:] = Dates.format.(dates, "yyyy-mm-dd")

        v = defVar(ds, "grid_lat", Float32, ("grid_lat",))
        v[:] = grid_lat

        v = defVar(ds, "grid_lon", Float32, ("grid_lon",))
        v[:] = grid_lon

        v = defVar(ds, "no2_trop_vcds", Float32, ("grid_lon", "grid_lat", "date"))
        v[:, :, :] = no2_vcds
        v.attrib["units"] = "mol m^-2"

        v = defVar(ds, "co_vcds", Float32, ("grid_lon", "grid_lat", "date"))
        v[:, :, :] = co_vcds
        v.attrib["units"] = "mol m^-2"

        special_units = Dict("enhancement_ratio" => "mmol mol^-1")
        for (key, res) in results
            grp = defGroup(ds, key)
            for (vname, varr) in res
                v = defVar(grp, vname, Float32, ("grid_lon", "grid_lat", "date"))
                v[:, :, :] = varr
                v.attrib["units"] = get(special_units, vname, "mol m^-2")
            end


            for (this_selector, subscene) in zip(selectors[key], scene["subscene"])
                add_background_vars!(grp, this_selector, subscene["name"])
            end
        end

    end
end


function make_btime_for_subscenes(grid_dir::AbstractString, scene)
    scene_name = scene["name"]
    scene_start = Date(scene["start"], dateformat"yyyy-mm-dd")
    fire_starts = [Date(subsc["fire_start"], dateformat"yyyy-mm-dd") for subsc in scene["subscene"]]
    file_pattern = joinpath(grid_dir, "{{Gas}}", "gridded_{{Gas}}_$(scene_name)_{{Date}}.nc")
    btimes = []
    for fire_start in fire_starts
        start = max(scene_start, fire_start - Day(7))
        this_btime = AnalysisDriver.enhancements.btime_from_dates(start, fire_start - Day(1), file_pattern)
        push!(btimes, this_btime)
    end
    return btimes
end

function make_quant_for_subscenes(quantile, scene)
    return [AnalysisDriver.enhancements.BackgroundQuantile(quantile) for _ in scene["subscene"]]
end

function make_scene_backgrounds(grid_dir, scene, selectors)
    no2_dates, no2_lat, no2_lon, no2_backgrounds = make_one_gas_scene_backgrounds(grid_dir, scene, "no2", "no2_trop", selectors)
    co_dates, co_lat, co_lon, co_backgrounds = make_one_gas_scene_backgrounds(grid_dir, scene, "co", "co", selectors)

    if any(no2_dates .!= co_dates)
        error("NO2 and CO have inconsistent dates in background calculation")
    end
    if !AnalysisDriver.utils.allclose(no2_lat, co_lat) || !AnalysisDriver.utils.allclose(no2_lon, co_lon)
        error("NO2 and CO have inconsistent latitudes in background calculation")
    end

    return no2_dates, no2_lat, no2_lon, no2_backgrounds, co_backgrounds
end

function make_one_gas_scene_backgrounds(grid_dir, scene, gas, gas_var, selectors)
    if length(selectors) != length(scene["subscene"])
        error("Number of background selectors must equal the number of subscenes")
    end

    name = scene["name"]
    start = Date(scene["start"], dateformat"yyyy-mm-dd")
    stop = Date(scene["end"], dateformat"yyyy-mm-dd")
    dates = collect(start:Day(1):stop)
    ndate = length(dates)

    backgrounds = nothing
    bck_lon = nothing
    bck_lat = nothing


    for (idate, date) in enumerate(dates)
        file = grid_file_name(grid_dir, gas, name, date)
        if !isfile(file)
            @warn "$file does not exist, skipping $gas enhancement calculation for $date"
            continue
        end
        lat, lon, cols = load_gridded_data(file, gas_var)
        if isnothing(backgrounds)
            nlon = length(lon)
            nlat = length(lat)
            backgrounds = fill(NaN, nlon, nlat, ndate)
            bck_lon = lon
            bck_lat = lat
        elseif !AnalysisDriver.utils.allclose(bck_lon, lon) || !AnalysisDriver.utils.allclose(bck_lat, lat)
            error("inconsistent $(gas) lat/lons across dates")
        end


        for (subscene, subscene_selector) in zip(scene["subscene"], selectors)
            # fire_start = Date(subscene["fire_start"], dateformat"yyyy-mm-dd")
            lonmin, lonmax, latmin, latmax = subscene["extent"]
            xx = (bck_lon .>= lonmin) .& (bck_lon .<= lonmax)
            yy = (bck_lat .>= latmin) .& (bck_lat .<= latmax)
            this_bck = get_background(subscene_selector, gas, bck_lat, bck_lon, cols, extent=subscene["extent"])
            backgrounds[xx, yy, idate] = this_bck[xx, yy]
        end
    end

    return dates, bck_lat, bck_lon, backgrounds
end

function compute_enhancements(dates, bck_lat, bck_lon, bck_no2, bck_co, grid_dir, scene_name)
    no2 = fill(NaN, size(bck_no2))
    no2_enhancements = fill(NaN, size(bck_no2))
    co = fill(NaN, size(bck_no2))
    co_enhancements = fill(NaN, size(bck_co))
    for (idate, date) in enumerate(dates)
        no2_file = grid_file_name(grid_dir, "no2", scene_name, date)
        if !isfile(no2_file)
            @warn "$no2_file does not exist, skipping enhancement calculations for $date"
            continue
        end
        no2_lat, no2_lon, no2_vcds = load_gridded_data(no2_file, "no2_trop")
        co_file = grid_file_name(grid_dir, "co", scene_name, date)
        if !isfile(co_file)
            @warn "$co_file does not exist, skipping enhancement calculations for $date"
            continue
        end
        co_lat, co_lon, co_vcds = load_gridded_data(co_file, "co")

        if !AnalysisDriver.utils.allclose(no2_lat, bck_lat) || !AnalysisDriver.utils.allclose(no2_lon, bck_lon)
            error("NO2 lat/lons on $(date) for $(scene_name) are inconsistent with the background lat/lons")
        end
        if !AnalysisDriver.utils.allclose(co_lat, bck_lat) || !AnalysisDriver.utils.allclose(co_lon, bck_lon)
            error("CO lat/lons on $(date) for $(scene_name) are inconsistent with the background lat/lons")
        end

        no2[:, :, idate] = no2_vcds
        no2_enhancements[:, :, idate] = no2_vcds .- bck_no2[:, :, idate]
        co[:, :, idate] = co_vcds
        co_enhancements[:, :, idate] = co_vcds .- bck_co[:, :, idate]
    end
    ratios = no2_enhancements ./ co_enhancements .* 1e3
    return no2, no2_enhancements, co, co_enhancements, ratios
end


"""
`BackgroundSelector` types are used to calculate a per-grid cell background
via the `get_background` function.
"""
abstract type BackgroundSelector end

"""
    function get_background(
        selector::BackgroundSelector,
        gas::AbstractString,
        grid_lat::AbstractArray,
        grid_lon::AbstractArray,
        grid_values::AbstractArray;
        extent=nothing
    )

This function defines the interface for a `BackgroundSelector`
"""
function get_background(
    selector::BackgroundSelector,
    gas::AbstractString,
    grid_lat::AbstractArray,
    grid_lon::AbstractArray,
    grid_values::AbstractArray{T};
    extent=nothing
)::AbstractArray{T} where {T<:AbstractFloat}

    error("Not implemented: concrete background selectors must implement this method")
end

function add_background_vars!(
    ds,
    selector::BackgroundSelector,
    subscene_name::AbstractString
)
end

"""
A background selector that takes values within a predefined box as the background.

If constructed directly, then the arguments are vectors of the latitude and longitude
corners of the box. It can also be constructed with the `bbox_from_width_and_angle` method.
"""
struct BackgroundBox <: BackgroundSelector
    latcorn::AbstractArray{Float64}
    loncorn::AbstractArray{Float64}
end

"""
    bbox_from_width_and_angle(base_lat, base_lon, angle, width, depth)

Construct a `BackgroundBox` that projects our from a given base latitude and longitude.
The resulting box will have one edge centered on `base_lat` and `base_lon` and extent
`depth` degrees out from those coordinates at `angle` degrees counterclockwise from east
with `width` perpendicular to that direction (i.e., the box's total area is `depth * width`).
"""
function bbox_from_width_and_angle(base_lat, base_lon, angle, width, depth)::BackgroundBox
    coords = [
        0.0 depth depth 0.0;
        width/2.0 width/2.0 -width/2.0 -width/2.0
    ]

    a = deg2rad(angle)
    r_mat = [cos(a) -sin(a); sin(a) cos(a)]
    coords = r_mat * coords
    loncorn = base_lon .+ coords[1, :]
    latcorn = base_lat .+ coords[2, :]
    BackgroundBox(latcorn, loncorn)
end


"""
    is_in_box(selector::BackgroundBox, grid_lat::AbstractArray, grid_lon::AbstractArray)

Helper function that returns a 2D boolean array indicating which grid points are within the
box defined by `selector`.
"""
function is_in_box(selector::BackgroundBox, grid_lat::AbstractArray, grid_lon::AbstractArray)
    mesh_lat, mesh_lon = AnalysisDriver.utils.make_grid(grid_lat, grid_lon)

    loncorn = zeros(Float64, length(selector.loncorn) + 1)
    loncorn[1:length(selector.loncorn)] = selector.loncorn
    loncorn[end] = selector.loncorn[1]

    latcorn = zeros(Float64, length(selector.latcorn) + 1)
    latcorn[1:length(selector.latcorn)] = selector.latcorn
    latcorn[end] = selector.latcorn[1]

    poly = [[x, y] for (x, y) in zip(loncorn, latcorn)]
    xx = [inpolygon((x, y), poly) for (x, y) in zip(mesh_lon[:], mesh_lat[:])]
    # 1 indicates in the polygon, opposed to outside on on the edge
    return reshape(xx .== 1, length(grid_lon), length(grid_lat))
end


"""
    get_background(
        selector::BackgroundBox,
        gas::AbstractString,
        grid_lat::AbstractArray,
        grid_lon::AbstractArray,
        grid_values::AbstractArray;
        extent=nothing
    )

Implementation of `get_background` for `BackgroundBox` types. The returned
background column amounts will be the mean of the values in the defined box.
(All values in the returned array will be the same.)
"""
function get_background(
    selector::BackgroundBox,
    gas::AbstractString,
    grid_lat::AbstractArray,
    grid_lon::AbstractArray,
    grid_values::AbstractArray{T};
    extent=nothing
)::AbstractArray{T} where {T<:AbstractFloat}

    xx = is_in_box(selector, grid_lat, grid_lon)
    gas_bckgnd = grid_values[xx]
    gas_bckgnd = mean(skipmissing(gas_bckgnd[.!isnan.(gas_bckgnd)]))
    backgrounds = fill(gas_bckgnd, size(grid_values))
    return backgrounds
end

function add_background_vars!(
    ds,
    selector::BackgroundBox,
    subscene_name::AbstractString,
)


    defDim(ds, "$(subscene_name)_background_box_corner", length(selector.loncorn))
    v = defVar(ds, "backgroun_box_loncorn", Float32, ("background_box_corner",))
    v[:] = selector.loncorn

    v = defVar(ds, "$(subscene_name)_backgroun_box_latcorn", Float32, ("background_box_corner",))
    v[:] = selector.latcorn
end


"""
A background selector that uses a mean from multiple days
before or after the event of interest as a background value.
This will normally be constructed with the `btime_from_dates`
method. If constructed directly, the parameters are:

- the 1D vector of grid latitudes,
- the 1D vector of grid longitudes,
- the gridded NO2 columns for the background, and
- the gridded CO columns for the background.
"""
struct BackgroundTime <: BackgroundSelector
    grid_lat
    grid_lon
    start_date
    end_date
    mean_no2_value
    mean_co_value
end

"""
    btime_from_dates(start_date, end_date, pattern)

Constructs a `BackgroundTime` instance by averaging gridded NO2 and CO data
within a given time range.

# Parameters
- `start_date`, `end_date`: the first and last dates to include in the average.
- `pattern`: a Mustache patterned path that will point to the gridded file to
  load for each date. `{{Date}}` will be replaced by the current date in `yyyymmdd`
  format. `{{Gas}}` will be replaced by the lowercase gas name.
"""
function btime_from_dates(start_date::Date, end_date::Date, pattern::AbstractString)::BackgroundTime
    grid_lat, grid_lon, mean_no2_value = _load_mean_gridded_value(start_date, end_date, "no2", pattern, "no2_trop")
    chk_lat, chk_lon, mean_co_value = _load_mean_gridded_value(start_date, end_date, "co", pattern, "co")
    if any(abs.(grid_lat - chk_lat) .> 1e-4) || any(abs.(grid_lon - chk_lon) .> 1e-4)
        error("no2 and co had different lat/lon grids")
    end
    BackgroundTime(grid_lat, grid_lon, start_date, end_date, mean_no2_value, mean_co_value)
end


"""
Helper function that averaged multiple days of gridded gas columns.
"""
function _load_mean_gridded_value(start_date, end_date, gas, pattern, gas_var)
    pattern = Mustache.parse(pattern)
    date = start_date
    counts = nothing
    running_sum = nothing
    grid_lats = nothing
    grid_lons = nothing

    while date <= end_date
        dstr = Dates.format(date, dateformat"yyyymmdd")
        file = Mustache.render(pattern, Dict("Date" => dstr, "Gas" => gas))

        this_lat, this_lon, this_col = Dataset(file) do ds
            lat = ds["grid_lat"][:]
            lon = ds["grid_lon"][:]
            col = nomissing(ds[gas_var][:, :], NaN)
            lat, lon, col
        end

        if isnothing(counts)
            counts = zeros(Int64, size(this_col))
            running_sum = zeros(eltype(this_col), size(this_col))
            grid_lats = this_lat
            grid_lons = this_lon
        else
            if size(counts) != size(this_col)
                error("inconsistent shapes")
            elseif any(abs.(this_lat - grid_lats) .> 1e-4) || any(abs.(this_lon - grid_lons) .> 1e-4)
                error("inconsistent coords")
            end
        end

        xx = .!isnan.(this_col)
        running_sum[xx] += this_col[xx]
        counts[xx] .+= 1

        date += Day(1)
    end

    mean_col = running_sum ./ counts
    # TODO: possibly fill in missing grid cells
    return grid_lats, grid_lons, mean_col
end

"""
    get_background(
        selector::BackgroundTime,
        gas::AbstractString,
        grid_lat::AbstractArray,
        grid_lon::AbstractArray,
        grid_values::AbstractArray;
        extent=nothing
    )

Implementation of `get_background` for `BackgroundTime` types. The returned
background column amounts will be the mean of their values during the time
period specified by the selector. (Each grid cell has its own unique value.)
"""
function get_background(
    selector::BackgroundTime,
    gas::AbstractString,
    grid_lat::AbstractArray,
    grid_lon::AbstractArray,
    grid_values::AbstractArray{T};
    extent=nothing
)::AbstractArray{T} where {T<:AbstractFloat}
    if !AnalysisDriver.utils.allclose(grid_lat, selector.grid_lat) || !AnalysisDriver.utils.allclose(grid_lon, selector.grid_lon)
        error("tried to apply BackgroundTime to data on a different lat/lon grid")
    end

    if gas == "no2"
        return selector.mean_no2_value
    elseif gas == "co"
        return selector.mean_co_value
    else
        error("gas $gas not implemented")
    end
end


function add_background_vars!(
    ds,
    selector::BackgroundTime,
    subscene_name::AbstractString
)
    ds.attrib["$(subscene_name)_background_date_range"] = "$(selector.start_date) to $(selector.end_date)"
    v = defVar(ds, "$(subscene_name)_no2_background", Float32, ("grid_lon", "grid_lat"))
    v[:, :,] = selector.mean_no2_value
    v.attrib["units"] = "mol m^-2"

    v = defVar(ds, "$(subscene_name)_co_background", Float32, ("grid_lon", "grid_lat"))
    v[:, :,] = selector.mean_co_value
    v.attrib["units"] = "mol m^-2"
end

"""
A background selector that takes a given quantile of the gridded
column data inside of the main extent as the background. This selector
should be constructed directly, with its only parameter being the
quantile to use (in the range 0.0 to 1.0).
"""
struct BackgroundQuantile <: BackgroundSelector
    q::AbstractFloat
end

"""
    get_background(
        selector::BackgroundQuantile,
        gas::AbstractString,
        grid_lat::AbstractArray,
        grid_lon::AbstractArray,
        grid_values::AbstractArray;
        extent=nothing
    )

Implementation of `get_background` for `BackgroundQuantile` types.
The returned background column amounts will be the qth quantile
of `grid_value` within `extent`. If `extent` is `nothing`, then
the quantile is computed over all of `grid_values`. (All values in
the returned array will be the same.)
"""
function get_background(
    selector::BackgroundQuantile,
    gas::AbstractString,
    grid_lat::AbstractArray,
    grid_lon::AbstractArray,
    grid_values::AbstractArray{T};
    extent=nothing
)::AbstractArray{T} where {T<:AbstractFloat}
    subset = AnalysisDriver.utils.subset_grid_values(extent, grid_lat, grid_lon, grid_values)
    bckgnd = NaN
    try
        bckgnd = quantile(skipmissing(subset[.!isnan.(subset)]), selector.q)
    catch err
        @error("When computing background quantile, got error: '$err'. Background will be a NaN for this date.")
    end
    bckgnd = fill(bckgnd, size(grid_values))
    return bckgnd
end


function add_background_vars!(
    ds,
    selector::BackgroundQuantile,
    subscene_name::AbstractString
)
    ds.attrib["$(subscene_name)_background_quantile"] = selector.q
end


function load_gridded_data(file, col_var)
    Dataset(file) do ds
        lon = ds["grid_lon"][:]
        lat = ds["grid_lat"][:]
        vcds = ds[col_var][:, :]
        lat, lon, vcds
    end
end

function grid_file_name(grid_dir::AbstractString, gas::AbstractString, scene_name::AbstractString, date::Date)
    dstr = Dates.format(date, "yyyymmdd")
    return joinpath(grid_dir, gas, "gridded_$(gas)_$(scene_name)_$(dstr).nc")
end

function grid_file_name(grid_dir::AbstractString, gas::AbstractString, scene_name::AbstractString, date::AbstractString)
    return joinpath(grid_dir, gas, "gridded_$(gas)_$(scene_name)_$(date).nc")
end

function enh_file_needs_remade(enh_file, grid_dir, scene)
    input_files = []

    start = Date(scene["start"], dateformat"yyyy-mm-dd")
    stop = Date(scene["end"], dateformat"yyyy-mm-dd")
    for date in start:Day(1):stop
        push!(input_files, grid_file_name(grid_dir, "no2", scene["name"], date))
        push!(input_files, grid_file_name(grid_dir, "co", scene["name"], date))
    end

    return AnalysisDriver.utils.check_mtime(enh_file, input_files)
end

end
