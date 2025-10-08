module APlots

using Dates
using NCDatasets
using PythonCall
using PythonPlot

STD_CO_ENH = 0.03  # mol/m2

(ccrs, cfeat, cio) = pyimport("cartopy" => ("crs", "feature", "io"))

struct PlumeMask
    mask::AbstractArray{Bool}
    lat::AbstractArray{<:AbstractFloat}
    lon::AbstractArray{<:AbstractFloat}
    dates::AbstractArray{Date}
end

function PlumeMask(enhancement_file::AbstractString; enhancement_group::AbstractString="Time", co_enhancement_threshold::AbstractFloat=STD_CO_ENH)::PlumeMask
    Dataset(enhancement_file) do ds
        dates = [Date(d, "yyyy-mm-dd") for d in ds["date"][:]]
        lat = ds["grid_lat"][:]
        lon = ds["grid_lon"][:]
        co_enh = ds.group[enhancement_group]["co_enhancement"][:, :, :]
        mask = co_enh .< co_enhancement_threshold
        PlumeMask(mask, lat, lon, dates)
    end
end

function plot_gridded_data(
    grid_file::AbstractString,
    variable::AbstractString,
    level_idx::Union{Integer,Nothing}=nothing;
    vscale::AbstractFloat=1.0,
    cblabel::AbstractString="",
    extent=nothing,
    plume_mask::Union{PlumeMask,Nothing}=nothing,
    ax=nothing,
    pcolormesh_kw...
)

    Dataset(grid_file) do ds
        lon = ds["grid_lon"][:]
        lat = ds["grid_lat"][:]


        var = ds[variable]
        is_3d = length(dimnames(var)) == 3
        values = if is_3d && isnothing(level_idx)
            error("$var is 3D, a level index must be passed")
        elseif is_3d
            var[:, :, level_idx]
        else
            var[:, :]
        end

        if !isnothing(plume_mask)
            # This assumes we're plotting one of the gridded files created
            # by running the main driver, which will have the date in its name.
            dstr = match(r"\d{8}", grid_file).match
            inferred_date = Date(dstr, "yyyymmdd")
            apply_plume_mask!(values, lon, lat, inferred_date, plume_mask)
        end


        (lon_inds, lat_inds) = if !isnothing(extent)
            xx = (lon .>= extent[1]) .& (lon .<= extent[2])
            yy = (lat .>= extent[3]) .& (lat .<= extent[4])
            (xx, yy)
        else
            xx = trues(size(lon))
            yy = trues(size(lat))
            (xx, yy)
        end

        lon = lon[lon_inds]
        lat = lat[lat_inds]
        values = vscale .* values[lon_inds, lat_inds]

        if isnothing(ax)
            _, ax = subplots(subplot_kw=Dict("projection" => ccrs.PlateCarree()))
        end

        h = ax.pcolormesh(lon, lat, values'; pcolormesh_kw...)
        cblabel = length(cblabel) == 0 ? variable : cblabel
        colorbar(h, label=cblabel, ax=ax)
        add_features(ax)
    end
end

function apply_plume_mask!(values, val_lon, val_lat, val_date, mask::PlumeMask; fill_val=NaN)
    if any(abs.(val_lon .- mask.lon) .> 1e-4) || any(abs.(val_lat .- mask.lat) .> 1e-4)
        # TODO: subset the mask?
        error("Value lat/lon and mask lat/lon are not consistent")
    end

    idate = findfirst(mask.dates .== val_date)
    if isnothing(idate)
        error("Date $val_date not found in the given mask")
    end

    mask_arr = mask.mask[:, :, idate]
    values[mask_arr] .= fill_val
end

function add_features(ax; gridlines=true, land=false, ocean=false, states=true, counties=true)
    if land
        ax.add_feature(cfeat.LAND)
    end
    if ocean
        ax.add_feature(cfeat.OCEAN)
    end
    if states
        ax.add_feature(cfeat.STATES)
    end
    if counties
        shp_fn = cio.shapereader.natural_earth(resolution="10m", category="cultural", name="admin_2_counties")
        shp = cio.shapereader.Reader(shp_fn)
        counties = collect(shp.geometries())
        feature = cfeat.ShapelyFeature(counties, ccrs.PlateCarree())
        ax.add_feature(feature, facecolor="none", edgecolor="gray")
    end
    if gridlines
        gl = ax.gridlines(draw_labels=true)
        gl.right_labels = false
        gl.top_labels = false
    end
end

end
