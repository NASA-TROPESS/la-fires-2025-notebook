module auto_plotting

using AnalysisDriver
using NCDatasets
using PythonCall
using PythonPlot


function plot_gridded_files(scenes, grid_dir, plot_dir)
    mkpath(plot_dir)
    matched_files = AnalysisDriver.utils.match_files(grid_dir)
    scene = nothing  # needed to expand the scope of scene outside the try block
    for (datestr, files) in matched_files
        try
            scene = AnalysisDriver.utils.match_scene(files["no2"], scenes)
        catch e
            @warn "$e, will not plot gridded files"
            continue
        end
        # only make plots if found a scene - this way we can use configs
        # that are a subset of the full
        save_as = joinpath(plot_dir, "$(scene["name"])_$datestr.png")
        if AnalysisDriver.utils.check_mtime(save_as, collect(values(files)))
            plot_scene(datestr, files["no2"], files["co"], scene, save_as=save_as)
        end
    end
end


function plot_scene(datestr, no2_file, co_file, scene_dict; save_as=nothing)
    ccrs = pyimport("cartopy" => "crs")
    no2_grid = Dataset(no2_file) do ds
        Dict("vcd" => ds["no2_trop"][:, :], "lon" => ds["grid_lon"][:], "lat" => ds["grid_lat"][:])
    end
    co_grid = Dataset(co_file) do ds
        Dict("vcd" => ds["co"][:, :], "lon" => ds["grid_lon"][:], "lat" => ds["grid_lat"][:])
    end

    vcd_conv = 6.022e23 / 100^2
    fig, axs = subplots(1, 3, figsize=(18, 4), subplot_kw=Dict("projection" => ccrs.PlateCarree()))
    no2_clim = get(scene_dict, "no2_clim", (0.0, 2e16))
    _pcolor_helper(
        axs[0], no2_grid["lon"], no2_grid["lat"], vcd_conv .* no2_grid["vcd"]',
        cmap="Greens", cblabel="Trop. NO\$_2\$ (molec. cm\$^{-2}\$)", vmin=no2_clim[1], vmax=no2_clim[2]
    )

    co_clim = get(scene_dict, "co_clim", (0.0, 3e18))
    _pcolor_helper(
        axs[1], co_grid["lon"], no2_grid["lat"], vcd_conv .* co_grid["vcd"]',
        cmap="Oranges", cblabel="Total CO (molec. cm\$^{-2}\$)", vmin=co_clim[1], vmax=co_clim[2]
    )

    if !AnalysisDriver.utils.allclose(no2_grid["lon"], co_grid["lon"])
        error("Not implemented: different NO2 and CO longitude grids")
    end
    if !AnalysisDriver.utils.allclose(no2_grid["lat"], co_grid["lat"])
        error("Not implemented: different NO2 and CO latitude grids")
    end
    ratio = 1e3 .* no2_grid["vcd"] ./ co_grid["vcd"]
    ratio_clim = get(scene_dict, "ratio_clim", (0.0, 20.0))
    _pcolor_helper(
        axs[2], no2_grid["lon"], no2_grid["lat"], ratio',
        cmap="rainbow", cblabel="NO\$_2\$ / CO \$\\times\$ 1000", vmin=ratio_clim[1], vmax=ratio_clim[2]
    )
    scene_name = scene_dict["name"]
    suptitle("$scene_name - $datestr")

    if !isnothing(save_as)
        savefig(save_as, bbox_inches="tight")
        @info "Saved plot $save_as"
    end
    plotclose()
end


function _pcolor_helper(ax, lon, lat, vcd; cmap, cblabel, vmin=nothing, vmax=nothing)
    cfeat = pyimport("cartopy" => "feature")
    ex_bottom = isnothing(vmin) ? false : vmin > minimum(vcd[.!isnan.(vcd)])
    ex_top = isnothing(vmax) ? false : vmax < maximum(vcd[.!isnan.(vcd)])
    extend = "neither"
    if ex_bottom && ex_top
        extend = "both"
    elseif ex_bottom
        extend = "min"
    elseif ex_top
        extend = "max"
    end

    h = ax.pcolormesh(lon, lat, vcd, cmap=cmap, vmin=vmin, vmax=vmax)
    colorbar(h, ax=ax, label=cblabel, extend=extend)
    ax.add_feature(cfeat.COASTLINE)
    ax.add_feature(cfeat.LAND)
    ax.add_feature(cfeat.OCEAN)
    ax.add_feature(cfeat.LAKES)

    gl = ax.gridlines(draw_labels=true)
    gl.right_labels = false
    gl.top_labels = false
end




end
