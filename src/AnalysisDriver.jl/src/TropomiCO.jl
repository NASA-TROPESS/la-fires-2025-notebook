module tropomi_co

using NCDatasets
using OversampledGridding

function tropomi_co_config()
    return GridConfig(
        space_dims=["scanline", "ground_pixel"],
        time_dim="time",
        loncorn_var="PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds",
        latcorn_var="PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds",
        keep_if_gt=Dict("PRODUCT/qa_value" => 0.3),
        vars_to_grid=Dict(
            "co" => "PRODUCT/carbonmonoxide_total_column_corrected",
            "co_uncorrected" => "PRODUCT/carbonmonoxide_total_column",
            "surface_pressure" => "PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure",
            "co_ak" => "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/column_averaging_kernel",
            "prior_co" => "PRODUCT/SUPPORT_DATA/INPUT_DATA/carbonmonoxide_profile_apriori",
            "layer_pressure" => get_co_layer_pressures,
        )
    )
end


function get_co_layer_pressures(ds)
    # The CO file gives the pressures at the bottom of each layer,
    # so we assume that the top of each layer is the bottom of the next
    # layer up, and that the top layer goes up to 0 hPa
    # Because Julia reads the dimension in reverse order to how they
    # show up in ncdump, the level dim is the first one, and it
    # is ordered space-to-surface

    dims = dimnames(ds.group["PRODUCT"].group["SUPPORT_DATA"].group["DETAILED_RESULTS"]["pressure_levels"])
    pbottom = ds.group["PRODUCT"].group["SUPPORT_DATA"].group["DETAILED_RESULTS"]["pressure_levels"][:, :, :, :]
    pmid = Array{eltype(pbottom)}(undef, size(pbottom))
    pmid[2:end, :, :, :] = 0.5 * (pbottom[1:end-1, :, :, :] + pbottom[2:end, :, :, :])
    pmid[1, :, :, :] = 0.5 * pbottom[1, :, :, :]
    return pmid, dims
end

end
