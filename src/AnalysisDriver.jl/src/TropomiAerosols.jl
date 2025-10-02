module tropomi_aerosols

using OversampledGridding

function tropomi_alh_config()
    return GridConfig(
        space_dims=["scanline", "ground_pixel"],
        time_dim="time",
        loncorn_var="PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds",
        latcorn_var="PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds",
        keep_if_gt=Dict("PRODUCT/qa_value" => 0.5),
        vars_to_grid=Dict(
            "alh" => "PRODUCT/aerosol_mid_height",
            "alh_prec" => "PRODUCT/aerosol_mid_height_precision",
            "alh" => "PRODUCT/aerosol_mid_pressure",
            "alh_prec" => "PRODUCT/aerosol_mid_pressure_precision",
        )
    )
end

function tropomi_aer_config()
    return GridConfig(
        space_dims=["scanline", "ground_pixel"],
        time_dim="", # there is no time dim on the variables we want
        loncorn_var="GEODATA/longitude_bounds",
        latcorn_var="GEODATA/latitude_bounds",
        keep_if_in=Dict("SCIDATA/FinalAlgorithmFlags" => [0, 1]),
        vars_to_grid=Dict(
            "aod" => "SCIDATA/FinalAerosolOpticalDepth",
            "ssa" => "SCIDATA/FinalAerosolSingleScattAlb",
        )
    )
end

end
