
module tropomi_no2

using Dates
using Interpolations
using NCDatasets
using OversampledGridding

function tropomi_no2_config(prior_file)
    return GridConfig(
        space_dims=["scanline", "ground_pixel"],
        time_dim="time",
        loncorn_var="PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds",
        latcorn_var="PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds",
        keep_if_gt=Dict("PRODUCT/qa_value" => 0.5),
        vars_to_grid=Dict(
            "no2_trop" => "PRODUCT/nitrogendioxide_tropospheric_column",
            "no2_trop_err" => "PRODUCT/nitrogendioxide_tropospheric_column_precision",
            "surface_pressure" => "PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_pressure",
            "layer_pressure" => get_tropomi_layer_pressures,
            "no2_ak" => "PRODUCT/averaging_kernel",
            "no2_amf_trop" => "PRODUCT/air_mass_factor_troposphere",
            "no2_amf_tot" => "PRODUCT/air_mass_factor_total"
        ),
        extra_vars=Dict(
            "time" => get_tropomi_time,
            "lon" => "PRODUCT/longitude",
            "lat" => "PRODUCT/latitude",
        ),
        extra_data_init_fxn=() -> no2_prior_data_init_fxn(prior_file),
        extra_data_fxn=(data) -> sample_no2_prior_data(data, prior_file)
    )
end

function get_tropomi_time(ds)
    n_across = ds.group["PRODUCT"].dim["ground_pixel"]
    time = ds.group["PRODUCT"]["delta_time"][:, :]
    time = reshape(time, 1, size(time, 1), size(time, 2))
    time = repeat(time, n_across)
    return time, ["ground_pixel", "scanline", "time"]
end

function get_tropomi_layer_pressures(ds)
    a = ds.group["PRODUCT"]["tm5_constant_a"][:, :]
    b = ds.group["PRODUCT"]["tm5_constant_b"][:, :]
    ps = ds.group["PRODUCT"].group["SUPPORT_DATA"].group["INPUT_DATA"]["surface_pressure"][:, :, :]
    dims = dimnames(ds.group["PRODUCT"].group["SUPPORT_DATA"].group["INPUT_DATA"]["surface_pressure"])
    n = size(a, 2)
    p1 = reshape(a[1, :], 1, 1, 1, n) .+ reshape(b[1, :], 1, 1, 1, n) .* ps
    p2 = reshape(a[2, :], 1, 1, 1, n) .+ reshape(b[2, :], 1, 1, 1, n) .* ps
    p = 0.5 .* (p1 .+ p2)
    return p, (dims..., "layer")
end


function no2_prior_data_init_fxn(prior_file)
    Dataset(prior_file) do ds
        nmid = ds.dim["lev"]
        nint = ds.dim["levi"]
        Dict("prior_no2" => [nmid], "prior_pmid" => [nmid], "prior_pedge" => [nint])
    end
end


function sample_no2_prior_data(pixel_data::PixelData, prior_file)
    pix_lon = pixel_data.extra_data["lon"]
    pix_lat = pixel_data.extra_data["lat"]
    pix_time = pixel_data.extra_data["time"]

    prior_data = load_no2_priors(prior_file)

    for (key, value) in prior_data.data
        this_interp_arr = interp_profiles_to_pixels(
            value,
            prior_data.lon,
            prior_data.lat,
            prior_data.time,
            pix_lon,
            pix_lat,
            pix_time,
        )
        pixel_data.data_to_grid["prior_$key"] = this_interp_arr
    end

    return pixel_data

end

struct NO2Priors
    time
    lat
    lon
    data
end

function load_no2_priors(file, scene_lonlim=[-180.0, 180.0], scene_latlim=[-90.0, 90.0])
    # Since the NO2 profiles are ~1 deg, include 1-2
    # extra grid cells to ensure we fully bound the pixels
    # in the scene
    lon1 = scene_lonlim[1] - 2.0
    lon2 = scene_lonlim[2] + 2.0
    lat1 = scene_latlim[1] - 2.0
    lat2 = scene_latlim[2] + 2.0

    Dataset(file) do ds
        lon = ds["lon"][:]
        lat = ds["lat"][:]
        date_components = ds["date"][:, :]
        time = [DateTime(date_components[:, i]...) for i in 1:size(date_components, 2)]

        a = ds["hyam"][:]
        ae = ds["hyai"][:]
        b = ds["hybm"][:]
        be = ds["hybi"][:]
        ps = ds["ps"][:, :, :]

        no2 = ds["no2"][:, :, :, :]

        xx = (lon .>= lon1) .& (lon .<= lon2)
        ix1 = max(findfirst(xx) - 1, 1)
        ix2 = min(findlast(xx) + 1, length(lon))
        yy = (lat .>= lat1) .& (lat .<= lat2)
        iy1 = max(findfirst(yy) - 1, 1)
        iy2 = min(findlast(yy) + 1, length(lat))

        lon = lon[ix1:ix2]
        lat = lat[iy1:iy2]
        ps = ps[ix1:ix2, iy1:iy2, :]
        no2 = no2[ix1:ix2, iy1:iy2, :, :]

        nx, ny, nt = size(ps)

        ps = reshape(ps, nx, ny, 1, nt)
        a = reshape(a, 1, 1, :, 1)
        b = reshape(b, 1, 1, :, 1)
        pmid = a .+ b .* ps

        ae = reshape(ae, 1, 1, :, 1)
        be = reshape(be, 1, 1, :, 1)
        pedge = ae .+ be .* ps

        NO2Priors(time, lat, lon, Dict("pmid" => pmid, "pedge" => pedge, "no2" => no2))
    end
end

function interp_profiles_to_pixels(profiles, prof_lon, prof_lat, prof_time, sat_lon, sat_lat, sat_time)
    nlay = size(profiles, 3)
    n = length(sat_time)
    pixel_profiles = zeros(n, nlay)

    pix_time_inds = find_pix_prior_time_inds(sat_time, prof_time)
    for t in unique(pix_time_inds)
        for k in 1:nlay
            # Only construct the interpolator once per time & level.
            # We will only interpolate into pixels for which this is
            # the closest time. We will do constant value extrapolation, which should
            # only be needed around the poles, and so is not likely to affect out results
            # in any case.
            interpolator = linear_interpolation((prof_lon, prof_lat), profiles[:, :, k, t], extrapolation_bc=Flat())
            for i in 1:n
                if pix_time_inds[i] == t
                    tmpx = sat_lon[i]
                    tmpy = sat_lat[i]
                    try
                        pixel_profiles[i, k] = interpolator(sat_lon[i], sat_lat[i])
                    catch err
                        println("interp to $(sat_lon[i]), $(sat_lat[i]), k = $k, t = $t, i = $i")
                        println("i=", i, " t=", t, " k=", k)
                        println(tmpx, " --> ", sat_lon[i])
                        println(tmpy, " --> ", sat_lat[i])
                        throw(err)
                    end
                end
            end
        end
    end

    return pixel_profiles
end

function find_pix_prior_time_inds(pixel_times, prior_times)
    inds = zeros(Int64, size(pixel_times))
    for (i, pix_t) in enumerate(pixel_times)
        _, inds[i] = findmin(abs.(prior_times .- pix_t))
    end
    return inds
end

end
