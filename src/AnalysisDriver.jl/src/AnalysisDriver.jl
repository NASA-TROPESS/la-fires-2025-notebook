module AnalysisDriver

using PythonCall
using TOML

include("Utils.jl")
include("TropomiNO2.jl")
include("TropomiCO.jl")
include("DoGridding.jl")
include("Plotting.jl")
include("Enhancements.jl")

function end2end_driver(config_file, full_recheck=false, force_rerun=false, debug_level=0)
    pyssd = pyimport("sat_scene_download")
    cfg = load_config_file(config_file)
    pyssd.download_driver(cfg, full_recheck=full_recheck, force_redownload=force_rerun)

    l2_dir = cfg["files"]["l2_dir"]
    grid_dir = cfg["files"]["grid_dir"]
    plot_dir = cfg["files"]["plot_dir"]
    enhancement_dir = cfg["files"]["enhancement_dir"]
    for scene in cfg["scene"]
        gridding_driver.grid_no2_scene_from_config(
            scene_dict=scene,
            l2_dir=l2_dir,
            grid_dir=grid_dir,
            debug_level=debug_level
        )
        gridding_driver.grid_co_scene_from_config(
            scene_dict=scene,
            l2_dir=l2_dir,
            grid_dir=grid_dir,
            debug_level=debug_level
        )
        enhancements.create_enhancement_file(
            scene=scene,
            grid_dir=grid_dir,
            enhancement_dir=enhancement_dir
        )
    end



    plotting.plot_gridded_files(
        cfg["scene"],
        grid_dir,
        plot_dir,
    )
end

function load_config_file(config_file)
    config = TOML.parsefile(config_file)
    # Ensure that the paths under "files" are made relative to the configuration
    # file - joinpath keeps later absolute paths unchanged.
    cfg_dir = dirname(config_file)
    config["files"] = Dict(k => joinpath(cfg_dir, v) for (k, v) in config["files"])
    return config
end

end # module AnalysisDriver
