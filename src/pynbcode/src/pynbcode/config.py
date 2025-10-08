import tomllib
from pathlib import Path
from typing import Any, Dict


def load_config_file(config_file: str) -> Dict[str, Any]:
    """
    Loads a TOML configuration file and resolves file paths.

    Paths found under the "files" key in the TOML file are made absolute 
    by resolving them relative to the configuration file's location.

    Args:
        config_file: The path to the TOML configuration file.

    Returns:
        A dictionary containing the parsed and processed configuration.
    """
    config_path = Path(config_file)
    cfg_dir = config_path.parent

    with config_path.open(mode="rb") as f:
        config = tomllib.load(f)

    # Ensure the "files" section exists before processing it
    if "files" in config and isinstance(config.get("files"), dict):
        # Use a dictionary comprehension to update the paths
        config["files"] = {
            key: str(cfg_dir / value)
            for key, value in config["files"].items()
        }

    return config
