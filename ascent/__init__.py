import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ENV_ASCENT_HOME = "ASCENT_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


def get_ascent_root() -> Path:
    """Resolves the root directory for the `ASCENT` package.

    Returns:
        Path to the root directory for the `ASCENT` package.
    """
    return Path(__file__).resolve().parent


def get_ascent_home() -> Path:
    """Resolves the home directory for the `ASCENT` library, used to save/cache data reusable
    across scripts/runs.

    Returns:
        Path to the home directory for the `ASCENT` library.
    """
    load_dotenv(override=True)
    ascent_home = os.getenv(ENV_ASCENT_HOME)
    if ascent_home is None:
        user_cache_dir = os.getenv("XDG_CACHE_HOME", DEFAULT_CACHE_DIR)
        ascent_home = os.path.join(user_cache_dir, "ascent")
    return Path(ascent_home).expanduser()


def setup_root(
    project_root_env_var: bool = True,
    dotenv: bool = True,
    pythonpath: bool = True,
    cwd: bool = False,
) -> Path:
    """Find and setup the project root.

    Args:
        project_root_env_var (bool, optional): Whether to set ASCENT_PROJECT_ROOT environment variable.
        dotenv (bool, optional): Whether to load `.env` file.
        pythonpath (bool, optional): Whether to add project root to pythonpath.
        cwd (bool, optional): Whether to set current working directory to project root.

    Raises:
        FileNotFoundError: If root is not found.

    Returns:
        Path: Path to project root.
    """
    # Get the project root path
    path = str(get_ascent_root())

    if not os.path.exists(path):
        raise FileNotFoundError(f"Project root path does not exist: {path}")

    # Set the `ASCENT_PROJECT_ROOT` that will be used in Hydra default path config
    if project_root_env_var:
        os.environ["ASCENT_PROJECT_ROOT"] = path

    # Load any available `.env` file
    if dotenv:
        load_dotenv(override=True)

    # Add project root to pythonpath
    if pythonpath:
        sys.path.insert(0, path)

    # Set current working directory to project root
    if cwd:
        os.chdir(path)

    return path
