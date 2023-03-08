import hydra
import pyrootutils
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from ascent import utils

log = utils.get_pylogger(__name__)


def preprocess_and_plan(cfg: DictConfig) -> None:
    """Preprocess dataset and plan experiments.

    Args:
        cfg: Configuration composed by Hydra.
    """

    log.info(f"Instantiating preprocessor <{cfg.preprocessor._target_}>")
    preprocessor = hydra.utils.instantiate(cfg.preprocessor)

    log.info("Start preprocessing...")
    preprocessor.run()

    if cfg.pl2d:
        log.info(f'Instantiating 2D planner <{cfg.planner["planner2d"]._target_}>')
        planner2d = hydra.utils.instantiate(cfg.planner["planner2d"])
        planner2d.plan_experiment()

    if cfg.pl3d:
        log.info(f'Instantiating 3D planner <{cfg.planner["planner3d"]._target_}>')
        planner3d = hydra.utils.instantiate(cfg.planner["planner3d"])
        planner3d.plan_experiment()


@hydra.main(version_base="1.3.2", config_path="../configs", config_name="preprocess_and_plan")
def main(cfg: DictConfig) -> None:
    preprocess_and_plan(cfg)


if __name__ == "__main__":
    main()
