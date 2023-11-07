from abc import ABC

import hydra
from omegaconf import DictConfig

from ascent import setup_root, utils

log = utils.get_pylogger(__name__)


class Ascent_Preprocessor_Planner(ABC):
    @classmethod
    def main(cls) -> None:
        """Runs the requested experiment."""
        # Set up the environment
        cls.pre_run_routine()

        # Run the planning and preprocessing with `preprocess_and_plan` config loaded by @hydra.main
        cls.preprocess_and_plan()

    @classmethod
    def pre_run_routine(cls) -> None:
        """Sets-up the environment before running the training/testing."""
        # Load environment variables from `.env` file if it exists
        # Load before hydra main to allow for setting environment variables with ${oc.env:ENV_NAME}
        setup_root()

    @staticmethod
    @hydra.main(version_base="1.3", config_path="configs", config_name="preprocess_and_plan")
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


def main() -> None:
    """Run the script."""
    Ascent_Preprocessor_Planner.main()


if __name__ == "__main__":
    main()
