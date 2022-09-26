import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="torchaudio")

import hydra
from omegaconf import DictConfig


def preprocess_and_plan(cfg: DictConfig):
    """Preprocess dataset and plan experiments.

    Args:
        cfg: Configuration composed by Hydra.
    """

    print(f"Instantiating preprocessor <{cfg.preprocessor._target_}>")
    preprocessor = hydra.utils.instantiate(cfg.preprocessor)

    print("Start preprocessing...")
    preprocessor.run()

    if cfg.pl2d:
        print(f'Instantiating 2D planner <{cfg.planner["planner2d"]._target_}>')
        planner2d = hydra.utils.instantiate(cfg.planner["planner2d"])

    if cfg.pl3d:
        print(f'Instantiating 3D planner <{cfg.planner["planner3d"]._target_}>')
        planner3d = hydra.utils.instantiate(cfg.planner["planner3d"])

    if cfg.pl2d:
        planner2d.plan_experiment()

    if cfg.pl3d:
        planner3d.plan_experiment()


@hydra.main(version_base="1.2", config_path="../configs", config_name="preprocess_and_plan")
def main(cfg: DictConfig) -> None:
    preprocess_and_plan(cfg)


if __name__ == "__main__":
    main()