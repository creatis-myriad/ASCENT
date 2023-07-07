from typing import List, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from ascent import utils
from ascent.train import AscentTrainer

log = utils.get_pylogger(__name__)


class AscentEvaluator(AscentTrainer):
    """Ascent evaluator that runs the main testing loop using Lightning Trainer."""

    @staticmethod
    @hydra.main(version_base="1.3", config_path="configs", config_name="eval")
    @utils.task_wrapper
    def run_system(cfg: DictConfig) -> Tuple[dict, dict]:
        """Evaluates given checkpoint on a datamodule testset.

        This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
        failure. Useful for multiruns, saving info about the crash, etc.

        Args:
            cfg (DictConfig): Configuration composed by Hydra.

        Returns:
            Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.

        Raises:
            ValueError: Error when checkpoint path is not provided.
        """
        # apply extra utilities
        # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
        utils.extras(cfg)

        if not cfg.ckpt_path:
            raise ValueError("ckpt_path must not be provided!")

        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info("Instantiating loggers...")
        logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            log.info("Logging hyperparameters!")
            utils.log_hyperparameters(object_dict)

        log.info("Starting testing!")
        log.info(f"Using checkpoint: {cfg.ckpt_path}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

        metric_dict = trainer.callback_metrics

        return metric_dict, object_dict


def main():
    """Run the script."""
    AscentEvaluator.main()


if __name__ == "__main__":
    AscentEvaluator.main()
