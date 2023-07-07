from abc import ABC
from typing import List, Tuple

import hydra
import lightning as pl
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import CometLogger, Logger
from omegaconf import DictConfig

from ascent import setup_root, utils

log = utils.get_pylogger(__name__)


class AscentTrainer(ABC):
    """Abstract trainer that runs the main training loop using Lightning Trainer."""

    @classmethod
    def main(cls) -> None:
        """Runs the requested experiment."""
        # Set up the environment
        cls.pre_run_routine()

        # Run the system with config loaded by @hydra.main
        cls.run_system()

    @classmethod
    def pre_run_routine(cls) -> None:
        """Sets-up the environment before running the training/testing."""
        # Load environment variables from `.env` file if it exists
        # Load before hydra main to allow for setting environment variables with ${oc.env:ENV_NAME}
        setup_root()

    @staticmethod
    @hydra.main(version_base="1.3", config_path="configs", config_name="train")
    @utils.task_wrapper
    def run_system(cfg: DictConfig) -> Tuple[dict, dict]:
        """Trains the model. Can additionally evaluate on a testset, using best/last weights
        obtained during training.

        This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
        failure. Useful for multiruns, saving info about the crash, etc.

        Args:
            cfg (DictConfig): Configuration composed by Hydra.

        Returns:
            Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
        """
        # apply extra utilities
        # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
        utils.extras(cfg)

        # set seed for random number generators in pytorch, numpy and python.random
        if cfg.get("seed"):
            pl.seed_everything(cfg.seed, workers=True)

        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        if cfg.get("transfer_training") and cfg.get("ckpt_path"):
            log.info(f"Loading weights from {cfg.ckpt_path}")
            model.load_state_dict(
                torch.load(cfg.get("ckpt_path"), map_location=model.device)["state_dict"]
            )

        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

        log.info("Instantiating loggers...")
        logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            log.info("Logging hyperparameters!")
            utils.log_hyperparameters(object_dict)

        if cfg.get("compile"):
            log.info("Compiling model!")
            model = torch.compile(model)

        if cfg.get("train"):
            log.info("Starting training!")
            if not cfg.transfer_training:
                trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
            else:
                trainer.fit(model=model, datamodule=datamodule)

            if isinstance(trainer.logger, CometLogger) and cfg.comet_save_model:
                if trainer.checkpoint_callback.best_model_path:
                    trainer.logger.experiment.log_model(
                        "best-model", trainer.checkpoint_callback.best_model_path
                    )
                if trainer.checkpoint_callback.last_model_path:
                    trainer.logger.experiment.log_model(
                        "last-model", trainer.checkpoint_callback.last_model_path
                    )

        train_metrics = trainer.callback_metrics

        if cfg.get("test"):
            log.info("Starting testing!")
            if cfg.get("best_model"):
                ckpt_path = trainer.checkpoint_callback.best_model_path
                if ckpt_path == "":
                    log.warning("Best ckpt not found! Using current weights for testing...")
                    ckpt_path = None
                else:
                    log.info(f"Loading best ckpt: {ckpt_path}")
            else:
                ckpt_path = trainer.checkpoint_callback.last_model_path
                if ckpt_path == "":
                    log.warning("Last ckpt not found! Using current weights for testing...")
                    ckpt_path = None
                else:
                    log.info(f"Loading last ckpt: {ckpt_path}")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        test_metrics = trainer.callback_metrics

        # merge train and test metrics
        metric_dict = {**train_metrics, **test_metrics}

        return metric_dict, object_dict


def main():
    """Run the script."""
    AscentTrainer.main()


if __name__ == "__main__":
    main()
