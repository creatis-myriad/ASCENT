from abc import ABC
from pathlib import Path

import hydra
import numpy as np
import onnxruntime
import torch
from lightning import LightningModule
from omegaconf import DictConfig

from ascent import setup_root, utils

log = utils.get_pylogger(__name__)


class AscentTorchScript(ABC):
    """Abstract converter that converts trained Lightning's model to TorchScript format for
    production."""

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
    @hydra.main(version_base="1.3", config_path="configs", config_name="torchscript")
    def run_system(cfg: DictConfig) -> None:
        """Loads a model and converts it to ONNX format.

        Args:
            cfg (DictConfig): Configuration composed by Hydra.
        """

        # apply extra utilities
        # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
        utils.extras(cfg)

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        if cfg.get("ckpt_path"):
            log.info(f"Loading weights from {cfg.ckpt_path}")
            model.load_state_dict(
                torch.load(cfg.get("ckpt_path"), map_location=model.device, weights_only=False)[
                    "state_dict"
                ]
            )

        log.info("Converting to TorchScript format...")
        input_sample = torch.randn((1, model.net.in_channels, *list(model.net.patch_size)))

        save_dir = Path(cfg.get("save_dir"))
        save_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        if cfg.get("cpu"):
            traced_model = torch.jit.trace(model.net, input_sample)
            torch.jit.save(
                traced_model,
                save_dir / f"{cfg.get('output_name')}_cpu.pt",
            )

        if cfg.get("gpu"):
            traced_model = torch.jit.trace(
                model.net.to(torch.device("cuda")), input_sample.to(torch.device("cuda"))
            )
            torch.jit.save(traced_model, save_dir / f"{cfg.get('output_name')}_gpu.pt")

        # Make sure the converted model produces the same output as the original model
        with torch.no_grad():
            model.to(torch.device("cpu"))
            original_output = model(input_sample).numpy()
            if cfg.get("cpu"):
                traced_model = torch.jit.load(save_dir / f"{cfg.get('output_name')}_cpu.pt")
                cpu_output = traced_model(input_sample.cpu()).detach().cpu().numpy()
                np.testing.assert_allclose(original_output, cpu_output, rtol=1e-03, atol=1e-05)
            if cfg.get("gpu"):
                traced_model = torch.jit.load(save_dir / f"{cfg.get('output_name')}_gpu.pt")
                gpu_output = (
                    traced_model(input_sample.to(torch.device("cuda"))).detach().cpu().numpy()
                )
                np.testing.assert_allclose(original_output, gpu_output, rtol=1e-03, atol=1e-05)
            log.info("Model converted successfully and outputs match.")
        log.info(f"Model saved to '{save_dir.as_posix()}'")


def main():
    """Run the script."""
    AscentTorchScript.main()


if __name__ == "__main__":
    main()
