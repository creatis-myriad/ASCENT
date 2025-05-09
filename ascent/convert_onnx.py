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


class AscentONNX(ABC):
    """Abstract converter that converts trained Lightning's model to ONNX format for production."""

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
    @hydra.main(version_base="1.3", config_path="configs", config_name="onnx")
    def run_system(cfg: DictConfig) -> None:
        """Loads a model and converts it to ONNX format.

        Args:
            cfg (DictConfig): Configuration composed by Hydra.
        """

        def enable_running_stats(model) -> None:
            """Temporarily enable track_running_stats and initialize running statistics.

            Args:
                model (torch.nn.Module): The model to modify.
            """
            for module in model.modules():
                if isinstance(module, torch.nn.InstanceNorm2d) or isinstance(
                    module, torch.nn.InstanceNorm3d
                ):
                    num_features = module.num_features
                    module.running_mean = torch.zeros(num_features, device=model.device)
                    module.running_var = torch.ones(num_features, device=model.device)
                    module.track_running_stats = True

        # apply extra utilities
        # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
        utils.extras(cfg)

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        if cfg.get("ckpt_path"):
            log.info(f"Loading weights from {cfg.ckpt_path}")
            model.load_state_dict(
                torch.load(cfg.get("ckpt_path"), map_location=model.device)["state_dict"]
            )

        log.info("Converting to ONNX format...")
        filepath = Path(cfg.onnx_path)
        input_sample = torch.randn((1, model.net.in_channels, *list(model.net.patch_size)))
        model.eval()
        # enable_running_stats(model) # This solution is still buggy and needs to be fixed
        torch.onnx.export(
            model.net,
            input_sample,
            filepath.as_posix(),
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
        )
        # Make sure the converted model produces the same output as the original model
        with torch.no_grad():
            original_output = model(input_sample).numpy()
            ort_session = onnxruntime.InferenceSession(filepath)
            input_name = ort_session.get_inputs()[0].name
            ort_inputs = {input_name: input_sample.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            np.testing.assert_allclose(original_output, ort_outs[0], rtol=1e-03, atol=1e-05)
            log.info("Model converted successfully and outputs match.")
        log.info(f"Saving to '{filepath.as_posix()}'")


def main():
    """Run the script."""
    AscentONNX.main()


if __name__ == "__main__":
    main()
