import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
from einops.einops import rearrange
from torch import Tensor
from torchmetrics.functional import mean_squared_error

from ascent.models.nnunet_module import nnUNetLitModule
from ascent.preprocessing.preprocessing import check_anisotropy, get_lowres_axis, resample_image


class nnUNetRegLitModule(nnUNetLitModule):
    """`nnUNet` lightning module for regression.

    nnUNetRegLitModule is similar to nnUNetLitModule except for the loss (smooth L1) and evaluation
    metrics (MSE).
    """

    def __init__(self, **kwargs):
        """Initialize class instance.

        Args:
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(**kwargs)

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        img, label = batch["image"], batch["label"]

        # Only the highest resolution output is returned during the validation
        pred = self.forward(img)
        loss = self.loss(pred, label)

        val_metric = mean_squared_error(pred, label)

        self.validation_step_outputs.append({"val/loss": loss, "val/mse": val_metric})

        return {"val/loss": loss, "val/mse": val_metric}

    def on_validation_epoch_end(self):  # noqa: D102
        loss = self.metric_mean("val/loss", self.validation_step_outputs)
        metric = self.metric_mean("val/mse", self.validation_step_outputs)
        self.validation_step_outputs.clear()  # free memory

        self.all_val_eval_metrics.append(metric.item())

        self.update_eval_criterion_MA()
        self.maybe_update_best_val_eval_criterion_MA()

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )

        self.log(
            "val/mse",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )

        self.log(
            "val/mse_MA",
            self.val_eval_criterion_MA,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        img, label, image_meta_dict = batch["image"], batch["label"], batch["image_meta_dict"]

        start_time = time.time()
        preds = (
            self.tta_predict(img, apply_softmax=False)
            if self.hparams.tta
            else self.predict(img, apply_softmax=False)
        )
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

        test_mse = mean_squared_error(preds, label)

        self.log(
            "test/mse",
            test_mse,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )

        properties_dict = self.get_properties(image_meta_dict)

        if self.hparams.save_predictions:
            preds = preds.squeeze(0).cpu().detach().numpy()
            original_shape = properties_dict.get("original_shape")
            if len(preds.shape[1:]) == len(original_shape) - 1:
                preds = preds[..., None]
            if properties_dict.get("resampling_flag"):
                shape_after_cropping = properties_dict.get("shape_after_cropping")
                if check_anisotropy(properties_dict.get("original_spacing")):
                    anisotropy_flag = True
                    axis = get_lowres_axis(properties_dict.get("original_spacing"))
                elif check_anisotropy(properties_dict.get("spacing_after_resampling")):
                    anisotropy_flag = True
                    axis = get_lowres_axis(properties_dict.get("spacing_after_resampling"))
                else:
                    anisotropy_flag = False
                    axis = None

                if axis is not None:
                    if len(axis) == 2:
                        # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case
                        # we do not want to resample separately in the out of plane axis
                        anisotropy_flag = False

                preds = resample_image(preds, shape_after_cropping, anisotropy_flag, axis, 1, 0)

            final_preds = np.zeros([preds.shape[0], *original_shape])

            if len(properties_dict.get("crop_bbox")):
                box_start, box_end = properties_dict.get("crop_bbox")
                min_w, min_h, min_d = box_start
                max_w, max_h, max_d = box_end
                final_preds[:, min_w:max_w, min_h:max_h, min_d:max_d] = preds
            else:
                final_preds = preds

            if self.trainer.datamodule.hparams.test_splits:
                save_dir = os.path.join(self.trainer.default_root_dir, "testing_raw")
            else:
                save_dir = os.path.join(self.trainer.default_root_dir, "validation_raw")

            fname = properties_dict.get("case_identifier")
            spacing = properties_dict.get("original_spacing")

            final_preds = final_preds.squeeze(0)

            self.save_predictions(final_preds, fname, spacing, save_dir)

        self.test_step_outputs.append({"test/mse": test_mse})

        return {"test/mse": test_mse}

    def on_test_epoch_end(self):  # noqa: D102
        mean_mse = self.metric_mean("test/mse", self.test_step_outputs)
        self.test_step_outputs.clear()  # free memory

        self.log(
            "test/average_mse",
            mean_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int):  # noqa: D102
        img, image_meta_dict = batch["image"], batch["image_meta_dict"]

        start_time = time.time()
        preds = (
            self.tta_predict(img, apply_softmax=False)
            if self.hparams.tta
            else self.predict(img, apply_softmax=False)
        )
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

        properties_dict = self.get_properties(image_meta_dict)

        preds = preds.squeeze(0).cpu().detach().numpy()
        original_shape = properties_dict.get("original_shape")
        if len(preds.shape[1:]) == len(original_shape) - 1:
            preds = preds[..., None]
        if properties_dict.get("resampling_flag"):
            shape_after_cropping = properties_dict.get("shape_after_cropping")
            if check_anisotropy(properties_dict.get("original_spacing")):
                anisotropy_flag = True
                axis = get_lowres_axis(properties_dict.get("original_spacing"))
            elif check_anisotropy(properties_dict.get("spacing_after_resampling")):
                anisotropy_flag = True
                axis = get_lowres_axis(properties_dict.get("spacing_after_resampling"))
            else:
                anisotropy_flag = False
                axis = None

            if axis is not None:
                if len(axis) == 2:
                    # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case
                    # we do not want to resample separately in the out of plane axis
                    anisotropy_flag = False

            preds = resample_image(preds, shape_after_cropping, anisotropy_flag, axis, 1, 0)

        final_preds = np.zeros([preds.shape[0], *original_shape])

        if len(properties_dict.get("crop_bbox")):
            box_start, box_end = properties_dict.get("crop_bbox")
            min_w, min_h, min_d = box_start
            max_w, max_h, max_d = box_end
            final_preds[:, min_w:max_w, min_h:max_h, min_d:max_d] = preds
        else:
            final_preds = preds

        save_dir = os.path.join(self.trainer.default_root_dir, "inference_raw")

        fname = properties_dict.get("case_identifier")
        spacing = properties_dict.get("original_spacing")

        final_preds = final_preds.squeeze(0)

        self.save_predictions(final_preds, fname, spacing, save_dir)

    @staticmethod
    def get_properties(image_meta_dict: dict) -> OrderedDict:
        """Convert values in image meta dictionary loaded from torch.tensor to normal list/boolean.

        Args:
            image_meta_dict: Dictionary containing image meta information.

        Returns:
            Converted properties dictionary.
        """
        properties_dict = OrderedDict()
        properties_dict["original_shape"] = image_meta_dict["original_shape"][0].tolist()
        properties_dict["resampling_flag"] = image_meta_dict["resampling_flag"].item()
        properties_dict["shape_after_cropping"] = image_meta_dict["shape_after_cropping"][
            0
        ].tolist()
        properties_dict["anisotropy_flag"] = image_meta_dict["anisotropy_flag"].item()
        if len(image_meta_dict["crop_bbox"]):
            properties_dict["crop_bbox"] = image_meta_dict["crop_bbox"][0].tolist()
        else:
            properties_dict["crop_bbox"] = []
        properties_dict["case_identifier"] = image_meta_dict["case_identifier"][0]
        properties_dict["original_spacing"] = image_meta_dict["original_spacing"][0].tolist()

        return properties_dict

    def save_predictions(
        self, preds: np.ndarray, fname: str, spacing: np.ndarray, save_dir: Union[str, Path]
    ) -> None:
        """Save segmentation mask to the given save directory.

        Args:
            preds: Predicted segmentation mask.
            fname: Filename to save.
            spacing: Spacing to save the segmentation mask.
            save_dir: Directory to save the segmentation mask.
        """
        print(f"Saving prediction for {fname}...")

        os.makedirs(save_dir, exist_ok=True)

        preds = preds.astype(np.float32)
        itk_image = sitk.GetImageFromArray(rearrange(preds, "w h d ->  d h w"))
        itk_image.SetSpacing(spacing)
        sitk.WriteImage(itk_image, os.path.join(save_dir, fname + ".nii.gz"))


if __name__ == "__main__":
    from typing import List

    import hydra
    import omegaconf
    import pyrootutils
    from hydra import compose, initialize
    from lightning import Callback, LightningDataModule, LightningModule, Trainer
    from omegaconf import OmegaConf

    from ascent import utils

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    with initialize(version_base="1.2", config_path="../../configs/model"):
        cfg = compose(config_name="unwrap_2d.yaml")
        print(OmegaConf.to_yaml(cfg))

    # cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "nnunet.yaml")
    cfg.scheduler.max_decay_steps = 1000
    # cfg.net.in_channels = 3
    # cfg.net.num_classes = 1
    # cfg.net.patch_size = [40, 192]
    # cfg.net.kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    # cfg.net.strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]]
    nnunet: LightningModule = hydra.utils.instantiate(cfg)

    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "nnunet.yaml")
    cfg.data_dir = str(root / "data")
    cfg.dataset_name = "UNWRAP"
    cfg.patch_size = [40, 192]
    cfg.do_dummy_2D_data_aug = False
    cfg.in_channels = 3
    # cfg.patch_size = [128, 128, 12]
    cfg.batch_size = 2
    cfg.fold = 0
    camus_datamodule: LightningDataModule = hydra.utils.instantiate(cfg)

    cfg = omegaconf.OmegaConf.load(root / "configs" / "callbacks" / "nnunet.yaml")
    cfg.model_checkpoint.monitor = "val/mse_MA"
    cfg.model_checkpoint.mode = "max"
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg)

    # cfg = omegaconf.OmegaConf.load(root / "configs" / "trainer" / "nnunet.yaml")
    # trainer: Trainer = hydra.utils.instantiate(cfg, callbacks=callbacks)
    trainer = Trainer(
        max_epochs=2,
        deterministic=False,
        limit_train_batches=20,
        limit_val_batches=10,
        limit_test_batches=2,
        gradient_clip_val=12,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(model=nnunet, datamodule=camus_datamodule)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    print("Starting testing!")
    # ckpt_path = "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/logs/lightning_logs/version_0/checkpoints/epoch=1-step=500.ckpt"
    trainer.test(model=nnunet, datamodule=camus_datamodule, ckpt_path=ckpt_path)
