import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from einops.einops import rearrange
from monai.data import MetaTensor
from pytorch_lightning import LightningModule
from skimage.transform import resize
from torch import Tensor
from torchmetrics.functional import mean_squared_error

from ascent.datamodules.components.inferers import sliding_window_inference


class nnUNetDealiasLitModule(LightningModule):
    """nnUNet lightning module for deep unfolding to perform dealiasing.

    Similar to nnUNetRegLitModule except for the definition of class's variables: threeD,
    patch_size, num_classes as the U-Net is no longer equals self.net but self.net.denoiser.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        tta: bool = True,
        sliding_window_overlap: float = 0.5,
        sliding_window_importance_map: bool = "gaussian",
        save_predictions: bool = True,
        save_npz: bool = False,
    ):
        """Saves the system's configuration in `hparams`. Initialize variables for training and
        validation loop.

        Args:
            net: Network architecture.
            optimizer: Optimizer.
            loss: Loss function.
            scheduler: Scheduler for training.
            tta: Whether to use the test time augmentation, i.e. flip.
            sliding_window_overlap: Minimum overlap for sliding window inference.
            sliding_window_importance_map: Importance map used for sliding window inference.
            save_prediction: Whether to save the test predictions.
        """
        super().__init__()
        # ignore net and loss as they are nn.module and will be saved automatically
        self.save_hyperparameters(logger=False, ignore=["net", "loss"])

        self.net = net
        self.threeD = len(self.net.denoiser.patch_size) == 3
        self.patch_size = list(self.net.denoiser.patch_size)

        self.num_classes = self.net.denoiser.num_classes

        # declare a dummy input for display model summary
        self.example_input_array = torch.rand(1, 2, *self.patch_size, device=self.device)

        # loss function smooth L1
        self.loss = loss

        # parameter alpha for calculating moving average dice -> alpha * old + (1-alpha) * new
        self.val_eval_criterion_alpha = 0.9

        # current moving average dice
        self.val_eval_criterion_MA = None

        # best moving average dice
        self.best_val_eval_criterion_MA = None

        # list to store all the moving average dice during the training
        self.all_val_eval_metrics = []

        if self.hparams.tta:
            self.tta_flips = self.get_tta_flips()
        self.test_idx = 0
        self.test_imgs = []

    def forward(self, img: Union[Tensor, MetaTensor]) -> Union[Tensor, MetaTensor]:  # noqa: D102
        return self.net(img)

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        img, label = batch["image"], batch["label"]

        # Need to handle carefully the multi-scale outputs from deep supervision heads
        pred = self.forward(img)
        loss = self.compute_loss(pred, label)

        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )
        return {"loss": loss}

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        img, label = batch["image"], batch["label"]

        # Only the highest resolution output is returned during the validation
        pred = self.forward(img)
        loss = self.loss(pred, label)

        val_metric = mean_squared_error(pred, label)

        return {"val/loss": loss, "val/mse": val_metric}

    def validation_epoch_end(self, validation_step_outputs: dict[str, Tensor]):  # noqa: D102
        loss = self.metric_mean("val/loss", validation_step_outputs)
        metric = self.metric_mean("val/mse", validation_step_outputs)

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
        preds = self.tta_predict(img) if self.hparams.tta else self.predict(img)
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
                preds = self.recovery_prediction(
                    preds, shape_after_cropping, properties_dict.get("anisotropy_flag")
                )

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

        return {"test/mse": test_mse}

    def test_epoch_end(self, test_step_outputs: dict[str, Tensor]):  # noqa: D102
        mean_dice = self.metric_mean("test/mse", test_step_outputs)
        self.log(
            "test/average_mse",
            mean_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int):  # noqa: D102
        img, image_meta_dict = batch["image"], batch["image_meta_dict"]

        start_time = time.time()
        preds = self.tta_predict(img) if self.hparams.tta else self.predict(img)
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

        properties_dict = self.get_properties(image_meta_dict)

        preds = preds.squeeze(0).cpu().detach().numpy()
        original_shape = properties_dict.get("original_shape")
        if len(preds.shape[1:]) == len(original_shape) - 1:
            preds = preds[..., None]
        if properties_dict.get("resampling_flag"):
            shape_after_cropping = properties_dict.get("shape_after_cropping")
            preds = self.recovery_prediction(
                preds, shape_after_cropping, properties_dict.get("anisotropy_flag")
            )

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

    def configure_optimizers(self) -> dict[Literal["optimizer", "lr_scheduler"], Any]:
        """Configures optimizers/LR schedulers.

        Returns:
            A dict with an `optimizer` key, and an optional `lr_scheduler` if a scheduler is used.
        """
        configured_optimizer = {"optimizer": self.hparams.optimizer(params=self.parameters())}
        if self.hparams.scheduler is not None:
            configured_optimizer["lr_scheduler"] = self.hparams.scheduler(
                optimizer=configured_optimizer["optimizer"]
            )
        return configured_optimizer

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Save extra information in checkpoint, i.e. the evaluation metrics for all epochs.

        Args:
            checkpoint: Checkpoint dictionary.
        """
        checkpoint["all_val_eval_metrics"] = self.all_val_eval_metrics

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Load information from checkpoint to class attribute, i.e. the evaluation metrics for all
        epochs.

        Args:
            checkpoint: Checkpoint dictionary.
        """
        self.all_val_eval_metrics = checkpoint["all_val_eval_metrics"]

    def compute_loss(
        self, preds: Union[Tensor, MetaTensor], label: Union[Tensor, MetaTensor]
    ) -> float:
        """Compute the multi-scale loss if deep supervision is set to True.

        Args:
            preds: Predicted logits.
            label: Ground truth label.

        Returns:
            Train loss.
        """
        if self.net.denoiser.deep_supervision:
            loss = self.loss(preds[0], label)
            for i, pred in enumerate(preds[1:]):
                downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(pred, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return self.loss(preds, label)

    def predict(self, image: Union[Tensor, MetaTensor]) -> Union[Tensor, MetaTensor]:
        """Predict 2D/3D images with sliding window inference.

        Args:
            image: Image to predict.

        Returns:
            Logits of prediction.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
            ValueError: If 3D patch is requested to predict 2D images.
        """
        if len(image.shape) == 5:
            if len(self.patch_size) == 3:
                return self.predict_3D_3Dconv_tiled(image)
            elif len(self.patch_size) == 2:
                return self.predict_3D_2Dconv_tiled(image)
            else:
                raise NotImplementedError
        if len(image.shape) == 4:
            if len(self.patch_size) == 2:
                return self.predict_2D_2Dconv_tiled(image)
            elif len(self.patch_size) == 3:
                raise ValueError("You can't predict a 2D image with 3D model. You dummy.")
            else:
                raise NotImplementedError

    def tta_predict(self, image: Union[Tensor, MetaTensor]) -> Union[Tensor, MetaTensor]:
        """Predict with test time augmentation.

        Args:
            image: Image to predict.

        Returns:
            Logits averaged over the number of flips.
        """
        preds = self.predict(image)
        for flip_idx in self.tta_flips:
            preds += torch.flip(self.predict(torch.flip(image, flip_idx)), flip_idx)
        preds /= len(self.tta_flips) + 1
        return preds

    def predict_2D_2Dconv_tiled(
        self, image: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor]:
        """Predict 2D image with 2D model.

        Args:
            image: Image to predict.

        Returns:
            Logits of prediction.

        Raises:
            ValueError: If image is not 2D.
        """
        if not len(image.shape) == 4:
            raise ValueError("image must be (b, c, w, h)")
        return self.sliding_window_inference(image)

    def predict_3D_3Dconv_tiled(
        self, image: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor]:
        """Predict 3D image with 3D model.

        Args:
            image: Image to predict.

        Returns:
            Logits of prediction.

        Raises:
            ValueError: If image is not 3D.
        """
        if not len(image.shape) == 5:
            raise ValueError("image must be (b, c, w, h, d)")
        return self.sliding_window_inference(image)

    def predict_3D_2Dconv_tiled(
        self, image: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor]:
        """Predict 3D image with 2D model.

        Args:
            image: Image to predict.

        Returns:
            Logits of prediction.

        Raises:
            ValueError: If image is not 3D.
        """
        if not len(image.shape) == 5:
            raise ValueError("image must be (b, c, w, h, d)")
        preds_shape = (image.shape[0], self.num_classes, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[-1]):
            preds[..., depth] = self.predict_2D_2Dconv_tiled(image[..., depth])
        return preds

    @staticmethod
    def recovery_prediction(
        prediction: np.array,
        new_shape: Union[tuple, list],
        anisotropy_flag: bool,
    ) -> np.array:
        """Recover prediction to its original shape in case of resampling.

        Args:
            prediciton: Predicted logits. (c, W, H, D)
            new_shape: Shape for resampling. (W, H, D)
            anisotropy_flag: Whether to use anisotropic resampling.

        Returns:
            (c, W, H, D) Resampled prediction.
        """
        shape = np.array(prediction[0].shape)
        if np.any(shape != np.array(new_shape)):
            resized_channels = []
            if anisotropy_flag:
                for image_c in prediction:
                    resized_slices = []
                    for i in range(image_c.shape[-1]):
                        image_c_2d_slice = image_c[:, :, i]
                        image_c_2d_slice = resize(
                            image_c_2d_slice,
                            new_shape[:-1],
                            order=1,
                            mode="edge",
                            cval=0,
                            clip=True,
                            anti_aliasing=False,
                        )
                        resized_slices.append(image_c_2d_slice)
                    resized = np.stack(resized_slices, axis=-1)
                    resized = resize(
                        resized,
                        new_shape,
                        order=0,
                        mode="constant",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    resized_channels.append(resized)
            else:
                for image_c in prediction:
                    resized = resize(
                        image_c,
                        new_shape,
                        order=3,
                        mode="edge",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    resized_channels.append(resized)
            reshaped = np.stack(resized_channels, axis=0)
            return reshaped
        else:
            return prediction

    def get_tta_flips(self) -> list[list[int]]:
        """Get the all possible flips for test time augmentation.

        Returns:
            List of axes to flip an 2D or 3D image.
        """
        if self.threeD:
            return [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        else:
            return [[2], [3], [2, 3]]

    def sliding_window_inference(
        self, image: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor]:
        """Inference using sliding window.

        Args:
            image: Image to predict.

        Returns:
            Predicted logits.
        """
        if self.trainer.datamodule is None:
            sw_batch_size = 2
        else:
            sw_batch_size = self.trainer.datamodule.hparams.batch_size
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=sw_batch_size,
            predictor=self.net,
            overlap=self.hparams.sliding_window_overlap,
            mode=self.hparams.sliding_window_importance_map,
        )

    @staticmethod
    def metric_mean(name: str, outputs: dict) -> Tensor:
        """Average metrics across batch dimension at epoch end.

        Args:
            name: Name of metrics to average.
            outputs: Outputs dictionary returned at step end.

        Returns:
            Averaged metrics tensor.
        """
        return torch.stack([out[name] for out in outputs]).mean(dim=0)

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
        print(f"Saving prediction for {fname}...\n")

        os.makedirs(save_dir, exist_ok=True)

        preds = preds.astype(np.float32)
        itk_image = sitk.GetImageFromArray(rearrange(preds, "w h d ->  d h w"))
        itk_image.SetSpacing(spacing)
        sitk.WriteImage(itk_image, os.path.join(save_dir, fname + ".nii.gz"))

    def update_eval_criterion_MA(self):
        """Update moving average validation loss."""
        if self.val_eval_criterion_MA is None:
            self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            self.val_eval_criterion_MA = (
                self.val_eval_criterion_alpha * self.val_eval_criterion_MA
                + (1 - self.val_eval_criterion_alpha) * self.all_val_eval_metrics[-1]
            )

    def maybe_update_best_val_eval_criterion_MA(self):
        """Update moving average validation metrics."""
        if self.best_val_eval_criterion_MA is None:
            self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
        if self.val_eval_criterion_MA < self.best_val_eval_criterion_MA:
            self.best_val_eval_criterion_MA = self.val_eval_criterion_MA


if __name__ == "__main__":
    from typing import List

    import hydra
    import omegaconf
    import pyrootutils
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from pytorch_lightning import (
        Callback,
        LightningDataModule,
        LightningModule,
        Trainer,
    )

    from ascent import utils

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    with initialize(version_base="1.2", config_path="../../configs/model"):
        cfg = compose(config_name="unwrapv2_2d.yaml")
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
    cfg._target_ = "ascent.datamodules.dealias_datamodule.DealiasDataModule"
    cfg.data_dir = str(root / "data")
    cfg.dataset_name = "UNWRAPV2"
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
