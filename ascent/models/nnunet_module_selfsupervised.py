import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, Optional, Union

import matplotlib.pyplot as plt
import monai
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from einops.einops import rearrange
from lightning import LightningModule
from monai.data import MetaTensor
from monai.transforms import Rotate
from torch import Tensor
from torch.nn.functional import pad

from ascent.preprocessing.preprocessing import check_anisotropy, get_lowres_axis, resample_image
from ascent.utils.file_and_folder_operations import save_pickle
from ascent.utils.inferers import SlidingWindowInferer
from ascent.utils.softmax import softmax_helper
from ascent.utils.tensor_utils import sum_tensor


class nnUNetLitModule(LightningModule):
    """`nnUNet` training, evaluation and test strategy converted to PyTorch Lightning.

    nnUNetLitModule includes all nnUNet key features, including the test time augmentation, sliding
    window inference etc. Currently only 2D and 3D_fullres nnUNet are supported.
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
            name: str = "nnUNet"
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
            name: Name of the network.
        """
        super().__init__()
        # ignore net and loss as they are nn.module and will be saved automatically
        self.save_hyperparameters(logger=False, ignore=["net", "loss"])

        self.net = net

        # loss function (CE - Dice), min = -1
        self.loss = loss

        # list to store all the moving average dice during the training
        self.all_val_eval_metrics = []

        # list to store the metrics computed during evaluation steps
        self.online_eval_foreground_dc = []

        # store validation/test steps output as we can no longer receive steps output in
        # `on_validation_epoch_end` and `on_test_epoch_end`
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.validation_step_ssim = []

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        # to initialize some class variables that depend on the model
        self.threeD = len(self.net.patch_size) == 3
        self.patch_size = list(self.net.patch_size)
        self.num_classes = self.net.num_classes

        # create a dummy input to display model summary
        self.example_input_array = torch.rand(
            1, self.net.in_channels, *self.patch_size, device=self.device
        )
        self.ssim = monai.metrics.regression.SSIMMetric(spatial_dims=len(self.net.patch_size))

        # get the flipping axes in case of tta
        if self.hparams.tta:
            self.tta_flips = self.get_tta_flips()

    def generate_random_mask(self, image_shape, mask_percentage=0.25):
        num_batch, channels, *dim = image_shape
        mask = np.ones(image_shape, dtype=np.float32)

        num_pixels = 1
        for d in dim:
            num_pixels = num_pixels * d

        num_pixels_to_mask = int(num_pixels * mask_percentage)

        for b in range(num_batch):
            mask_indices = np.random.choice(num_pixels, num_pixels_to_mask, replace=False)
            for c in range(channels):
                mask[b, c][np.unravel_index(mask_indices, dim)] = 0

        return torch.tensor(mask, dtype=torch.float32).to(self.device)

    def forward(self, img: Union[Tensor, MetaTensor]) -> Union[Tensor, MetaTensor]:  # noqa: D102
        return self.net(img)

    def training_step(
            self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        img = batch["image"]

        mask = self.generate_random_mask(img.shape, mask_percentage=0.25)
        masked_img = img * mask

        # Need to handle carefully the multi-scale outputs from deep supervision heads
        pred = self.forward(masked_img)
        loss = self.compute_loss(pred, img)

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
        img = batch["image"]

        mask = self.generate_random_mask(img.shape, mask_percentage=0.25)
        masked_img = img * mask

        # Only the highest resolution output is returned during the validation
        pred = self.forward(masked_img)
        loss = self.loss(pred, img)
        if batch_idx==0:
            plt.imshow(pred[0, 0, :, :].detach().cpu())
            plt.show()

        test_ssim = self.ssim(img, pred).mean()

        self.validation_step_outputs.append({"val/loss": loss})
        self.validation_step_ssim.append({"val/ssim": test_ssim})

        return {"val/loss": loss}

    def on_validation_epoch_end(self):  # noqa: D102
        loss = self.metric_mean("val/loss", self.validation_step_outputs)
        mean_ssim = self.metric_mean("val/ssim", self.validation_step_ssim)
        self.validation_step_outputs.clear()  # free memory

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
            "val/acc",
            mean_ssim,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )

    def on_test_start(self) -> None:  # noqa: D102
        super().on_test_start()
        if self.trainer.datamodule is None:
            sw_batch_size = 2
        else:
            sw_batch_size = self.trainer.datamodule.hparams.batch_size

        self.inferer = SlidingWindowInferer(
            roi_size=self.patch_size,
            sw_batch_size=sw_batch_size,
            overlap=self.hparams.sliding_window_overlap,
            mode=self.hparams.sliding_window_importance_map,
            cache_roi_weight_map=True,
        )

    def test_step(
            self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        img, key = batch["image"], batch["key"]

        mask = self.generate_random_mask(img.shape, mask_percentage=0.25)
        masked_img = img * mask

        start_time = time.time()
        preds = self.predict(masked_img, apply_softmax=False)
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

        self.ssim = monai.metrics.regression.SSIMMetric(spatial_dims=len(preds.shape)-2, win_size=5)
        test_ssim = self.ssim(img, preds)

        if self.trainer.datamodule.hparams.test_splits:
            save_dir = os.path.join(self.trainer.default_root_dir, "testing_raw")
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = os.path.join(self.trainer.default_root_dir, "validation_raw")
            os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(os.path.join(save_dir, f"{key[0]}.npz"), image=preds.detach().cpu())

        self.test_step_outputs.append({"test/ssim": test_ssim})

        return {"test/ssim": test_ssim}

    def on_test_epoch_end(self):  # noqa: D102
        mean_dice = self.metric_mean("test/ssim", self.test_step_outputs)
        self.test_step_outputs.clear()  # free memory

        self.log(
            "test/mean_acc",
            mean_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )

    def configure_optimizers(self) -> dict[Literal["optimizer", "lr_scheduler"], Any]:
        """Configures optimizers/LR schedulers.

        Returns:
            A dict with an `optimizer` key, and an optional `lr_scheduler` if a scheduler is used.
        """
        configured_optimizer = {"optimizer": self.hparams.optimizer(params=self.parameters())}
        if self.hparams.scheduler:
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
        if self.net.deep_supervision:
            loss = self.loss(preds[0], label)
            for i, pred in enumerate(preds[1:]):
                downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(pred, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return self.loss(preds, label)

    def predict(
            self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 2D/3D images with sliding window inference.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
            ValueError: If 3D patch is requested to predict 2D images.
        """
        if len(image.shape) == 5:
            if len(self.patch_size) == 3:
                # Pad the last dimension to avoid 3D segmentation border artifacts
                extra_pad = 0
                while image.shape[-1] <= 6:
                    image = pad(image, (1, 1, 0, 0, 0, 0), mode="reflect")
                    extra_pad += 1
                image = pad(image, (6, 6, 0, 0, 0, 0), mode="reflect")
                pred = self.predict_3D_3Dconv_tiled(image, apply_softmax)
                # Inverse the padding after prediction
                return pred[..., (6 + extra_pad): (-6 - extra_pad)]
            elif len(self.patch_size) == 2:
                return self.predict_3D_2Dconv_tiled(image, apply_softmax)
            else:
                raise NotImplementedError
        if len(image.shape) == 4:
            if len(self.patch_size) == 2:
                return self.predict_2D_2Dconv_tiled(image, apply_softmax)
            elif len(self.patch_size) == 3:
                raise ValueError("You can't predict a 2D image with 3D model. You dummy.")
            else:
                raise NotImplementedError

    def tta_predict(
            self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict with test time augmentation.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over number of flips.
        """
        preds = self.predict(image, apply_softmax)
        for flip_idx in self.tta_flips:
            preds += torch.flip(self.predict(torch.flip(image, flip_idx), apply_softmax), flip_idx)
        preds /= len(self.tta_flips) + 1
        return preds

    def predict_2D_2Dconv_tiled(
            self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 2D image with 2D model.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            ValueError: If image is not 2D.
        """
        if not len(image.shape) == 4:
            raise ValueError("image must be (b, c, w, h)")

        if apply_softmax:
            return softmax_helper(self.sliding_window_inference(image))
        else:
            return self.sliding_window_inference(image)

    def predict_3D_3Dconv_tiled(
            self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 3D image with 3D model.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            ValueError: If image is not 3D.
        """
        if not len(image.shape) == 5:
            raise ValueError("image must be (b, c, w, h, d)")

        if apply_softmax:
            return softmax_helper(self.sliding_window_inference(image))
        else:
            return self.sliding_window_inference(image)

    def predict_3D_2Dconv_tiled(
            self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 3D image with 2D model.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            ValueError: If image is not 3D.
        """
        if not len(image.shape) == 5:
            raise ValueError("image must be (b, c, w, h, d)")
        preds_shape = (image.shape[0], self.num_classes, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[-1]):
            preds[..., depth] = self.predict_2D_2Dconv_tiled(image[..., depth], apply_softmax)
        return preds

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
        return self.inferer(
            inputs=image,
            network=self.net,
        )

    @staticmethod
    def metric_mean(name: str, outputs: list[dict[str, Tensor]]) -> Tensor:
        """Average metrics across batch dimension at epoch end.

        Args:
            name: Name of metrics to average.
            outputs: List containing outputs dictionary returned at step end.

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
        properties_dict["rotate_flag"] = image_meta_dict["rotate_flag"].item()
        if properties_dict.get("rotate_flag"):
            properties_dict["angle_rotation"] = image_meta_dict["angle_rotation"]
        if properties_dict.get("resampling_flag"):
            properties_dict["spacing_after_resampling"] = image_meta_dict[
                "spacing_after_resampling"
            ][0].tolist()
            properties_dict["anisotropy_flag"] = image_meta_dict["anisotropy_flag"].item()
        if len(image_meta_dict["crop_bbox"]):
            properties_dict["crop_bbox"] = image_meta_dict["crop_bbox"][0].tolist()
        else:
            properties_dict["crop_bbox"] = []
        properties_dict["case_identifier"] = image_meta_dict["case_identifier"][0]
        properties_dict["original_spacing"] = image_meta_dict["original_spacing"][0].tolist()

        return properties_dict

    def save_mask(
            self, preds: np.ndarray, fname: str, spacing: np.ndarray, save_dir: Union[str, Path]
    ) -> None:
        """Save segmentation mask to the given save directory.

        Args:
            preds: Predicted segmentation mask.
            fname: Filename to save.
            spacing: Spacing to save the segmentation mask.
            save_dir: Directory to save the segmentation mask.
        """
        print(f"Saving segmentation for {fname}...")

        os.makedirs(save_dir, exist_ok=True)

        preds = preds.astype(np.uint8)
        itk_image = sitk.GetImageFromArray(rearrange(preds, "w h d ->  d h w"))
        itk_image.SetSpacing(spacing)
        sitk.WriteImage(itk_image, os.path.join(save_dir, fname + ".nii.gz"))

    def save_npz_and_properties(
            self, preds: np.ndarray, properties_dict: dict, fname: str, save_dir: Union[str, Path]
    ) -> None:
        """Save softmax probabilities to the given save directory.

        Args:
            preds: Predicted softmax.
            properties_dict: Dictionary containing properties of predicted data (eg. spacing).
            fname: Filename to save.
            spacing: Spacing to save the segmentation mask.
            save_dir: Directory to save the segmentation mask.
        """
        print(f"Saving softmax for {fname}...")

        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(save_dir, fname + ".npz"), softmax=preds.astype(np.float16)
        )
        save_pickle(properties_dict, os.path.join(save_dir, fname + ".pkl"))


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
        cfg = compose(config_name="camus_2d.yaml")
        print(OmegaConf.to_yaml(cfg))

    # cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "nnunet.yaml")
    cfg.scheduler.max_decay_steps = 1000
    cfg.net.in_channels = 1
    cfg.net.num_classes = 3
    cfg.net.patch_size = [128, 128]
    cfg.net.kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    cfg.net.strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]]
    nnunet: LightningModule = hydra.utils.instantiate(cfg)

    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "nnunet.yaml")
    cfg.data_dir = str(root / "data")
    cfg.dataset_name = "CAMUS"
    cfg.patch_size = [128, 128]
    cfg.do_dummy_2D_data_aug = False
    cfg.in_channels = 1
    # cfg.patch_size = [128, 128, 12]
    cfg.batch_size = 2
    cfg.fold = 4
    camus_datamodule: LightningDataModule = hydra.utils.instantiate(cfg)

    cfg = omegaconf.OmegaConf.load(root / "configs" / "callbacks" / "nnunet.yaml")
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

    # trainer.fit(model=nnunet, datamodule=camus_datamodule)
    print("Starting testing!")
    ckpt_path = "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/logs/lightning_logs/version_0/checkpoints/epoch=1-step=500.ckpt"
    trainer.test(model=nnunet, datamodule=camus_datamodule, ckpt_path=ckpt_path)
