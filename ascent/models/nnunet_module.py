import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from einops.einops import rearrange
from lightning import LightningModule
from monai.data import MetaTensor
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
        name: str = "nnUNet",
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

        # parameter alpha for calculating moving average eval metrics
        # MA_metric = alpha * old + (1-alpha) * new
        self.val_eval_criterion_alpha = 0.9

        # current moving average dice
        self.val_eval_criterion_MA = None

        # best moving average dice
        self.best_val_eval_criterion_MA = None

        # list to store all the moving average dice during the training
        self.all_val_eval_metrics = []

        # list to store the metrics computed during evaluation steps
        self.online_eval_foreground_dc = []

        # we consider all the evaluation batches as a single element and only compute the global
        # foreground dice at the end of the evaluation epoch
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        # store validation/test steps output as we can no longer receive steps output in
        # `on_validation_epoch_end` and `on_test_epoch_end`
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        # to initialize some class variables that depend on the model
        self.threeD = len(self.net.patch_size) == 3
        self.patch_size = list(self.net.patch_size)
        self.num_classes = self.net.num_classes

        # create a dummy input to display model summary
        self.example_input_array = torch.rand(
            1, self.net.in_channels, *self.patch_size, device=self.device
        )

        # get the flipping axes in case of tta
        if self.hparams.tta:
            self.tta_flips = self.get_tta_flips()

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

        # Compute the stats that will be used to compute the final dice metric during the end of
        # epoch
        num_classes = pred.shape[1]
        pred_softmax = softmax_helper(pred)
        pred_seg = pred_softmax.argmax(1)
        label = label[:, 0]
        axes = tuple(range(1, len(label.shape)))
        tp_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)
        fp_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)
        fn_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)

        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor(
                (pred_seg == c).float() * (label == c).float(), axes=axes
            )
            fp_hard[:, c - 1] = sum_tensor(
                (pred_seg == c).float() * (label != c).float(), axes=axes
            )
            fn_hard[:, c - 1] = sum_tensor(
                (pred_seg != c).float() * (label == c).float(), axes=axes
            )

        tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

        self.online_eval_foreground_dc.append(
            list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))
        )
        self.online_eval_tp.append(list(tp_hard))
        self.online_eval_fp.append(list(fp_hard))
        self.online_eval_fn.append(list(fn_hard))

        self.validation_step_outputs.append({"val/loss": loss})

        return {"val/loss": loss}

    def on_validation_epoch_end(self):  # noqa: D102
        loss = self.metric_mean("val/loss", self.validation_step_outputs)
        self.validation_step_outputs.clear()  # free memory

        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [
            i if not np.isnan(i) else 0.0
            for i in [
                2 * i / (2 * i + j + k)
                for i, j, k in zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)
            ]
        ]

        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

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
            "val/dice_MA",
            self.val_eval_criterion_MA,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )
        for label, dice in zip(range(len(global_dc_per_class)), global_dc_per_class):
            self.log(
                f"val/dice/{label}",
                np.round(dice, 4),
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
        img, label, image_meta_dict = batch["image"], batch["label"], batch["image_meta_dict"]

        start_time = time.time()
        preds = self.tta_predict(img) if self.hparams.tta else self.predict(img)
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

        num_classes = preds.shape[1]
        pred_seg = preds.argmax(1)
        label = label[:, 0]
        axes = tuple(range(1, len(label.shape)))
        tp_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)
        fp_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)
        fn_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor(
                (pred_seg == c).float() * (label == c).float(), axes=axes
            )
            fp_hard[:, c - 1] = sum_tensor(
                (pred_seg == c).float() * (label != c).float(), axes=axes
            )
            fn_hard[:, c - 1] = sum_tensor(
                (pred_seg != c).float() * (label == c).float(), axes=axes
            )

        tp_hard = tp_hard.sum(0, keepdim=False)
        fp_hard = fp_hard.sum(0, keepdim=False)
        fn_hard = fn_hard.sum(0, keepdim=False)
        test_dice = (2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)

        test_dice = torch.mean(test_dice, 0)

        self.log(
            "test/dice",
            test_dice,
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

            box_start, box_end = properties_dict.get("crop_bbox")
            min_w, min_h, min_d = box_start
            max_w, max_h, max_d = box_end

            final_preds = np.zeros([preds.shape[0], *original_shape])
            final_preds[:, min_w:max_w, min_h:max_h, min_d:max_d] = preds

            if self.trainer.datamodule.hparams.test_splits:
                save_dir = os.path.join(self.trainer.default_root_dir, "testing_raw")
            else:
                save_dir = os.path.join(self.trainer.default_root_dir, "validation_raw")

            fname = properties_dict.get("case_identifier")
            spacing = properties_dict.get("original_spacing")

            if self.hparams.save_npz:
                self.save_npz_and_properties(final_preds, properties_dict, fname, save_dir)

            final_preds = final_preds.argmax(0)

            self.save_mask(final_preds, fname, spacing, save_dir)

        self.test_step_outputs.append({"test/dice": test_dice})

        return {"test/dice": test_dice}

    def on_test_epoch_end(self):  # noqa: D102
        mean_dice = self.metric_mean("test/dice", self.test_step_outputs)
        self.test_step_outputs.clear()  # free memory

        self.log(
            "test/mean_dice",
            mean_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )

    def on_predict_start(self) -> None:  # noqa: D102
        super().on_predict_start()
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

        box_start, box_end = properties_dict.get("crop_bbox")
        min_w, min_h, min_d = box_start
        max_w, max_h, max_d = box_end

        final_preds = np.zeros([preds.shape[0], *original_shape])
        final_preds[:, min_w:max_w, min_h:max_h, min_d:max_d] = preds

        save_dir = os.path.join(self.trainer.default_root_dir, "inference_raw")

        fname = properties_dict.get("case_identifier")
        spacing = properties_dict.get("original_spacing")

        if self.hparams.save_npz:
            self.save_npz_and_properties(final_preds, properties_dict, fname, save_dir)

        final_preds = final_preds.argmax(0)

        self.save_mask(final_preds, fname, spacing, save_dir)

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
                image = pad(image, (6, 6, 0, 0, 0, 0), mode="reflect")
                pred = self.predict_3D_3Dconv_tiled(image, apply_softmax)
                # Inverse the padding after prediction
                return pred[..., 6:-6]
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
        if properties_dict.get("resampling_flag"):
            properties_dict["anisotropy_flag"] = image_meta_dict["anisotropy_flag"].item()
        properties_dict["crop_bbox"] = image_meta_dict["crop_bbox"][0].tolist()
        properties_dict["case_identifier"] = image_meta_dict["case_identifier"][0]
        properties_dict["original_spacing"] = image_meta_dict["original_spacing"][0].tolist()
        properties_dict["spacing_after_resampling"] = image_meta_dict["spacing_after_resampling"][
            0
        ].tolist()

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
        if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
            self.best_val_eval_criterion_MA = self.val_eval_criterion_MA


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
