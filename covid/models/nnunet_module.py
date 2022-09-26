import os
from typing import Union

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from monai.data import MetaTensor
from pytorch_lightning import LightningModule
from skimage.transform import resize

from covid.datamodules.components.inferers import sliding_window_inference
from covid.models.components.unet_related.utils import softmax_helper, sum_tensor


class nnUNetLitModule(LightningModule):
    """nnUNet training, evaluation and test strategy converted to PyTorch Lightning.

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
    ):
        """Saves the system's configuration in `hparams`. Initialize variables for training and
        validation loop.

        Args:
            net: Network architecture. Defaults to U-Net.
            optimizer: Optimizer. Defaults to SGD optimizer.
            loss: Loss function. Defaults to Cross Entropy - Dice
            scheduler: Scheduler for training. Defaults to Polynomial Decay Scheduler.
            tta: Whether to use the test time augmentation, i.e. flip.
            sliding_window_overlap: Minimum overlap for sliding window inference.
            sliding_window_importance_map: Importance map used for sliding window inference.
            save_prediction: Whether to save the test predictions.
        """
        super().__init__()
        # ignore net and loss as they are nn.module and will be saved automatically
        self.save_hyperparameters(logger=False, ignore=["net", "loss"])

        self.net = net
        self.threeD = len(self.net.patch_size) == 3
        self.patch_size = list(self.net.patch_size)

        self.num_classes = self.net.num_classes

        # declare a dummy input for display model summary
        self.example_input_array = torch.rand(
            1, self.net.in_channels, *self.patch_size, device=self.device
        )

        # loss function (CE - Dice), min = -1
        self.loss = loss

        # parameter alpha for calculating moving average dice -> alpha * old + (1-alpha) * new
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
        # foreground dice at the end of evaluation epoch
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        self.tta = tta
        if self.tta:
            self.tta_flips = self.get_tta_flips()
        self.test_idx = 0
        self.test_imgs = []

        self.save_predictions = save_predictions

    def forward(self, img):  # noqa: D102
        return self.net(img)

    def training_step(self, batch, batch_idx: int):  # noqa: D102
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

    def validation_step(self, batch, batch_idx):  # noqa: D102
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

        return {"val/loss": loss}

    def validation_epoch_end(self, validation_step_outputs):  # noqa: D102
        loss = self.metric_mean("val/loss", validation_step_outputs)
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

    def test_step(self, batch, batch_idx):  # noqa: D102
        img, label, image_meta_dict = batch["image"], batch["label"], batch["image_meta_dict"]
        preds = self.tta_predict(img) if self.tta else self.predict(img)

        num_classes = preds.shape[1]
        pred_softmax = softmax_helper(preds)
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
        return {
            "test/dice": test_dice,
            "preds": preds,
            "image_meta_dict": image_meta_dict,
        }

    def test_step_end(self, test_step_outputs):  # noqa: D102
        preds = test_step_outputs["preds"]
        image_meta_dict = test_step_outputs["image_meta_dict"]
        if self.save_predictions:
            preds = softmax_helper(preds).argmax(1)
            preds = preds.squeeze(0).cpu().detach().numpy()
            original_shape = image_meta_dict["original_shape"][0].tolist()
            if len(preds.shape) == len(original_shape) - 1:
                preds = preds[..., None]
            if image_meta_dict["resampling_flag"].item():
                shape_after_cropping = image_meta_dict["shape_after_cropping"][0].tolist()
                preds = self.recovery_prediction(
                    preds, shape_after_cropping, image_meta_dict["anisotropy_flag"].item()
                )

            box_start, box_end = image_meta_dict["crop_bbox"][0].tolist()
            min_w, min_h, min_d = box_start
            max_w, max_h, max_d = box_end

            final_preds = np.zeros(original_shape)
            final_preds[min_w:max_w, min_h:max_h, min_d:max_d] = preds
            fname = image_meta_dict["case_identifier"][0]
            spacing = image_meta_dict["original_spacing"][0].tolist()

            self.save_mask(final_preds, fname, spacing)

    def test_epoch_end(self, test_step_outputs):  # noqa: D102
        mean_dice = self.metric_mean("test/dice", test_step_outputs)
        self.log(
            "test/mean_dice",
            mean_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )

    def configure_optimizers(self):  # noqa: D102
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer)

        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):  # noqa: D102
        checkpoint["all_val_eval_metrics"] = self.all_val_eval_metrics

    def on_load_checkpoint(self, checkpoint):  # noqa: D102
        self.all_val_eval_metrics = checkpoint["all_val_eval_metrics"]

    def compute_loss(
        self, preds: Union[torch.Tensor, MetaTensor], label: Union[torch.Tensor, MetaTensor]
    ):
        """Compute the multi-scale loss if deep supervision is set to True.

        Args:
            preds: Logits predicted
            label: Ground truth label
        """

        if self.net.deep_supervision:
            loss = self.loss(preds[0], label)
            for i, pred in enumerate(preds[1:]):
                downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(pred, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return self.loss(preds, label)

    def predict(self, image) -> Union[torch.Tensor, MetaTensor]:
        """Predict 2D/3D images with sliding window inference.

        Args:
            image: Image to predict.

        Returns:
            Logits of prediction.
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

    def tta_predict(
        self, image: Union[torch.Tensor, MetaTensor]
    ) -> Union[torch.Tensor, MetaTensor]:
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
        self, image: Union[torch.Tensor, MetaTensor]
    ) -> Union[torch.Tensor, MetaTensor]:
        """Predict 2D image with 2D model.

        Args:
            image: Image to predict.

        Returns:
            Logits of prediction.
        """

        assert len(image.shape) == 4, "data must be b, c, w, h"
        return self.sliding_window_inference(image)

    def predict_3D_3Dconv_tiled(
        self, image: Union[torch.Tensor, MetaTensor]
    ) -> Union[torch.Tensor, MetaTensor]:
        """Predict 3D image with 3D model.

        Args:
            image: Image to predict.

        Returns:
            Logits of prediction.
        """

        assert len(image.shape) == 5, "data must be b, c, w, h, d"
        return self.sliding_window_inference(image)

    def predict_3D_2Dconv_tiled(
        self, image: Union[torch.Tensor, MetaTensor]
    ) -> Union[torch.Tensor, MetaTensor]:
        """Predict 3D image with 2D model.

        Args:
            image: Image to predict.

        Returns:
            Logits of prediction.
        """

        assert len(image.shape) == 5, "data must be b, c, w, h, d"
        preds_shape = (image.shape[0], self.num_classes, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[-1]):
            preds[..., depth] = self.predict_2D_2Dconv_tiled(image[..., depth])
        return preds

    @staticmethod
    def recovery_prediction(
        prediction: Union[torch.Tensor, MetaTensor],
        new_shape: Union[tuple, list],
        anisotropy_flag: bool,
    ) -> np.array:
        """Recover prediction to its original shape in case of resampling.

        Args:
            prediciton: Logits predicted.
            new_shape: Shape for resampling.
            anisotropy_flag: Whether to use anisotropic resampling.

        Returns:
            Resampled prediction to the original spacing.
        """
        shape = np.array(prediction.shape)
        if np.any(shape != np.array(new_shape)):
            reshaped = np.zeros(new_shape, dtype=np.uint8)
            n_class = np.max(prediction)
            if anisotropy_flag:
                shape_2d = new_shape[:-1]
                depth = prediction.shape[-1]
                reshaped_2d = np.zeros((*shape_2d, depth), dtype=np.uint8)

                for class_ in range(1, int(n_class) + 1):
                    for depth_ in range(depth):
                        mask = prediction[:, :, depth_] == class_
                        resized_2d = resize(
                            mask.astype(float),
                            shape_2d,
                            order=1,
                            mode="edge",
                            cval=0,
                            clip=True,
                            anti_aliasing=False,
                        )
                        reshaped_2d[:, :, depth_][resized_2d >= 0.5] = class_
                for class_ in range(1, int(n_class) + 1):
                    mask = reshaped_2d == class_
                    resized = resize(
                        mask.astype(float),
                        new_shape,
                        order=0,
                        mode="constant",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    reshaped[resized >= 0.5] = class_
            else:
                for class_ in range(1, int(n_class) + 1):
                    mask = prediction == class_
                    resized = resize(
                        mask.astype(float),
                        new_shape,
                        order=1,
                        mode="edge",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    reshaped[resized >= 0.5] = class_

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

    def sliding_window_inference(self, image):
        """"""
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=self.trainer.datamodule.hparams.batch_size,
            predictor=self.net,
            overlap=self.hparams.sliding_window_overlap,
            mode=self.hparams.sliding_window_importance_map,
        )

    @staticmethod
    def metric_mean(name, outputs):
        return torch.stack([out[name] for out in outputs]).mean(dim=0)

    def save_mask(self, preds, fname, spacing):
        print(f"\nSaving prediction for {fname}...\n")
        if self.trainer.datamodule.hparams.test_splits:
            save_dir = os.path.join(self.trainer.default_root_dir, "testing_raw")
        else:
            save_dir = os.path.join(self.trainer.default_root_dir, "validation_raw")
        os.makedirs(save_dir, exist_ok=True)

        preds = preds.astype(np.uint8)
        itk_image = sitk.GetImageFromArray(np.transpose(preds, list(range(0, len(spacing)))[::-1]))
        itk_image.SetSpacing(spacing)
        sitk.WriteImage(itk_image, os.path.join(save_dir, fname + ".nii.gz"))

    def update_eval_criterion_MA(self):
        if self.val_eval_criterion_MA is None:
            self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            self.val_eval_criterion_MA = (
                self.val_eval_criterion_alpha * self.val_eval_criterion_MA
                + (1 - self.val_eval_criterion_alpha) * self.all_val_eval_metrics[-1]
            )

    def maybe_update_best_val_eval_criterion_MA(self):
        if self.best_val_eval_criterion_MA is None:
            self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
        if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
            self.best_val_eval_criterion_MA = self.val_eval_criterion_MA


if __name__ == "__main__":
    from typing import List

    import hydra
    import omegaconf
    import pyrootutils
    from pytorch_lightning import (
        Callback,
        LightningDataModule,
        LightningModule,
        Trainer,
    )

    from covid import utils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "camus_2d.yaml")
    cfg.scheduler.max_decay_steps = 1000
    cfg.net.patch_size = [128, 128]
    cfg.net.kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    cfg.net.strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]]
    nnunet: LightningModule = hydra.utils.instantiate(cfg)

    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "camus.yaml")
    cfg.data_dir = str(root / "data")
    cfg.patch_size = [128, 128]
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
    ckpt_path = "C:/Users/ling/Desktop/Thesis/REPO/CoVID/logs/lightning_logs/version_0/checkpoints/epoch=1-step=500.ckpt"
    trainer.test(model=nnunet, datamodule=camus_datamodule, ckpt_path=ckpt_path)
