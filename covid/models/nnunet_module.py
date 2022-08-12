import os

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
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
        deep_supervision: bool = True,
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
            deep_supervision: Whether to use deep supervision heads. Defaults to true.
            tta: Whether to use the test time augmentation, i.e. flip. Defaults to true.
            sliding_window_overlap: Minimum overlap for sliding window inference. Defaults to 0.5.
            sliding_window_importance_map: Importance map used for sliding window inference. Defaults to 'gaussian'
            save_prediction: Whether to save the test predictions. Defaults to true.
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

        # we consider all the evaluation batches as a single element and only compute the global foreground dice at the end of evaluation epoch
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

        # Compute the stats that will be used to compute the final dice metric during the end of epoch
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
        img, label = batch["image"], batch["label"]
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

        tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()
        test_dice = (2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)

        test_dice = torch.tensor(np.mean(test_dice, 0))

        self.log(
            "test/dice",
            test_dice,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )
        return {"test/dice": test_dice, "preds": preds, "targets": label}

    def test_step_end(self, test_step_outputs):
        preds = test_step_outputs["preds"]
        if self.save_predictions:
            # meta = batch["meta"][0].cpu().detach().numpy()
            # original_shape = meta[2]
            # min_d, max_d = meta[0, 0], meta[1, 0]
            # min_h, max_h = meta[0, 1], meta[1, 1]
            # min_w, max_w = meta[0, 2], meta[1, 2]

            # final_pred = torch.zeros((1, pred.shape[1], *original_shape), device=img.device)
            # final_pred[:, :, min_d:max_d, min_h:max_h, min_w:max_w] = pred
            fname = preds.meta["filename_or_obj"][0]
            fname = os.path.basename(fname).rsplit("_", 1)[0]
            spacing = list(preds.meta["pixdim"][0].cpu().detach().numpy())
            spacing = [float(i) for i in spacing[1:] if i]
            final_preds = softmax_helper(preds).argmax(1)
            final_preds = final_preds.squeeze(0).cpu().detach().numpy()
            # batch["image_meta_dict"]["filename_or_obj"]
            # if not all(original_shape == final_pred.shape[1:]):
            #     class_ = final_pred.shape[0]
            #     resized_pred = np.zeros((class_, *original_shape))
            #     for i in range(class_):
            #         resized_pred[i] = resize(
            #             final_pred[i], original_shape, order=3, mode="edge", cval=0, clip=True, anti_aliasing=False
            #         )
            #     final_pred = resized_pred

            self.save_mask(final_preds, fname, spacing)

    def test_epoch_end(self, test_step_outputs):
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

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer)

        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["all_val_eval_metrics"] = self.all_val_eval_metrics

    def on_load_checkpoint(self, checkpoint):
        self.all_val_eval_metrics = checkpoint["all_val_eval_metrics"]

    def compute_loss(self, preds, label):
        """Compute the multi-scale loss if deep supervision is set to True."""
        if self.hparams.deep_supervision:
            loss = self.loss(preds[0], label)
            for i, pred in enumerate(preds[1:]):
                downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(pred, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return self.loss(preds, label)

    def predict(self, image):
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

    def tta_predict(self, img):
        preds = self.predict(img)
        for flip_idx in self.tta_flips:
            preds += torch.flip(self.predict(torch.flip(img, flip_idx)), flip_idx)
        preds /= len(self.tta_flips) + 1
        return preds

    def predict_2D_2Dconv_tiled(self, image):
        assert len(image.shape) == 4, "data must be b, c, w, h"
        return self.sliding_window_inference(image)

    def predict_3D_3Dconv_tiled(self, image):
        assert len(image.shape) == 5, "data must be b, c, w, h, d"
        return self.sliding_window_inference(image)

    def predict_3D_2Dconv_tiled(self, image):
        assert len(image.shape) == 5, "data must be b, c, w, h, d"
        preds_shape = (image.shape[0], self.num_classes, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[2]):
            preds[:, :, depth] = self.predict_2D_2Dconv_tiled(image[:, :, depth])
        return preds

    def get_tta_flips(self):
        if self.threeD:
            return [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        else:
            return [[2], [3], [2, 3]]

    def sliding_window_inference(self, image):
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
        save_dir = os.path.join(self.trainer.default_root_dir, "testing_raw")
        os.makedirs(save_dir, exist_ok=True)
        if len(preds.shape) == len(spacing) - 1:
            preds = preds[..., None].astype(np.uint8)
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "camus.yaml")
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
    cfg.fold = 0
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
