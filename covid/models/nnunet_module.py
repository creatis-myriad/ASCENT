import os

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from skimage.transform import resize

from covid.datamodules.components.inferers import sliding_window_inference
from covid.models.components.unet_related.metrics import Dice
from covid.models.components.unet_related.utils import softmax_helper, sum_tensor


class nnUNetLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        deep_supervision=True,
        save_predictions=True,
        tta=True,
        sliding_window_overlap=True,
        sliding_windows_importance_map="gaussian",
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net", "loss"])

        self.net = net
        self.threeD = len(self.net.patch_size) == 3
        self.patch_size = list(self.net.patch_size)

        self.num_classes = self.net.num_classes

        # parameters related to the data
        # self.example_input_array = torch.rand(
        #     self.net.in_channels, *self.patch_size, device=self.device
        # )

        # loss function (CE - Dice), min = -1
        self.loss = loss

        # metric
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        self.val_eval_criterion_MA = None
        self.best_val_eval_criterion_MA = None
        self.all_val_eval_metrics = []
        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        self.test_dice = Dice(self.num_classes)

        self.tta = tta
        if self.tta:
            self.tta_flips = self.get_tta_flips()
        self.test_idx = 0
        self.test_imgs = []

        # whether to save the predictions during test and inference
        self.save_predictions = save_predictions

    def forward(self, img):
        return self.net(img)

    def training_step(self, batch, batch_idx):
        img, label = batch["image"], batch["label"]
        # Need to handle carefully the multi-scale outputs from deep supervision heads
        pred = self.forward(img)
        loss = self.compute_loss(pred, label)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
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

    def validation_epoch_end(self, validation_step_outputs):
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

        self.log("val/loss", loss)
        self.log("val/dice_MA", self.val_eval_criterion_MA)
        for label, dice in zip(range(1, len(global_dc_per_class)), global_dc_per_class):
            self.log(f"val/dice/{label}", np.round(dice, 4))

    def test_step(self, batch, batch_idx):
        img, label = batch["image"], batch["label"]
        preds = self.tta_predict(img) if self.args.tta else self.predict(img)
        test_dice = self.test_dice(preds, label)
        self.log("test/dice", test_dice, on_step=False, on_epoch=True)
        return {"preds": preds, "targets": label}

    # def test_step_end(self, test_step_outputs):
    #     preds = test_step_outputs["preds"]
    #     if self.save_predictions:
    #         # meta = batch["meta"][0].cpu().detach().numpy()
    #         # original_shape = meta[2]
    #         # min_d, max_d = meta[0, 0], meta[1, 0]
    #         # min_h, max_h = meta[0, 1], meta[1, 1]
    #         # min_w, max_w = meta[0, 2], meta[1, 2]

    #         # final_pred = torch.zeros((1, pred.shape[1], *original_shape), device=img.device)
    #         # final_pred[:, :, min_d:max_d, min_h:max_h, min_w:max_w] = pred
    #         final_pred = nn.functional.softmax(preds, dim=1)
    #         final_pred = final_pred.squeeze(0).cpu().detach().numpy()

    #         # if not all(original_shape == final_pred.shape[1:]):
    #         #     class_ = final_pred.shape[0]
    #         #     resized_pred = np.zeros((class_, *original_shape))
    #         #     for i in range(class_):
    #         #         resized_pred[i] = resize(
    #         #             final_pred[i], original_shape, order=3, mode="edge", cval=0, clip=True, anti_aliasing=False
    #         #         )
    #         #     final_pred = resized_pred

    #         self.save_mask(final_pred)

    def compute_loss(self, preds, label):
        if self.hparams.deep_supervision:
            loss = self.loss(preds[0], label)
            for i, pred in enumerate(preds[1:]):
                downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(pred, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return self.loss(preds, label)

    def predict(self, image):
        if len(image.size()) == 5:
            if len(self.patch_size) == 3:
                return self.predict_3D_3Dconv_tiled(image)
            elif len(self.patch_size) == 2:
                return self.predict_3D_2Dconv_tiled(image)
            else:
                raise NotImplementedError
        if len(image.size()) == 4:
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
        assert len(image.size()) == 4, "data must be b, c, w, h"
        return self.sliding_window_inference(image)

    def predict_3D_3Dconv_tiled(self, image):
        assert len(image.size()) == 5, "data must be b, c, w, h, d"
        return self.sliding_window_inference(image)

    def predict_3D_2Dconv_tiled(self, image):
        assert len(image.size()) == 5, "data must be b, c, w, h, d"
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

    def inference2d(self, image):
        batch_modulo = image.shape[2] % self.args.val_batch_size
        if batch_modulo != 0:
            batch_pad = self.args.val_batch_size - batch_modulo
            image = nn.ConstantPad3d((0, 0, 0, 0, batch_pad, 0), 0)(image)
        image = torch.transpose(image.squeeze(0), 0, 1)
        preds_shape = (image.shape[0], self.num_classes + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for start in range(
            0, image.shape[0] - self.args.val_batch_size + 1, self.args.val_batch_size
        ):
            end = start + self.args.val_batch_size
            pred = self.net(image[start:end])
            preds[start:end] = pred.data
        if batch_modulo != 0:
            preds = preds[batch_pad:]
        return torch.transpose(preds, 0, 1).unsqueeze(0)

    def inference2d_test(self, image):
        preds_shape = (image.shape[0], self.num_classes + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[2]):
            preds[:, :, depth] = self.sliding_window_inference(image[:, :, depth])
        return preds

    def sliding_window_inference(self, image):
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=self.trainer.datamodule.batch_size,
            predictor=self.net,
            overlap=self.hparams.sliding_window_overlap,
            mode=self.hparams.sliding_windows_importance_map,
        )

    @staticmethod
    def metric_mean(name, outputs):
        return torch.stack([out[name] for out in outputs]).mean(dim=0)

    def test_epoch_end(self, outputs):
        if self.args.exec_mode == "evaluate":
            self.eval_dice = self.dice.compute()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer)

        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["all_val_eval_metrics"] = self.all_val_eval_metrics

    def on_load_checkpoint(self, checkpoint):
        self.all_val_eval_metrics = checkpoint["all_val_eval_metrics"]

    # def save_mask(self, pred):
    #     if self.test_idx == 0:
    #         data_path = get_path(self.args)
    #         self.test_imgs, _ = get_test_fnames(self.args, data_path)
    #     fname = os.path.basename(self.test_imgs[self.test_idx]).replace("_x", "")
    #     np.save(os.path.join(self.save_dir, fname), pred, allow_pickle=False)
    #     self.test_idx += 1

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
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "camus.yaml")
    cfg.scheduler.max_decay_steps = 1000
    nnunet = hydra.utils.instantiate(cfg)
