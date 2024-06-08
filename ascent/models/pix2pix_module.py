from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.transforms import AsDiscrete
from torch import Tensor

from ascent.models.nnunet_module import nnUNetLitModule


class Pix2PixGANLitModule(nnUNetLitModule):
    """`Pix2PixGAN` lightning module for segmentation."""

    def __init__(
        self,
        loss_d: torch.nn.Module,
        optimizer_d: torch.optim.Optimizer,
        scheduler_d: torch.optim.lr_scheduler._LRScheduler,
        **kwargs
    ):
        """Initialize class instance.

        Args:
            loss_d: Discriminator loss function.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False, ignore=["net", "loss", "loss_d"])
        self.loss_d = loss_d
        self.automatic_optimization = False
        self.one_hot = AsDiscrete(to_onehot=self.net.generator.decoder.num_classes, dim=1)

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        # to initialize some class variables that depend on the model
        self.threeD = len(self.net.generator.patch_size) == 3
        self.patch_size = list(self.net.generator.patch_size)
        self.num_classes = self.net.generator.num_classes

        # create a dummy input to display model summary
        self.example_input_array = torch.rand(
            1, self.net.generator.in_channels, *self.patch_size, device=self.device
        )

        # get the flipping axes in case of tta
        if self.hparams.tta:
            self.tta_flips = self.get_tta_flips()

        # Create labels for the discriminator
        self.real_label = torch.ones(
            (self.trainer.datamodule.hparams.batch_size, 1, *self.net.discriminator_output_shape),
            device=self.device,
        )
        self.fake_label = torch.zeros(
            (self.trainer.datamodule.hparams.batch_size, 1, *self.net.discriminator_output_shape),
            device=self.device,
        )

    def forward(self, img: Union[Tensor, MetaTensor]) -> Union[Tensor, MetaTensor]:  # noqa: D102
        return self.net.generator(img)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int):  # noqa: D102
        g_opt, d_opt = self.optimizers()

        img, label = batch["image"], batch["label"]

        pred = self.forward(img)
        img_pred = torch.cat([img, pred[0]], dim=1)

        ##########################
        # Optimize Discriminator #
        ##########################
        # enable backpropagation for discriminator
        self.set_requires_grad(self.net.discriminator, True)
        # detach to avoid backprop through generator
        pred_fake = self.net.discriminator(img_pred.detach())
        loss_d_fake = self.loss_d(pred_fake, self.fake_label)

        pred_real = self.net.discriminator(
            torch.cat(
                [img, self.one_hot(label.long()).float()],
                dim=1,
            )
        )
        loss_d_real = self.loss_d(pred_real, self.real_label)

        loss_d = (loss_d_fake + loss_d_real) / 2

        d_opt.zero_grad()
        self.manual_backward(loss_d)
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        # disable backpropagation for discriminator
        self.set_requires_grad(self.net.discriminator, False)
        # no detach here
        pred_fake = self.net.discriminator(img_pred)

        # generator should be able to fool the discriminator
        loss_d_fake_g = self.loss_d(pred_fake, self.real_label)

        # segmentation loss
        loss = self.compute_loss(pred, label)

        loss_g = loss + loss_d_fake_g
        g_opt.zero_grad()
        self.manual_backward(loss_g)
        # self.clip_gradients(g_opt, 12)
        g_opt.step()

        self.log_dict(
            {
                "train/loss/g_dc_ce": loss,
                "train/loss/g": loss_d_fake_g,
                "train/loss/d_real": loss_d_real,
                "train/loss/d_fake": loss_d_fake,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
        )

    def on_train_epoch_end(self):  # noqa: D102
        # step the lr schedulers
        sch_g, sch_d = self.lr_schedulers()

        sch_d.step()
        sch_g.step()

    def configure_optimizers(self) -> tuple[Union[dict[str, Any], dict[str, Any]], ...]:
        """Configures optimizers/LR schedulers.

        Returns:
            A dict with an `optimizer` key, and an optional `lr_scheduler` if a scheduler is used
            or a list of dicts if multiple optimizers/schedulers are used.
        """
        configured_optimizers = []
        configured_optimizer_g = {
            "optimizer": self.hparams.optimizer(params=self.net.generator.parameters())
        }
        if self.hparams.scheduler:
            configured_optimizer_g["lr_scheduler"] = self.hparams.scheduler(
                optimizer=configured_optimizer_g["optimizer"]
            )
        configured_optimizers.append(configured_optimizer_g)

        configured_optimizer_d = {
            "optimizer": self.hparams.optimizer_d(params=self.net.discriminator.parameters())
        }
        if self.hparams.scheduler_d:
            configured_optimizer_d["lr_scheduler"] = self.hparams.scheduler_d(
                optimizer=configured_optimizer_d["optimizer"]
            )
        configured_optimizers.append(configured_optimizer_d)
        return tuple(configured_optimizers)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False) -> None:
        """Set `requies_grad=Fasle` for the networks to avoid unnecessary computations.

        Args:
            nets: List of networks
            requires_grad: Whether to set requires_grad to True or False
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
