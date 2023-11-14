from typing import Optional, Union

import torch
from monai.data import MetaTensor
from torch import Tensor, nn

from ascent.models.nnunet_reg_module import nnUNetRegLitModule


class nnUNetDealiasLitModule(nnUNetRegLitModule):
    """`nnUNet` lightning module for deep unfolding to perform dealiasing.

    Similar to nnUNetRegLitModule except for the definition of class's variables: threeD,
    patch_size, num_classes as the U-Net is no longer equals self.net but self.net.denoiser.
    """

    def __init__(self, **kwargs):
        """Initialize class instance.

        Args:
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(**kwargs)

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        # to initialize some class variables that depend on the model
        self.threeD = len(self.net.denoiser.patch_size) == 3
        self.patch_size = list(self.net.denoiser.patch_size)
        self.num_classes = self.net.denoiser.num_classes

        # create a dummy input to display model summary
        self.example_input_array = torch.rand(1, 2, *self.patch_size, device=self.device)

        # get the flipping axes in case of tta
        if self.hparams.tta:
            self.tta_flips = self.get_tta_flips()

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
        # deep supervision is defined in the denoiser
        if self.net.denoiser.deep_supervision:
            loss = self.loss(preds[0], label)
            for i, pred in enumerate(preds[1:]):
                downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(pred, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return self.loss(preds, label)


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
