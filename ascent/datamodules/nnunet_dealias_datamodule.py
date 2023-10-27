import numpy as np
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CopyItemsd,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    SelectItemsd,
    SpatialPadd,
    SplitDimd,
)

from ascent.datamodules.nnunet_datamodule import nnUNetDataModule
from ascent.utils.transforms import Convert2Dto3Dd, Convert3Dto2Dd, LoadNpyd, MayBeSqueezed


class nnUNetDealiasDataModule(nnUNetDataModule):
    """Data module for dealiasing using segmentation."""

    def __init__(
        self,
        separate_transform: bool = True,
        exclude_Dpower: bool = False,
        **kwargs,
    ):
        """Initialize class instance.

        Args:
            alias_transform: Whether to apply artificial aliasing augmentation.
            separate_transform: Whether to apply separate on Doppler power and velocity.
            exclude_Dpower: Whether to exclude Doppler power in case of velocity-power concatenated
                input.
        """
        super().__init__(**kwargs)

    def setup_transforms(self) -> None:
        """Define the data augmentations used by nnUNet including the data reading using
        `monai.transforms` libraries.

        An additional artificial aliasing augmentation is added on top of the data augmentations
        defined in nnUNetDataModule.
        """
        shared_train_val_transforms = [
            LoadNpyd(keys=["data"], seg_label=self.hparams.seg_label),
            EnsureChannelFirstd(keys=["image", "label"]),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=self.crop_patch_size,
                mode="constant",
                value=0,
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.crop_patch_size,
                pos=0.33,
                neg=0.67,
                num_samples=1,
            ),
        ]

        other_transforms = []

        if not self.threeD:
            other_transforms.append(MayBeSqueezed(keys=["image", "label"], dim=-1))

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(Convert3Dto2Dd(keys=["image", "label"]))

        if self.augmentation.get("artificial_aliasing"):
            other_transforms.append(self.augmentation.get("artificial_aliasing"))

        if self.augmentation.get("rotate"):
            other_transforms.append(self.augmentation.get("rotate"))

        if self.augmentation.get("zoom"):
            other_transforms.append(self.augmentation.get("zoom"))

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(
                Convert2Dto3Dd(keys=["image", "label"], num_channel=self.hparams.in_channels)
            )

        # Split the image into two channels, one for aliased Doppler and one for Doppler power
        if self.hparams.separate_transform:
            other_transforms.append(SplitDimd(keys=["image"], dim=0))

        if self.augmentation.get("gaussian_noise"):
            other_transforms.append(self.augmentation.get("gaussian_noise"))

        if self.augmentation.get("gaussian_smooth"):
            other_transforms.append(self.augmentation.get("gaussian_smooth"))

        if self.augmentation.get("scale_intensity"):
            other_transforms.append(self.augmentation.get("scale_intensity"))

        if self.augmentation.get("adjust_contrast"):
            other_transforms.append(self.augmentation.get("adjust_contrast"))

        # Concatenate the two channels back into one image
        if self.hparams.separate_transform:
            other_transforms.extend(
                [
                    SelectItemsd(keys=["image_0", "image_1", "label"]),
                    ConcatItemsd(keys=["image_0", "image_1"], name="image"),
                    SelectItemsd(keys=["image", "label"]),
                ]
            )

        if self.augmentation.get("flip_x"):
            other_transforms.append(self.augmentation.get("flip_x"))

        if self.augmentation.get("flip_y"):
            other_transforms.append(self.augmentation.get("flip_y"))

        if self.threeD:
            if self.augmentation.get("flip_z"):
                other_transforms.append(self.augmentation.get("flip_z"))

        # Exclude Doppler power (channel 1) from the multichannel input if required
        if self.hparams.exclude_Dpower and self.hparams.in_channels == 1:
            other_transforms.extend(
                [
                    SplitDimd(keys=["image"], dim=0),
                    SelectItemsd(keys=["image_0", "label"]),
                    CopyItemsd(keys=["image_0"], names="image"),
                    SelectItemsd(keys=["image", "label"]),
                ]
            )

        val_transforms = shared_train_val_transforms.copy()

        if not self.threeD:
            val_transforms.append(MayBeSqueezed(keys=["image", "label"], dim=-1))

        # Exclude Doppler power (channel 1) from the multichannel input if required
        if self.hparams.exclude_Dpower and self.hparams.in_channels == 1:
            val_transforms.extend(
                [
                    SplitDimd(keys=["image"], dim=0),
                    SelectItemsd(keys=["image_0", "label"]),
                    CopyItemsd(keys=["image_0"], names="image"),
                    SelectItemsd(keys=["image", "label"]),
                ]
            )

        test_transforms = [
            LoadNpyd(
                keys=["data", "image_meta_dict"],
                test=True,
                seg_label=self.hparams.seg_label,
            ),
            EnsureChannelFirstd(keys=["image", "label"]),
        ]

        if not self.threeD:
            test_transforms.append(MayBeSqueezed(keys=["image", "label"], dim=-1))

        # Exclude Doppler power (channel 1) from the multichannel input if required
        if self.hparams.exclude_Dpower and self.hparams.in_channels == 1:
            test_transforms.extend(
                [
                    SplitDimd(keys=["image"], dim=0),
                    SelectItemsd(keys=["image_0", "label", "image_meta_dict"]),
                    CopyItemsd(keys=["image_0"], names="image"),
                    SelectItemsd(keys=["image", "label", "image_meta_dict"]),
                ]
            )

        self.train_transforms = Compose(shared_train_val_transforms + other_transforms)
        self.val_transforms = Compose(val_transforms)
        self.test_transforms = Compose(test_transforms)


if __name__ == "__main__":
    import hydra
    import pyrootutils
    from hydra import compose, initialize_config_dir
    from lightning.pytorch.trainer.states import TrainerFn
    from matplotlib import pyplot as plt
    from omegaconf import OmegaConf

    from ascent.utils.visualization import dopplermap, imagesc

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    initialize_config_dir(
        config_dir=str(root / "configs" / "datamodule"), job_name="test", version_base="1.2"
    )
    cfg = compose(config_name="dealias_2d.yaml")
    cfg.in_channels = 1
    cfg.data_dir = str(root / "data")
    cfg.patch_size = [40, 192]
    cfg.batch_size = 1
    cfg.fold = 0
    print(OmegaConf.to_yaml(cfg))

    datamodule = hydra.utils.instantiate(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage=TrainerFn.FITTING)
    train_dl = datamodule.train_dataloader()
    gen = iter(train_dl)
    batch = next(gen)

    # datamodule.setup(stage=TrainerFn.TESTING)
    # test_dl = datamodule.test_dataloader()
    # gen = iter(test_dl)
    # batch = next(gen)

    cmap = dopplermap()

    img = batch["image"][0].array
    label = batch["label"][0].array
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    print(batch["image"][0]._meta["filename_or_obj"])
    plt.figure("image", (18, 6))
    ax = plt.subplot(1, 3, 1)
    imagesc(ax, img[0, :, :].transpose(), "image", cmap, clim=[-1, 1])
    ax = plt.subplot(1, 3, 2)
    imagesc(ax, img[1, :, :].transpose(), "power", cmap, clim=[-1, 1])
    ax = plt.subplot(1, 3, 3)
    imagesc(ax, label[0, :, :].transpose(), "label", cmap, clim=[0, 2])
    print("max of seg: ", np.max(label[0, :, :]))
    plt.show()
