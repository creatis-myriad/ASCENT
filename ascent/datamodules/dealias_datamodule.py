import numpy as np
from monai.transforms import (
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandZoomd,
    SelectItemsd,
    SpatialPadd,
    SplitDimd,
)

from ascent.datamodules.nnunet_datamodule import nnUNetDataModule
from ascent.utils.transforms import (
    ArtfclAliasingd,
    Convert2Dto3Dd,
    Convert3Dto2Dd,
    DealiasLoadNpyd,
    MayBeSqueezed,
)


class DealiasDataModule(nnUNetDataModule):
    """Data module for dealiasing using deep unfolding."""

    def __init__(self, alias_transform: bool = True, **kwargs):
        """Initializes class instance.

        Args:
            alias_transform: Whether to apply artificial aliasing augmentation.
        """

        super().__init__(**kwargs)

    def setup_transforms(self) -> None:
        """Define the data augmentations used by nnUNet including the data reading using
        monai.transforms libraries.

        An additional artificial aliasing augmentation is added on top of the data augmentations
        defined in nnUNetDataModule.
        """
        if self.threeD:
            rot_inter_mode = "bilinear"
            zoom_inter_mode = "trilinear"
            range_x = range_y = range_z = [-30.0 / 180 * np.pi, 30.0 / 180 * np.pi]

            if self.hparams.do_dummy_2D_data_aug:
                zoom_inter_mode = rot_inter_mode = "bicubic"
                range_x = [-180.0 / 180 * np.pi, 180.0 / 180 * np.pi]
                range_y = range_z = 0.0

        else:
            zoom_inter_mode = rot_inter_mode = "bicubic"
            range_x = [-180.0 / 180 * np.pi, 180.0 / 180 * np.pi]
            range_y = range_z = 0.0
            if max(self.hparams.patch_size) / min(self.hparams.patch_size) > 1.5:
                range_x = [-15.0 / 180 * np.pi, 15.0 / 180 * np.pi]

        shared_train_val_transforms = [
            DealiasLoadNpyd(keys=["data"]),
            EnsureChannelFirstd(keys=["image", "label", "seg"]),
            SpatialPadd(
                keys=["image", "label", "seg"],
                spatial_size=self.crop_patch_size,
                mode="constant",
                value=0,
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label", "seg"],
                label_key="seg",
                spatial_size=self.crop_patch_size,
                pos=0.33,
                neg=0.67,
                num_samples=1,
            ),
        ]

        other_transforms = []

        if not self.threeD:
            other_transforms.append(MayBeSqueezed(keys=["image", "label", "seg"], dim=-1))

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(Convert3Dto2Dd(keys=["image", "label", "seg"]))

        if self.hparams.alias_transform:
            other_transforms.append(ArtfclAliasingd(keys=["image", "label", "seg"], prob=0.5))

        other_transforms.extend(
            [
                RandRotated(
                    keys=["image", "label", "seg"],
                    range_x=range_x,
                    range_y=range_y,
                    range_z=range_z,
                    mode=[rot_inter_mode, rot_inter_mode, "nearest"],
                    padding_mode="zeros",
                    prob=0.2,
                ),
                RandZoomd(
                    keys=["image", "label", "seg"],
                    min_zoom=0.7,
                    max_zoom=1.4,
                    mode=[zoom_inter_mode, zoom_inter_mode, "nearest"],
                    padding_mode="constant",
                    align_corners=(True, True, None),
                    prob=0.2,
                ),
            ]
        )

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(Convert2Dto3Dd(keys=["image", "label", "seg"], in_channels=2))

        other_transforms.extend(
            [
                SplitDimd(keys=["image"], dim=0),
                RandGaussianNoised(keys=["image_0", "label"], std=0.01, prob=0.15),
                RandGaussianSmoothd(
                    keys=["image_0", "label"],
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    prob=0.15,
                ),
                RandScaleIntensityd(keys=["image_0", "label"], factors=0.3, prob=0.15),
                RandAdjustContrastd(keys=["image_0", "label"], gamma=(0.7, 1.5), prob=0.3),
                SelectItemsd(keys=["image_0", "image_1", "label", "seg"]),
                ConcatItemsd(keys=["image_0", "image_1"], name="image"),
                SelectItemsd(keys=["image", "label", "seg"]),
                RandFlipd(keys=["image", "label", "seg"], spatial_axis=[0], prob=0.5),
                RandFlipd(keys=["image", "label", "seg"], spatial_axis=[1], prob=0.5),
            ]
        )

        if self.threeD:
            other_transforms.append(
                RandFlipd(keys=["image", "label", "seg"], spatial_axis=[2], prob=0.5)
            )

        val_transforms = shared_train_val_transforms

        if not self.threeD:
            val_transforms.append(MayBeSqueezed(keys=["image", "label", "seg"], dim=-1))

        test_transforms = [
            DealiasLoadNpyd(
                keys=["data", "image_meta_dict"],
                test=True,
            ),
            EnsureChannelFirstd(keys=["image", "label", "seg"]),
        ]

        if not self.threeD:
            test_transforms.append(MayBeSqueezed(keys=["image", "label", "seg"], dim=-1))

        self.train_transforms = Compose(shared_train_val_transforms + other_transforms)
        self.val_transforms = Compose(val_transforms)
        self.test_transforms = Compose(test_transforms)


if __name__ == "__main__":
    import hydra
    import pyrootutils
    from hydra import compose, initialize_config_dir
    from matplotlib import pyplot as plt
    from omegaconf import OmegaConf
    from pytorch_lightning.trainer.states import TrainerFn

    from ascent.utils.visualization import dopplermap, imagesc

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    initialize_config_dir(
        config_dir=str(root / "configs" / "datamodule"), job_name="test", version_base="1.2"
    )
    cfg = compose(config_name="unwrap_2d.yaml")

    cfg.data_dir = str(root / "data")
    cfg.in_channels = 3
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
    seg = batch["seg"][0].array
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    print(batch["image"][0]._meta["filename_or_obj"])
    plt.figure("image", (18, 6))
    ax = plt.subplot(1, 3, 1)
    imagesc(ax, img[0, :, :].transpose(), "image", cmap, clim=[-1, 1])
    ax = plt.subplot(1, 3, 2)
    imagesc(ax, label[0, :, :].transpose(), "label", cmap, clim=[-1, 1])
    ax = plt.subplot(1, 3, 3)
    imagesc(ax, seg[0, :, :].transpose(), "seg", cmap)
    print("max of seg: ", np.max(seg[0, :, :]))
    plt.show()
