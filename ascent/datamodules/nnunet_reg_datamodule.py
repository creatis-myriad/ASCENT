import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandSpatialCropd,
    RandZoomd,
    SpatialPadd,
)

from ascent.datamodules.nnunet_datamodule import nnUNetDataModule
from ascent.utils.transforms import Convert2Dto3Dd, Convert3Dto2Dd, LoadNpyd, MayBeSqueezed


class nnUNetRegDataModule(nnUNetDataModule):
    """Data module for regression."""

    def setup_transforms(self) -> None:
        """Define the data augmentations used by nnUNet including the data reading using
        monai.transforms libraries.

        The only difference with nnUNet framework is the patch creation.
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
            LoadNpyd(keys=["data"], seg_label=self.hparams.seg_label),
            EnsureChannelFirstd(keys=["image", "label"]),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=self.crop_patch_size,
                mode="constant",
                value=0,
            ),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=self.crop_patch_size,
                random_size=False,
            ),
        ]

        other_transforms = []

        if not self.threeD:
            other_transforms.append(MayBeSqueezed(keys=["image", "label"], dim=-1))

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(Convert3Dto2Dd(keys=["image", "label"]))

        other_transforms.extend(
            [
                RandRotated(
                    keys=["image", "label"],
                    range_x=range_x,
                    range_y=range_y,
                    range_z=range_z,
                    mode=rot_inter_mode,
                    padding_mode="zeros",
                    prob=0.2,
                ),
                RandZoomd(
                    keys=["image", "label"],
                    min_zoom=0.7,
                    max_zoom=1.4,
                    mode=zoom_inter_mode,
                    padding_mode="constant",
                    align_corners=True,
                    prob=0.2,
                ),
            ]
        )

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(
                Convert2Dto3Dd(keys=["image", "label"], num_channel=self.hparams.in_channels)
            )

        other_transforms.extend(
            [
                RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
                RandGaussianSmoothd(
                    keys=["image"],
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    prob=0.15,
                ),
                RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
                RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),
                RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
                RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
            ]
        )

        if self.threeD:
            other_transforms.append(RandFlipd(["image", "label"], spatial_axis=[2], prob=0.5))

        val_transforms = shared_train_val_transforms.copy()

        if not self.threeD:
            val_transforms.append(MayBeSqueezed(keys=["image", "label"], dim=-1))

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
    cfg = compose(config_name="unwrap_2d.yaml")
    print(OmegaConf.to_yaml(cfg))

    cfg.data_dir = str(root / "data")
    # cfg.patch_size = [128, 128]
    cfg.in_channels = 3
    cfg.patch_size = [40, 192]
    # cfg.patch_size = [128, 128, 12]
    cfg.batch_size = 1
    cfg.fold = 0
    camus_datamodule = hydra.utils.instantiate(cfg)
    camus_datamodule.prepare_data()
    camus_datamodule.setup(stage=TrainerFn.FITTING)
    train_dl = camus_datamodule.train_dataloader()
    # camus_datamodule.setup(stage=TrainerFn.TESTING)
    # test_dl = camus_datamodule.test_dataloader()

    # predict_files = [
    #     {
    #         "image_0": "C:/Users/ling/Desktop/nnUNet/nnUNet_raw/nnUNet_raw_data/Task129_DealiasingConcat/imagesTr/Dealias_0001_0000.nii.gz",
    #         "image_1": "C:/Users/ling/Desktop/nnUNet/nnUNet_raw/nnUNet_raw_data/Task129_DealiasingConcat/imagesTr/Dealias_0001_0001.nii.gz",
    #     }
    # ]

    # camus_datamodule.prepare_for_prediction(predict_files)
    # camus_datamodule.setup(stage=TrainerFn.PREDICTING)
    # predict_dl = camus_datamodule.predict_dataloader()
    gen = iter(train_dl)
    batch = next(gen)
    # batch = next(iter(test_dl))
    # batch = next(iter(predict_dl))

    cmap = dopplermap()

    img = batch["image"][0]
    label = batch["label"][0]
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    print(img._meta["filename_or_obj"])
    plt.figure("image", (18, 6))
    ax = plt.subplot(1, 2, 1)
    imagesc(ax, img[0, :, :], "image", cmap, clim=[-1, 1])
    ax = plt.subplot(1, 2, 2)
    imagesc(ax, label[0, :, :], "label", cmap, clim=[-1, 1])
    plt.show()
