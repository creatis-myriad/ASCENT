from monai.transforms import Compose, EnsureChannelFirstd, RandSpatialCropd, SpatialPadd

from ascent.datamodules.nnunet_datamodule import nnUNetDataModule
from ascent.utils.transforms import Convert2Dto3Dd, Convert3Dto2Dd, LoadNpyd, MayBeSqueezed


class nnUNetRegDataModule(nnUNetDataModule):
    """Data module for regression."""

    def setup_transforms(self) -> None:
        """Define the data augmentations used by nnUNet including the data reading using
        `monai.transforms` libraries.

        The only difference with nnUNet framework is the patch creation.
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

        if self.augmentation.get("rotate"):
            other_transforms.append(self.augmentation.get("rotate"))

        if self.augmentation.get("zoom"):
            other_transforms.append(self.augmentation.get("zoom"))

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(
                Convert2Dto3Dd(keys=["image", "label"], num_channel=self.hparams.in_channels)
            )

        if self.augmentation.get("gaussian_noise"):
            other_transforms.append(self.augmentation.get("gaussian_noise"))

        if self.augmentation.get("gaussian_smooth"):
            other_transforms.append(self.augmentation.get("gaussian_smooth"))

        if self.augmentation.get("scale_intensity"):
            other_transforms.append(self.augmentation.get("scale_intensity"))

        if self.augmentation.get("adjust_contrast"):
            other_transforms.append(self.augmentation.get("adjust_contrast"))

        if self.augmentation.get("flip_x"):
            other_transforms.append(self.augmentation.get("flip_x"))

        if self.augmentation.get("flip_y"):
            other_transforms.append(self.augmentation.get("flip_y"))

        if self.threeD:
            if self.augmentation.get("flip_z"):
                other_transforms.append(self.augmentation.get("flip_z"))

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
