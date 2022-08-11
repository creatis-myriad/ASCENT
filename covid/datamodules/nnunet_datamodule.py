import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, IterableDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensity,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandZoomd,
    SpatialPadd,
    ToTensord,
)
from pytorch_lightning import LightningDataModule

from covid.datamodules.components.nnunet_iterator import nnUNet_Iterator
from covid.datamodules.components.transforms import (
    Convert2Dto3D,
    Convert3Dto2D,
    MayBeSqueezed,
)
from covid.utils.file_and_folder_operations import load_pickle


class nnUNetDataModule(LightningDataModule):
    """Lightning DataModule for nnUNet pipeline.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        dataset_name: str = "CAMUS",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        fold: int = 0,
        batch_size: int = 2,
        patch_size: Tuple[int, ...] = (128, 128, 128),
        num_classes: int = None,
        in_channels: int = 1,
        do_dummy_2D_aug: bool = True,
        do_test: bool = False,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.full_data_dir = os.path.join(data_dir, dataset_name, "raw")

        assert len(patch_size) in [2, 3], "Only 2D and 3D patches are supported right now!"

        self.crop_patch_size = patch_size
        self.threeD = len(patch_size) == 3
        if not self.threeD:
            self.crop_patch_size = [*patch_size, 1]

        # data transformations
        self.train_transforms = self.val_transforms = self.test_transform = []
        self.setup_transforms()

        self.data_train: Optional[torch.utils.Dataset] = None
        self.data_val: Optional[torch.utils.Dataset] = None
        if do_test:
            self.data_test: Optional[torch.utils.Dataset] = None

    @property
    def num_classeses(self):
        return self.hparams.num_classes

    def prepare_data(self):
        splits_file = os.path.join(self.full_data_dir, "splits_final.pkl")
        print("Using splits from existing split file:", splits_file)
        splits = load_pickle(splits_file)
        if self.hparams.fold < len(splits):
            print("Desired fold for training: %d" % self.hparams.fold)
            train_keys = splits[self.hparams.fold]["train"]
            val_keys = splits[self.hparams.fold]["val"]
            if self.hparams.do_test:
                test_keys = splits[self.hparams.fold]["test"]
            if self.hparams.do_test:
                print(
                    "This split has %d training, %d validation, and %d testing cases."
                    % (len(train_keys), len(val_keys), len(test_keys))
                )
            else:
                print(
                    "This split has %d training and %d validation cases."
                    % (len(train_keys), len(val_keys))
                )

            self.train_files = [
                {
                    "image": os.path.join(self.full_data_dir, "imagesTr", key + "_0000.nii.gz"),
                    "label": os.path.join(self.full_data_dir, "labelsTr", key + ".nii.gz"),
                }
                for key in train_keys
            ]
            self.val_files = [
                {
                    "image": os.path.join(self.full_data_dir, "imagesTr", key + "_0000.nii.gz"),
                    "label": os.path.join(self.full_data_dir, "labelsTr", key + ".nii.gz"),
                }
                for key in val_keys
            ]
            if self.hparams.do_test:
                self.test_files = [
                    {
                        "image": os.path.join(
                            self.full_data_dir, "imagesTr", key + "_0000.nii.gz"
                        ),
                        "label": os.path.join(self.full_data_dir, "labelsTr", key + ".nii.gz"),
                    }
                    for key in test_keys
                ]

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.data_train = IterableDataset(
            data=nnUNet_Iterator(self.train_files), transform=self.train_transforms
        )

        self.data_val = IterableDataset(
            data=nnUNet_Iterator(self.val_files), transform=self.val_transforms
        )

        if self.hparams.do_test:
            self.data_test = CacheDataset(
                data=self.test_files, transform=self.test_transforms, cache_rate=1.0
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        if self.hparams.do_test:
            return DataLoader(
                dataset=self.data_test,
                batch_size=1,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def setup_transforms(self):
        """Define the data augmentations used by nnUNet including the data reading using
        monai.transforms libraries.

        The only difference with nnUNet framework is the patch creation.
        """
        if self.threeD:
            inter_mode = "trilinear"
            range_x = range_y = range_z = [-30.0 / 180 * np.pi, 30.0 / 180 * np.pi]

            if self.hparams.do_dummy_2D_aug:
                inter_mode = "bicubic"
                range_x = [-180.0 / 180 * np.pi, 180.0 / 180 * np.pi]
                range_y = range_z = 0

        else:
            inter_mode = "bicubic"
            range_x = [-180.0 / 180 * np.pi, 180.0 / 180 * np.pi]
            range_y = range_z = 0
            if max(self.hparams.patch_size) / min(self.hparams.patch_size) > 1.5:
                range_x = [-15 / 180 * np.pi, 15 / 180 * np.pi]

        shared_train_val_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
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

        if self.hparams.do_dummy_2D_aug and self.threeD:
            other_transforms.append(Convert3Dto2D(keys=["image", "label"]))

        other_transforms.extend(
            [
                RandRotated(
                    keys=["image", "label"],
                    range_x=range_x,
                    range_y=range_y,
                    range_z=range_z,
                    mode=(inter_mode, "nearest"),
                    padding_mode="zeros",
                    prob=0.2,
                ),
                RandZoomd(
                    keys=["image", "label"],
                    min_zoom=0.7,
                    max_zoom=1.4,
                    mode=(inter_mode, "nearest"),
                    padding_mode="constant",
                    align_corners=(True, None),
                    prob=0.2,
                ),
            ]
        )

        if self.hparams.do_dummy_2D_aug and self.threeD:
            other_transforms.append(
                Convert2Dto3D(keys=["image", "label"], in_channels=self.hparams.in_channels)
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

        val_transforms = shared_train_val_transforms

        if not self.threeD:
            val_transforms.append(MayBeSqueezed(keys=["image", "label"], dim=-1))

        test_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True),
        ]

        if not self.threeD:
            test_transforms.append(MayBeSqueezed(keys=["image", "label"], dim=-1))

        self.train_transforms = Compose(shared_train_val_transforms + other_transforms)
        self.val_transforms = Compose(val_transforms)
        self.test_transforms = Compose(test_transforms)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "camus.yaml")
    cfg.data_dir = str(root / "data")
    cfg.patch_size = [640, 512]
    # cfg.patch_size = [128, 128, 12]
    cfg.batch_size = 1
    cfg.fold = 0
    camus_datamodule = hydra.utils.instantiate(cfg)
    camus_datamodule.prepare_data()
    camus_datamodule.setup()
    train_dl = camus_datamodule.train_dataloader()
    train_dl = camus_datamodule.test_dataloader()

    batch = next(iter(train_dl))

    from matplotlib import pyplot as plt

    img = batch["image"][0]
    label = batch["label"][0]
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[0, :, :], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[0, :, :])
    plt.show()
