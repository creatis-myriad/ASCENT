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
    """Data module for dealiasing using segmentation/deep unfolding/primal-dual."""

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
            kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)

    def setup_transforms(self) -> None:
        """Define the data augmentations used by nnUNet including the data reading using
        `monai.transforms` libraries.

        An additional artificial aliasing augmentation is added on top of the data augmentations
        defined in nnUNetDataModule.
        """
        (
            train_transforms,
            val_transforms,
            test_transforms,
        ) = self._get_train_val_test_loading_transforms()

        other_transforms = []

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(Convert3Dto2Dd(keys=self.hparams.data_keys.all_keys))

        if self.hparams.augmentation.get("aliasing"):
            for name, aug in self.hparams.augmentation.get("aliasing").items():
                other_transforms.append(aug)

        if self.hparams.augmentation.get("rotation"):
            for name, aug in self.hparams.augmentation.get("rotation").items():
                other_transforms.append(aug)

        if self.hparams.augmentation.get("zoom"):
            for name, aug in self.hparams.augmentation.get("zoom").items():
                other_transforms.append(aug)

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(
                Convert2Dto3Dd(
                    keys=self.hparams.data_keys.all_keys, num_channel=self.hparams.in_channels
                )
            )

        # Split the image into two channels, one for aliased Doppler and one for Doppler power
        if self.hparams.separate_transform:
            other_transforms.append(SplitDimd(keys=self.hparams.data_keys.image_key, dim=0))

        if self.hparams.augmentation.get("noise"):
            for name, aug in self.hparams.augmentation.get("noise").items():
                other_transforms.append(aug)

        if self.hparams.augmentation.get("intensity"):
            for name, aug in self.hparams.augmentation.get("intensity").items():
                other_transforms.append(aug)

        if (
            self.hparams.separate_transform
            or self.hparams.exclude_Dpower
            or self.hparams.in_channels == 1
        ):
            resting_keys_to_keep = self.hparams.data_keys.all_keys.copy()
            resting_keys_to_keep.remove(self.hparams.data_keys.image_key)

        # Concatenate the two channels back into one image
        if self.hparams.separate_transform:
            other_transforms.extend(
                [
                    SelectItemsd(
                        keys=[
                            f"{self.hparams.data_keys.image_key}_0",
                            f"{self.hparams.data_keys.image_key}_1",
                            *resting_keys_to_keep,
                        ]
                    ),
                    ConcatItemsd(
                        keys=[
                            f"{self.hparams.data_keys.image_key}_0",
                            f"{self.hparams.data_keys.image_key}_1",
                        ],
                        name=self.hparams.data_keys.image_key,
                    ),
                    SelectItemsd(keys=self.hparams.data_keys.all_keys),
                ]
            )

        if self.hparams.augmentation.get("flip"):
            for name, aug in self.hparams.augmentation.get("flip").items():
                other_transforms.append(aug)

        # Exclude Doppler power (channel 1) from the multichannel input if required
        if self.hparams.exclude_Dpower and self.hparams.in_channels == 1:
            other_transforms.extend(
                [
                    SplitDimd(keys=[self.hparams.data_keys.image_key], dim=0),
                    SelectItemsd(
                        keys=[
                            f"{self.hparams.data_keys.image_key}_0",
                            *resting_keys_to_keep,
                        ]
                    ),
                    CopyItemsd(
                        keys=[f"{self.hparams.data_keys.image_key}_0"],
                        names=self.hparams.data_keys.image_key,
                    ),
                    SelectItemsd(keys=self.hparams.data_keys.all_keys),
                ]
            )

        # Exclude Doppler power (channel 1) from the multichannel input if required
        if self.hparams.exclude_Dpower and self.hparams.in_channels == 1:
            val_transforms.extend(
                [
                    SplitDimd(keys=[self.hparams.data_keys.image_key], dim=0),
                    SelectItemsd(
                        keys=[
                            f"{self.hparams.data_keys.image_key}_0",
                            *resting_keys_to_keep,
                        ]
                    ),
                    CopyItemsd(
                        keys=[f"{self.hparams.data_keys.image_key}_0"],
                        names=self.hparams.data_keys.image_key,
                    ),
                    SelectItemsd(keys=self.hparams.data_keys.all_keys),
                ]
            )

        # Exclude Doppler power (channel 1) from the multichannel input if required
        if self.hparams.exclude_Dpower and self.hparams.in_channels == 1:
            test_transforms.extend(
                [
                    SplitDimd(keys=[self.hparams.data_keys.image_key], dim=0),
                    SelectItemsd(
                        keys=[
                            f"{self.hparams.data_keys.image_key}_0",
                            *resting_keys_to_keep,
                            "image_meta_dict",
                        ]
                    ),
                    CopyItemsd(keys=["image_0"], names=self.hparams.data_keys.image_key),
                    SelectItemsd(keys=[*self.hparams.data_keys.all_keys, "image_meta_dict"]),
                ]
            )

        self.train_transforms = Compose(train_transforms + other_transforms)
        self.val_transforms = Compose(val_transforms)
        self.test_transforms = Compose(test_transforms)
