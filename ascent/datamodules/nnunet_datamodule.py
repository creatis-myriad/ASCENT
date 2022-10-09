import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="torchaudio")

import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from monai.data import CacheDataset, DataLoader, IterableDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandSpatialCropd,
    RandZoomd,
    SpatialPadd,
)
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.model_selection import KFold, train_test_split

from ascent.datamodules.components.data_loading import (
    get_case_identifiers_from_npz_folders,
)
from ascent.datamodules.components.nnunet_iterator import nnUNet_Iterator
from ascent.datamodules.components.transforms import (
    Convert2Dto3Dd,
    Convert3Dto2Dd,
    LoadNpyd,
    MayBeSqueezed,
)
from ascent.utils.file_and_folder_operations import load_pickle, save_pickle, subfiles


class nnUNetDataModule(LightningDataModule):
    """Data module for nnUnet pipeline."""

    def __init__(
        self,
        data_dir: str = "data/",
        dataset_name: str = "CAMUS",
        fold: int = 0,
        batch_size: int = 2,
        patch_size: tuple[int, ...] = (128, 128, 128),
        in_channels: int = 1,
        do_dummy_2D_data_aug: bool = True,
        num_workers: int = os.cpu_count() - 1,
        pin_memory: bool = True,
        test_splits: bool = True,
        seg_label: bool = True,
    ):
        """Initializes class instance.

        Args:
            data_dir: Path to the data directory.
            dataset_name: Name of dataset to be used.
            train_split_ratio: Data split ratio for training.
            fold: Fold to be used for training, validation or test.
            batch_size: Batch size to be used for training and validation.
            patch_size: Patch size to crop the data..
            in_channels: Number of input channels.
            do_dummy_2D_data_aug: Whether to apply 2D transformation on 3D dataset.
            do_test: Whether to run inference on test dataset and compute test scores.
            num_workers: Number of subprocesses to use for data loading.
            pin_memory: Whether to pin memory to GPU.
            test_splits: Whether to split data into train/val/test (0.8/0.1/0.1).
            seg_label: Whether the labels are segmentations.
        """

        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.preprocessed_folder = os.path.join(data_dir, dataset_name, "preprocessed")
        self.full_data_dir = os.path.join(
            data_dir, dataset_name, "preprocessed", "data_and_properties"
        )

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
        self.data_test: Optional[torch.utils.Dataset] = None

    def prepare_data(self) -> None:
        """Data preparation.

        Unpacking .npz data to .npy for faster data loading during training.
        """

        print("\nUnpacking dataset...")
        self.unpack_dataset()
        print("Done")

    @staticmethod
    def do_splits(splits_file: Union[Path, str], preprocessed_path: Union[Path, str], test_splits):
        """Create 5-fold train/validation/test splits."""

        if not os.path.isfile(splits_file):
            print("Creating new split...")
            splits = []
            all_keys_sorted = np.sort(
                list(get_case_identifiers_from_npz_folders(preprocessed_path))
            )
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, val_and_test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                val_and_test_keys = np.array(all_keys_sorted)[val_and_test_idx]
                val_and_test_keys_sorted = np.sort(val_and_test_keys)
                if test_splits:
                    kfold_2 = KFold(n_splits=2, shuffle=True, random_state=12345)
                    _, (val_idx, test_idx) = kfold_2.split(val_and_test_keys_sorted)
                    val_keys = np.array(val_and_test_keys_sorted)[val_idx]
                    test_keys = np.array(val_and_test_keys_sorted)[test_idx]
                else:
                    val_keys = val_and_test_keys_sorted
                splits.append(OrderedDict())
                splits[-1]["train"] = train_keys
                splits[-1]["val"] = val_keys
                if test_splits:
                    splits[-1]["test"] = test_keys
            save_pickle(splits, splits_file)
        else:
            print("Using splits from existing split file:", splits_file)

    def setup(self, stage: Optional[str] = None):
        """Load data.

        More detailed steps:
        1. Split the dataset into 5 train, validation and test folds (80:10:10) if it was not done.
        2. Use the specified fold for training. Create random 80:10:10 split if requested fold is
           larger than the length of saved splits.
        3. Set variables: `self.data_train`, `self.data_val`, `self.data_test`, `self.data_predict`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        if stage == TrainerFn.FITTING or stage == TrainerFn.TESTING:
            splits_file = os.path.join(self.preprocessed_folder, "splits_final.pkl")
            self.do_splits(splits_file, self.full_data_dir, self.hparams.test_splits)
            splits = load_pickle(splits_file)
            if self.hparams.fold < len(splits):
                print("Desired fold for training: %d" % self.hparams.fold)
                train_keys = splits[self.hparams.fold]["train"]
                val_keys = splits[self.hparams.fold]["val"]

                if self.hparams.test_splits:
                    test_keys = splits[self.hparams.fold]["test"]
                    print(
                        "This split has %d training, %d validation, and %d testing cases."
                        % (len(train_keys), len(val_keys), len(test_keys))
                    )
                else:
                    print(
                        "This split has %d training and %d validation cases."
                        % (len(train_keys), len(val_keys))
                    )
            else:
                print(
                    "INFO: You requested fold %d for training but splits "
                    "contain only %d folds. I am now creating a "
                    "random (but seeded) 80:10:10 split!" % (self.hparams.fold, len(splits))
                )
                # if we request a fold that is not in the split file, create a random 80:10:10 split
                keys = np.sort(list(get_case_identifiers_from_npz_folders(self.full_data_dir)))
                train_keys, val_and_test_keys = train_test_split(
                    keys, train_size=0.8, random_state=(12345 + self.hparams.fold)
                )
                if self.hparams.test_splits:
                    val_keys, test_keys = train_test_split(
                        val_and_test_keys, test_size=0.5, random_state=(12345 + self.hparams.fold)
                    )
                    print(
                        "This random 80:10:10 split has %d training, %d validation, and %d testing cases."
                        % (len(train_keys), len(val_keys), len(test_keys))
                    )
                else:
                    val_keys = val_and_test_keys
                    print(
                        "This random 80:20 split has %d training and %d validation cases."
                        % (len(train_keys), len(val_keys))
                    )

            self.train_files = [
                {
                    "data": os.path.join(self.full_data_dir, "%s.npy" % key),
                }
                for key in train_keys
            ]
            self.val_files = [
                {
                    "data": os.path.join(self.full_data_dir, "%s.npy" % key),
                }
                for key in val_keys
            ]
            if stage == TrainerFn.TESTING:
                if self.hparams.test_splits:
                    self.test_files = [
                        {
                            "data": os.path.join(self.full_data_dir, "%s.npy" % key),
                            "image_meta_dict": os.path.join(self.full_data_dir, "%s.pkl" % key),
                        }
                        for key in test_keys
                    ]
                else:
                    self.test_files = [
                        {
                            "data": os.path.join(self.full_data_dir, "%s.npy" % key),
                            "image_meta_dict": os.path.join(self.full_data_dir, "%s.pkl" % key),
                        }
                        for key in val_keys
                    ]

        if stage == TrainerFn.FITTING:
            self.data_train = IterableDataset(
                data=nnUNet_Iterator(self.train_files), transform=self.train_transforms
            )
        if stage == TrainerFn.FITTING:
            self.data_val = IterableDataset(
                data=nnUNet_Iterator(self.val_files), transform=self.val_transforms
            )
        if stage == TrainerFn.TESTING:
            self.data_test = CacheDataset(
                data=self.test_files, transform=self.test_transforms, cache_rate=1.0
            )

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        # We use a batch size of 1 for testing as the images have different shapes and we can't
        # stack them

        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

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

        other_transforms.extend(
            [
                RandRotated(
                    keys=["image", "label"],
                    range_x=range_x,
                    range_y=range_y,
                    range_z=range_z,
                    mode=[rot_inter_mode, "nearest"],
                    padding_mode="zeros",
                    prob=0.2,
                ),
                RandZoomd(
                    keys=["image", "label"],
                    min_zoom=0.7,
                    max_zoom=1.4,
                    mode=[zoom_inter_mode, "nearest"],
                    padding_mode="constant",
                    align_corners=(True, None),
                    prob=0.2,
                ),
            ]
        )

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(
                Convert2Dto3Dd(keys=["image", "label"], in_channels=self.hparams.in_channels)
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

    def unpack_dataset(self):
        """Unpack dataset from .npz to .npy.

        Use Parallel to speed up the unpacking process.
        """
        npz_files = subfiles(self.full_data_dir, True, None, ".npz", True)
        Parallel(n_jobs=self.hparams.num_workers)(
            delayed(self.convert_to_npy)(npz_file) for npz_file in npz_files
        )

    @staticmethod
    def convert_to_npy(npz_file, key="data"):
        """Convert .npz file to .npy."""
        if not os.path.isfile(npz_file[:-3] + "npy"):
            a = np.load(npz_file)[key]
            np.save(npz_file[:-3] + "npy", a)


if __name__ == "__main__":
    import hydra
    import pyrootutils
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    initialize_config_dir(config_dir=str(root / "configs" / "datamodule"), job_name="test")
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
