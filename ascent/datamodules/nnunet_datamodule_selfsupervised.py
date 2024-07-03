import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Union

import hydra
import numpy as np
import torch
from joblib import Parallel, delayed
from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from monai.data import CacheDataset, DataLoader, IterableDataset
from monai.transforms import Compose
from omegaconf import DictConfig
from sklearn.model_selection import KFold, train_test_split

from ascent import utils
from ascent.utils.data_loading import get_case_identifiers_from_npz_folders
from ascent.utils.dataset import nnUNet_Iterator
from ascent.utils.dict_utils import flatten_dict
from ascent.utils.file_and_folder_operations import load_pickle, save_pickle, subfiles
from ascent.utils.transforms import Convert2Dto3Dd, Convert3Dto2Dd

log = utils.get_pylogger(__name__)


class nnUNetDataModule(LightningDataModule):
    """Data module for nnUnet pipeline."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/",
        dataset_name: str = "DE",
        fold: int = 0,
        batch_size: int = 2,
        patch_size: Union[tuple[int, ...], list[int]] = (128, 128, 128),
        in_channels: int = 1,
        do_dummy_2D_data_aug: bool = True,
        num_workers: int = os.cpu_count() - 1,
        pin_memory: bool = True,
        test_splits: bool = False,
        data_keys: DictConfig = None,
        augmentation: DictConfig = None,
        loading: DictConfig = None,
    ):
        """Initialize class instance.

        Args:
            data_dir: Path to the data directory.
            dataset_name: Name of dataset to be used.
            fold: Fold to be used for training, validation or test.
            batch_size: Batch size to be used for training and validation.
            patch_size: Patch size to crop the data.
            in_channels: Number of input channels.
            do_dummy_2D_data_aug: Whether to apply 2D transformation on 3D dataset.
            num_workers: Number of subprocesses to use for data loading.
            pin_memory: Whether to pin memory to GPU.
            test_splits: Whether to split data into train/val/test (0.8/0.1/0.1).
            data_keys: Dictionary config containing information about image key, label key, and
                all keys.
            augmentation: Dictionary config containing the transforms for data augmentation.
            loading: Dictionary config containing the transforms for data loading.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.preprocessed_folder = os.path.join(data_dir, dataset_name, "raw")
        self.full_data_dir = os.path.join(
            data_dir, dataset_name, "raw", "imagesTr"
        )

        if not len(patch_size) in [2, 3]:
            raise NotImplementedError("Only 2D and 3D patches are supported right now!")

        self.threeD = len(patch_size) == 3

        # data transformations
        self.train_transforms = self.val_transforms = self.test_transform = []

        # flatten nested augmentations dict to a single dict
        self.augmentation = flatten_dict(self.hparams.augmentation)

        # setup the transforms
        self.setup_transforms()

        self.data_train: Optional[torch.utils.Dataset] = None
        self.data_val: Optional[torch.utils.Dataset] = None

    def prepare_data(self) -> None:
        """Data preparation.

        Unpacking .npz data to .npy for faster data loading during training.
        """
        log.info("Unpacking dataset...")
        self.unpack_dataset()
        log.info("Done")

    @staticmethod
    def do_splits(
        splits_file: Union[Path, str], preprocessed_path: Union[Path, str], test_splits: bool
    ) -> None:
        """Create 5-fold train/validation splits or 10-fold train/validation/test splits if
        ```splits_final.pkl``` does not exist.

        Args:
            splits_file: Path containing ```splits_final.pkl```.
            preprocessed_path: Path to preprocessed folder.
            test_splits: Whether to do test splitting.
        """
        if not os.path.isfile(splits_file):
            log.info("Creating new split...")
            splits = []
            all_keys_sorted = np.sort(
                list(get_case_identifiers_from_npz_folders(preprocessed_path))
            )

            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, val_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                val_keys = np.array(all_keys_sorted)[val_idx]

                splits.append(OrderedDict())
                splits[-1]["train"] = train_keys
                splits[-1]["val"] = val_keys

            save_pickle(splits, splits_file)
        else:
            log.info(f"Using splits from existing split file: {splits_file}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        More detailed steps:
        1. Split the dataset into train, validation (and test) folds if it was not done.
        2. Use the specified fold for training. Create random 80:10:10 or 80:20 split if requested
           fold is larger than the length of saved splits.
        3. Set variables: `self.data_train`, `self.data_val`, `self.data_test`, `self.data_predict`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if stage == TrainerFn.FITTING or stage == TrainerFn.TESTING:
            splits_file = os.path.join(self.preprocessed_folder, "splits_final.pkl")
            self.do_splits(splits_file, self.full_data_dir, self.hparams.test_splits)
            splits = load_pickle(splits_file)
            if self.hparams.fold < len(splits):
                log.info(f"Desired fold for training or testing: {self.hparams.fold}")
                train_keys = splits[self.hparams.fold]["train"]
                val_keys = splits[self.hparams.fold]["val"]

                log.info(
                    f"This split has {len(train_keys)} training and {len(val_keys)} validation"
                    f" cases."
                )
            else:
                log.warning(
                    f"You requested fold {self.hparams.fold} for training but splits "
                    f"contain only {len(splits)} folds. I am now creating a "
                    f"random (but seeded) 80:20 split!"
                )
                # if we request a fold that is not in the split file, create a random 80:20 split
                keys = np.sort(list(get_case_identifiers_from_npz_folders(self.full_data_dir)))
                train_keys, val_keys = train_test_split(
                    keys, train_size=0.8, random_state=(12345 + self.hparams.fold)
                )
                log.info(
                    f"This random 80:20 split has {len(train_keys)} training and {len(val_keys)}"
                    f" validation cases."
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
                self.test_files = [
                    {
                        "data": os.path.join(self.full_data_dir, "%s.npy" % key),
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
            num_workers=max(self.hparams.num_workers, 1),
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=max(self.hparams.num_workers, 1),
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
            persistent_workers=True,
        )

    def get_train_val_test_loading_transforms(self) -> tuple[list[Callable], ...]:
        """Get the data loading transforms for train, validation and test.

        Returns:
            Tuple of lists of transforms for train, validation and test.
        """
        train_transforms = []
        test_transforms = []
        train_transforms.append(self.hparams.loading["train"].get("data_loading"))
        train_transforms.append(self.hparams.loading["train"].get("pad"))
        train_transforms.append(self.hparams.loading["train"].get("crop"))

        test_transforms.append(self.hparams.loading["test"].get("data_loading"))

        # squeeze the data if it is 2D to remove the singleton third dimension
        if not self.threeD:
            train_transforms.append(self.hparams.loading["train"].get("maybe_squeeze"))
            test_transforms.append(self.hparams.loading["test"].get("maybe_squeeze"))

        return train_transforms, train_transforms.copy(), test_transforms

    def setup_transforms(self) -> None:
        """Define the data augmentations used by nnUNet including the data reading using
        `monai.transforms` libraries.

        The only difference with nnUNet framework is the patch creation.
        """
        (
            train_transforms,
            val_transforms,
            test_transforms,
        ) = self.get_train_val_test_loading_transforms()

        other_transforms = []

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(Convert3Dto2Dd(keys=self.hparams.data_keys.all_keys))

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
        if self.hparams.augmentation.get("noise"):
            for name, aug in self.hparams.augmentation.get("noise").items():
                other_transforms.append(aug)

        if self.hparams.augmentation.get("intensity"):
            for name, aug in self.hparams.augmentation.get("intensity").items():
                other_transforms.append(aug)

        if self.hparams.augmentation.get("flip"):
            for name, aug in self.hparams.augmentation.get("flip").items():
                other_transforms.append(aug)

        self.train_transforms = Compose(train_transforms + other_transforms)
        self.val_transforms = Compose(val_transforms)
        self.test_transforms = Compose(test_transforms)

    def unpack_dataset(self):
        """Unpack dataset from .npz to .npy.

        Use Parallel to speed up the unpacking process.
        """
        npz_files = subfiles(self.full_data_dir, True, None, ".npz", True)
        Parallel(n_jobs=max(self.hparams.num_workers, 1))(
            delayed(self.convert_to_npy)(npz_file) for npz_file in npz_files
        )

    @staticmethod
    def convert_to_npy(npz_file: str, key="data"):
        """Decompress .npz file to .npy.

        Args:
            npz_file: Path of the npz file to be decompressed.
        """
        if not os.path.isfile(npz_file[:-3] + "npy"):
            a = np.load(npz_file)[key]
            np.save(npz_file[:-3] + "npy", a)


if __name__ == "__main__":
    from hydra import compose, initialize
    from matplotlib import pyplot as plt
    from omegaconf import OmegaConf

    from ascent import get_ascent_root
    from ascent.utils.visualization import dopplermap, imagesc

    root = get_ascent_root()

    # Example to visualize the data
    with initialize(config_path=str(root / "configs" / "datamodule"), version_base="1.3"):
        cfg = compose(config_name="dealias_2d")

    cfg.data_dir = str(root / "data")
    cfg.in_channels = 2
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

    cmap = dopplermap()

    img = batch["image"][0].array
    label = batch["label"][0].array
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    print(batch["image"][0]._meta["filename_or_obj"])
    plt.figure("image", (18, 6))
    ax = plt.subplot(1, 2, 1)
    imagesc(ax, img[0, :, :].transpose(), "image", cmap, clim=[-1, 1])
    ax = plt.subplot(1, 2, 2)
    imagesc(ax, label[0, :, :].transpose(), "label", cmap)
    print("max of seg: ", np.max(label[0, :, :]))
    plt.show()
