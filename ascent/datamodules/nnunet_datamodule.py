import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform
from joblib import Parallel, delayed
from monai.data import CacheDataset, DataLoader, IterableDataset
from monai.transforms import (
    AddChanneld,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandZoomd,
    SpatialPadd,
    SqueezeDimd,
    adaptor,
)
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.model_selection import KFold, train_test_split

from ascent import utils
from ascent.utils.data_loading import get_case_identifiers_from_npz_folders
from ascent.utils.dataset import nnUNet_Iterator
from ascent.utils.file_and_folder_operations import load_pickle, save_pickle, subfiles
from ascent.utils.transforms import (
    AddChannelFirstd,
    Convert2Dto3Dd,
    Convert3Dto2Dd,
    LoadNpyd,
    MayBeSqueezed,
)

log = utils.get_pylogger(__name__)


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
        """Initialize class instance.

        Args:
            data_dir: Path to the data directory.
            dataset_name: Name of dataset to be used.
            fold: Fold to be used for training, validation or test.
            batch_size: Batch size to be used for training and validation.
            patch_size: Patch size to crop the data..
            in_channels: Number of input channels.
            do_dummy_2D_data_aug: Whether to apply 2D transformation on 3D dataset.
            num_workers: Number of subprocesses to use for data loading.
            pin_memory: Whether to pin memory to GPU.
            test_splits: Whether to split data into train/val/test (0.8/0.1/0.1).
            seg_label: Whether the labels are segmentations.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.preprocessed_folder = os.path.join(data_dir, dataset_name, "preprocessed")
        self.full_data_dir = os.path.join(
            data_dir, dataset_name, "preprocessed", "data_and_properties"
        )

        if not len(patch_size) in [2, 3]:
            raise NotImplementedError("Only 2D and 3D patches are supported right now!")

        self.crop_patch_size = patch_size
        self.threeD = len(patch_size) == 3
        self.dim = len(patch_size)
        self.initial_patch_size = []
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

            if test_splits:
                kfold = KFold(n_splits=10, shuffle=True, random_state=12345)
                for i, (train_and_val_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_and_val_keys = np.array(all_keys_sorted)[train_and_val_idx]
                    train_and_val_keys_sorted = np.sort(train_and_val_keys)
                    test_keys = np.array(all_keys_sorted)[test_idx]

                    train_keys, val_keys = train_test_split(
                        train_and_val_keys_sorted, train_size=0.9, random_state=12345
                    )

                    splits.append(OrderedDict())
                    splits[-1]["train"] = train_keys
                    splits[-1]["val"] = val_keys
                    splits[-1]["test"] = test_keys
            else:
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

                if self.hparams.test_splits:
                    test_keys = splits[self.hparams.fold]["test"]
                    log.info(
                        f"This split has {len(train_keys)} training, {len(val_keys)} validation,"
                        f" and {len(test_keys)} testing cases."
                    )
                else:
                    log.info(
                        f"This split has {len(train_keys)} training and {len(val_keys)} validation"
                        f" cases."
                    )
            else:
                log.warning(
                    f"You requested fold {self.hparams.fold} for training but splits "
                    f"contain only {len(splits)} folds. I am now creating a "
                    f"random (but seeded) 80:10:10 split!"
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
                    log.info(
                        f"This random 80:10:10 split has {len(train_keys)} training, {len(val_keys)}"
                        f" validation, and {len(test_keys)} testing cases."
                    )
                else:
                    val_keys = val_and_test_keys
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
        )

    def setup_transforms(self) -> None:
        """Define the data augmentations used by nnUNet including the data reading using
        monai.transforms libraries.

        The only difference with nnUNet framework is the patch creation.
        """

        ignore_axes = None
        if self.threeD:
            # rot_inter_mode = "bilinear"
            # zoom_inter_mode = "trilinear"
            range_x = range_y = range_z = [-30.0 / 180 * np.pi, 30.0 / 180 * np.pi]

            if self.hparams.do_dummy_2D_data_aug:
                # zoom_inter_mode = rot_inter_mode = "bicubic"
                range_x = [-180.0 / 180 * np.pi, 180.0 / 180 * np.pi]
                range_y = range_z = 0.0
                ignore_axes = (0,)

        else:
            # zoom_inter_mode = rot_inter_mode = "bicubic"
            range_x = [-180.0 / 180 * np.pi, 180.0 / 180 * np.pi]
            range_y = range_z = 0.0
            if max(self.hparams.patch_size) / min(self.hparams.patch_size) > 1.5:
                range_x = [-15.0 / 180 * np.pi, 15.0 / 180 * np.pi]

        self.initial_patch_size = self.get_patch_size_for_spatial_transform(
            self.crop_patch_size[: self.dim], range_x, range_y, range_z, (0.7, 1.4)
        )

        if not self.threeD:
            self.initial_patch_size = [*self.initial_patch_size, 1]

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            self.initial_patch_size[-1] = self.crop_patch_size[-1]

        shared_train_val_transforms = [
            LoadNpyd(keys=["data"], seg_label=self.hparams.seg_label),
            EnsureChannelFirstd(keys=["image", "label"]),
        ]

        other_transforms = []
        other_transforms.extend(
            [
                SpatialPadd(
                    keys=["image"],
                    spatial_size=self.initial_patch_size,
                    mode="constant",
                    value=0,
                ),
                EnsureTyped(["label"], data_type="tensor", dtype=torch.int8),
                SpatialPadd(
                    keys=["label"],
                    spatial_size=self.initial_patch_size,
                    mode="constant",
                    value=-1,
                ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.initial_patch_size,
                    pos=0.33,
                    neg=0.67,
                    num_samples=1,
                ),
            ]
        )

        if not self.threeD:
            other_transforms.append(MayBeSqueezed(keys=["image", "label"], dim=-1))

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(Convert3Dto2Dd(keys=["image", "label"]))
            crop_patch_size = self.crop_patch_size[: self.dim - 1]
        else:
            crop_patch_size = self.crop_patch_size[: self.dim]

        other_transforms.extend(
            [
                AddChannelFirstd(keys=["image", "label"]),
                EnsureTyped(["image", "label"], data_type="numpy"),
                adaptor(
                    SpatialTransform(
                        crop_patch_size,
                        patch_center_dist_from_border=None,
                        do_elastic_deform=False,
                        alpha=(0, 0),
                        sigma=(0, 0),
                        do_rotation=True,
                        angle_x=range_x,
                        angle_y=range_y,
                        angle_z=range_z,
                        p_rot_per_axis=1,  # todo experiment with this
                        do_scale=True,
                        scale=(0.7, 1.4),
                        border_mode_data="constant",
                        border_cval_data=0,
                        order_data=3,
                        border_mode_seg="constant",
                        border_cval_seg=-1,
                        order_seg=1,
                        random_crop=False,  # random cropping is part of our dataloaders
                        p_el_per_sample=0,
                        p_scale_per_sample=0.2,
                        p_rot_per_sample=0.2,
                        independent_scale_for_each_axis=False,  # todo experiment with this,
                        data_key="image",
                        label_key="label",
                    ),
                    {"image": "image", "label": "label"},
                ),
                SqueezeDimd(["image", "label"], dim=0),
            ]
        )

        # other_transforms.extend(
        #     [
        #         RandRotated(
        #             keys=["image", "label"],
        #             range_x=range_x,
        #             range_y=range_y,
        #             range_z=range_z,
        #             mode=[rot_inter_mode, "nearest"],
        #             padding_mode="zeros",
        #             prob=0.2,
        #         ),
        #         RandZoomd(
        #             keys=["image", "label"],
        #             min_zoom=0.7,
        #             max_zoom=1.4,
        #             mode=[zoom_inter_mode, "nearest"],
        #             padding_mode="constant",
        #             align_corners=(True, None),
        #             prob=0.2,
        #         ),
        #     ]
        # )

        if self.hparams.do_dummy_2D_data_aug and self.threeD:
            other_transforms.append(Convert2Dto3Dd(keys=["image", "label"]))

        other_transforms.extend(
            [
                AddChannelFirstd(keys=["image", "label"]),
                EnsureTyped(["image", "label"], data_type="numpy"),
                adaptor(
                    GaussianNoiseTransform(p_per_sample=0.1, data_key="image"),
                    {"image": "image", "label": "label"},
                ),
                adaptor(
                    GaussianBlurTransform(
                        (0.5, 1.0),
                        different_sigma_per_channel=True,
                        p_per_sample=0.2,
                        p_per_channel=0.5,
                        data_key="image",
                    ),
                    {"image": "image", "label": "label"},
                ),
                adaptor(
                    BrightnessMultiplicativeTransform(
                        multiplier_range=(0.75, 1.25), p_per_sample=0.15, data_key="image"
                    ),
                    {"image": "image", "label": "label"},
                ),
                adaptor(
                    ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"),
                    {"image": "image", "label": "label"},
                ),
                adaptor(
                    SimulateLowResolutionTransform(
                        zoom_range=(0.5, 1),
                        per_channel=True,
                        p_per_channel=0.5,
                        order_downsample=0,
                        order_upsample=3,
                        p_per_sample=0.25,
                        ignore_axes=ignore_axes,
                        data_key="image",
                    ),
                    {"image": "image", "label": "label"},
                ),
                adaptor(
                    GammaTransform(
                        (0.7, 1.5),
                        True,
                        True,
                        retain_stats=True,
                        p_per_sample=0.1,
                        data_key="image",
                    ),
                    {"image": "image", "label": "label"},
                ),
                adaptor(
                    GammaTransform((0.7, 1.5), False, True, retain_stats=True, data_key="image"),
                    {"image": "image", "label": "label"},
                ),
                adaptor(
                    RemoveLabelTransform(-1, 0, input_key="label", output_key="label"),
                    {"image": "image", "label": "label"},
                ),
                SqueezeDimd(["image", "label"], dim=0),
                EnsureTyped(["image", "label"], data_type="tensor", track_meta=True),
                RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
                RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
            ]
        )

        # other_transforms.extend(
        #     [
        #         SqueezeDimd(["image", "label"], dim=0),
        #         EnsureTyped(["image", "label"], data_type="tensor", track_meta=True),
        #         RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
        #         RandGaussianSmoothd(
        #             keys=["image"],
        #             sigma_x=(0.5, 1.15),
        #             sigma_y=(0.5, 1.15),
        #             prob=0.15,
        #         ),
        #         RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
        #         RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),
        #         RandFlipd(["image", "label"], spatial_axis=[0], prob=0.5),
        #         RandFlipd(["image", "label"], spatial_axis=[1], prob=0.5),
        #     ]
        # )

        if self.threeD:
            other_transforms.append(RandFlipd(["image", "label"], spatial_axis=[2], prob=0.5))

        val_transforms = shared_train_val_transforms.copy()
        val_transforms.extend(
            [
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
        )

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

    @staticmethod
    def get_patch_size_for_spatial_transform(final_patch_size, rot_x, rot_y, rot_z, scale_range):
        if isinstance(rot_x, (tuple, list)):
            rot_x = max(np.abs(rot_x))
        if isinstance(rot_y, (tuple, list)):
            rot_y = max(np.abs(rot_y))
        if isinstance(rot_z, (tuple, list)):
            rot_z = max(np.abs(rot_z))
        rot_x = min(90 / 360 * 2.0 * np.pi, rot_x)
        rot_y = min(90 / 360 * 2.0 * np.pi, rot_y)
        rot_z = min(90 / 360 * 2.0 * np.pi, rot_z)
        from batchgenerators.augmentations.utils import rotate_coords_2d, rotate_coords_3d

        coords = np.array(final_patch_size)
        final_shape = np.copy(coords)
        if len(coords) == 3:
            final_shape = np.max(
                np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0
            )
            final_shape = np.max(
                np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0
            )
            final_shape = np.max(
                np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0
            )
        elif len(coords) == 2:
            final_shape = np.max(
                np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0
            )
        final_shape /= min(scale_range)
        return final_shape.astype(int)

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
    import hydra
    from hydra import compose, initialize_config_dir
    from matplotlib import pyplot as plt
    from omegaconf import OmegaConf

    from ascent import get_ascent_root
    from ascent.utils.visualization import dopplermap, imagesc

    root = get_ascent_root()

    initialize_config_dir(config_dir=str(root / "configs" / "datamodule"), job_name="test")
    cfg = compose(config_name="camus_challenge_2d.yaml")

    cfg.data_dir = str(root.resolve().parent / "data")
    # cfg.patch_size = [128, 128]
    cfg.in_channels = 1
    cfg.patch_size = [640, 1024]
    # cfg.patch_size = [128, 128, 12]
    cfg.batch_size = 1
    cfg.fold = 0
    print(OmegaConf.to_yaml(cfg))
    camus_datamodule = hydra.utils.instantiate(cfg)
    camus_datamodule.prepare_data()
    camus_datamodule.setup(stage=TrainerFn.FITTING)
    train_dl = camus_datamodule.train_dataloader()
    # camus_datamodule.setup(stage=TrainerFn.TESTING)
    # test_dl = camus_datamodule.test_dataloader()

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
    imagesc(ax, img[0, :, :].transpose(), "image", clim=[-1, 1])
    ax = plt.subplot(1, 2, 2)
    imagesc(ax, label[0, :, :].transpose(), "label")
    print("max of seg: ", np.max(label[0, :, :]))
    plt.show()
