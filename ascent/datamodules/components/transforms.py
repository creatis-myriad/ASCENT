import os
from typing import Hashable, Mapping, Optional, Union

import numpy as np
import torch
from einops.einops import rearrange
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import MetaTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms import (
    LoadImage,
    MapTransform,
    RandomizableTransform,
    SpatialCrop,
    SqueezeDim,
)
from monai.transforms.utils import generate_spatial_bounding_box
from monai.utils import convert_to_dst_type, convert_to_tensor
from torch import Tensor

from ascent.preprocessing.preprocessing import resample_image, resample_label
from ascent.utils.file_and_folder_operations import load_pickle


class ArtfclAliasing(RandomizableTransform):
    """Create realistic artificial aliasing by randomly wrapping Doppler velocities.

    Detailed steps:
        1. Pick a random wrapping parameter between the wrap_range.
        2. Dealias the Doppler velocities, Vd, based on the ground truth segmentation if they were
        aliased.
        3. Wrap any Vd > wrap_param.
        4. Normalize Vd.
        5. Create new ground truth segmentation and velocities.
    """

    def __init__(self, prob: float = 0.5, wrap_range: tuple[float, float] = (0.6, 0.9)) -> None:
        """Define the probability and the wrapping range.

        Args:
            prob: Probability to perform the transform.
            wrap_range: Velocity wrapping range, between 0 and 1.
        """
        RandomizableTransform.__init__(self, prob)
        self.wrap_range = wrap_range

    def randomize(self) -> None:
        if self.R.random() < self.prob:
            self._do_transform = True
            self.wrap_param = self.R.uniform(low=self.wrap_range[0], high=self.wrap_range[1])
        else:
            self._do_transform = False

    def alias(self, img: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
        """Create artificial aliasing, the corresponding ground truth segmentation and velocities.

        Args:
            img: Doppler velocity tensor.
            label: Segmentation of aliased pixels, 1: V + 2; 2: V - 2.

        Returns:
            Wrapped velocities, ground truth segmentation, ground truth velocities.

        Raises:
            NotImplementedError: If the input contains more than two channel dimensions.
        """
        if img.shape[0] == 1:
            vel = img.detach().cpu().numpy()
            power = None
        elif img.shape[0] == 2:
            vel = img[:-1].detach().cpu().numpy()
            power = img[-1:].detach().cpu().numpy()
        else:
            raise NotImplementedError(
                "Input more than two channel dimensions is currently not supported!"
            )

        v = vel.copy()
        ori_seg = label.detach().cpu().numpy()

        # check whether the frame contains any aliasing and dealias it if the frame is aliased
        if np.max(ori_seg) > 0:
            v = self.dealias(v, ori_seg)

        gt_v = v.copy()

        # specify a ROI (v > 0.3 and power > 0.4) to create more realistic aliasing
        if power is not None:
            roi = np.logical_and((np.abs(v) > 0.3), (power > 0.4))
        else:
            roi = np.abs(v) > (0.3 * 0.4)

        # create artificial aliasing if ROI is not empty
        if not (np.all(~roi)):
            self.wrap_param = self.wrap_param * np.max(np.abs(v[roi]))
            aliased_v = v.copy()
            aliased_v[roi] = self.wrap(aliased_v[roi], self.wrap_param, True)

            # recompute the ground truth labels based on the artificially aliased frame
            gt_seg = self.recompute_seg(v, aliased_v)
            v = aliased_v

            # clip the artificially aliased velocities
            v[v >= 1] = 0.9999
            v[v <= -1] = -0.9999

            # recompute ground truth velocities
            gt_v = v.copy()
            gt_v[gt_seg == 1] += 2
            gt_v[gt_seg == 2] -= 2

            # delete useless array
            del aliased_v
        else:
            # if ROI is empty, simply use the initial Doppler velocities and segmentation
            v = vel
            gt_seg = ori_seg.astype(np.uint8)

        # delete useless array
        del vel

        # concatenate the velocity with Doppler power if given
        if power is not None:
            v = np.concatenate((v, power))

        # convert the numpy arrays  back to tensors
        v = torch.as_tensor(v)
        gt_seg = torch.as_tensor(gt_seg)
        if isinstance(img, MetaTensor):
            v = convert_to_dst_type(v, dst=img, dtype=torch.float32)[0]
            gt_v = convert_to_dst_type(gt_v, dst=img, dtype=torch.float32)[0]
        if isinstance(label, MetaTensor):
            gt_seg = convert_to_dst_type(gt_seg, dst=label, dtype=torch.uint8)[0]
        return v, gt_seg, gt_v

    @staticmethod
    def recompute_seg(dealiased_vel: np.ndarray, aliased_vel: np.ndarray) -> np.ndarray:
        """Compute new ground truth segmentation for the wrapped Doppler velocities.

        Args:
            dealiased_vel: Dealiased Doppler velocities array.
            aliased_vel: Artificially aliased Doppler velocities array.

        Returns:
            Ground truth segmentation for the artificially aliased Doppler velocities array.
        """
        gt_seg = np.zeros(dealiased_vel.shape)
        diff = np.logical_and(
            (dealiased_vel != aliased_vel), (np.sign(dealiased_vel) != np.sign(aliased_vel))
        )
        plus_two = np.logical_and(diff, np.sign(aliased_vel) == -1)
        minus_two = np.logical_and(diff, np.sign(aliased_vel) == 1)
        if not (np.all(~plus_two)):
            gt_seg[plus_two] = 1
        if not (np.all(~minus_two)):
            gt_seg[minus_two] = 2

        return gt_seg.astype(np.uint8)

    @staticmethod
    def dealias(
        img: Union[np.ndarray, Tensor], seg: Union[np.ndarray, Tensor]
    ) -> Union[np.ndarray, Tensor]:
        """Apply dealiasing based on the ground truth segmentations.

        Args:
            img: Aliased Doppler velocities.
            seg: Ground truth segmentation.

        Returns:
            Dealiased Doppler velocities.
        """
        img[seg == 1] += 2
        img[seg == 2] -= 2

        return img

    @staticmethod
    def wrap(img: np.ndarray, wrap_param: float = 0.65, normalize: bool = False) -> np.ndarray:
        """Wrap any element with its absolute value surpassing the wrapping parameter.

        Args:
            img: Dealiased Doppler velocities array.
            wrap_param: Wrapping parameter.
            normalize: Whether to normalize the wrapped tensor between -1 and 1.

        Returns:
            Wrapped Doppler velocities array.
        """
        img = (img + wrap_param) % (2 * wrap_param) - wrap_param
        if normalize:
            return img / wrap_param
        else:
            return img

    def __call__(self, img: Tensor, seg: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Create random artificial aliasing.

        Args:
            img: Color Doppler image.
            seg: Segmented aliased pixels.

        Returns:
            (Artificially) aliased image, (modified) ground truth segmentation, (modified) ground truth velocity.
        """
        self.randomize()
        if self._do_transform:
            return self.alias(img, seg)
        else:
            return img, seg, self.dealias(img, seg)


class ArtfclAliasingd(RandomizableTransform, MapTransform):
    """Dictionary-based version of ArtfclAliasing."""

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        wrap_range: tuple[float, float] = (0.6, 0.9),
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize class instance.

        Args:
            keys: List containing all the keys in data.
            prob: Probability to perform the transform.
            wrap_range: Doppler velocity wrapping range, between 0 and 1.
            allow_missing_keys: Don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.artfcl_aliasing = ArtfclAliasing(prob=1.0, wrap_range=wrap_range)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "ArtfclAliasingd":
        super().set_random_state(seed, state)
        self.artfcl_aliasing.set_random_state(seed, state)
        return self

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        if "seg" in d.keys():
            d["image"], d["seg"], d["label"] = self.artfcl_aliasing(d["image"], d["seg"])
        else:
            d["image"], d["label"], _ = self.artfcl_aliasing(d["image"], d["label"])

        return d


class Convert3Dto2Dd(MapTransform):
    """Converts a 3D volume to a stack of 2D images."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize class instance.

        Args:
            keys: Keys of the corresponding items to be transformed.
            allow_missing_keys: Don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: dict[str, Tensor]):
        d = dict(data)
        for key in self.keys:
            d[key] = rearrange(d[key], "c w h d -> (c d) w h")
        return d


class Convert2Dto3Dd(MapTransform):
    """Converts a stack of 2D images to a 3D volume."""

    def __init__(
        self,
        keys: KeysCollection,
        num_channel: int,
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize class instance.

        Args:
            keys: Keys of the corresponding items to be transformed.
            num_channel: Number of channels of the converted 3D volume.
            allow_missing_keys: Don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.num_channel = num_channel

    def __call__(self, data: dict[str, Tensor]):
        d = dict(data)
        for key in self.keys:
            d[key] = rearrange(d[key], "(c d) w h -> c w h d", c=self.num_channel)
        return d


class MayBeSqueezed(MapTransform):
    """Squeeze a pseudo 3D image if necessary.

    As pseudo 3D patch shape (w, h, 1) is used in monai's RandCropByPosNegLabeld in nnUNet, the
    output patches are not consistent in terms of shape. It may output patches of shape (c, w, h,
    1) or (c, w, h). Hence, it might be necessary to verify and squeeze the additional dimension
    (last dimension) if required.
    """

    def __init__(
        self,
        keys: KeysCollection,
        dim: int,
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize class instance.

        Args:
            keys: Keys of the corresponding items to be transformed.
            dim: Dimension to squeeze.
            allow_missing_keys: Don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.squeeze = SqueezeDim(dim=dim)
        self.dim = dim

    def __call__(self, data: dict[str, Tensor]):
        d = dict(data)
        for key in self.keys:
            if len(d[key].shape) == 4 and d[key].shape[self.dim] == 1:
                d[key] = self.squeeze(d[key])
        return d


class Preprocessd(MapTransform):
    """Load and preprocess data path given in dictionary keys.

    Dictionary must contain the following key(s): "image" and/or "label".
    """

    def __init__(
        self, keys, target_spacing, intensity_properties, do_resample, do_normalize, modalities
    ) -> None:
        """Initialize class instance.

        Args:
            keys: Keys of the corresponding items to be transformed.
            target_spacing: Target spacing to resample the data.
            intensity_properties: Global properties required to normalize CT data (mean, std, 0.5% )
        """
        super().__init__(keys)
        self.keys = keys
        self.target_spacing = target_spacing
        self.intensity_properties = intensity_properties
        self.do_resample = do_resample
        self.do_normalize = do_normalize
        self.modalities = modalities

    def calculate_new_shape(self, spacing, shape):
        spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return bool(check(spacing) or check(self.target_spacing))

    def __call__(self, data: dict[str, str]):
        # load data
        d = dict(data)
        image = d["image"]

        if "label" in self.keys:
            label = d["label"]
            label[label < 0] = 0

        image_meta_dict = {}
        image_meta_dict["case_identifier"] = os.path.basename(image._meta["filename_or_obj"])[:-12]
        image_meta_dict["original_shape"] = np.array(image.shape[1:])
        image_meta_dict["original_spacing"] = np.array(image._meta["pixdim"][1:4].tolist())

        image_meta_dict["resampling_flag"] = self.do_resample

        box_start, box_end = generate_spatial_bounding_box(image)
        image = SpatialCrop(roi_start=box_start, roi_end=box_end)(image)
        image_meta_dict["crop_bbox"] = np.vstack([box_start, box_end])
        image_meta_dict["shape_after_cropping"] = np.array(image.shape[1:])

        anisotropy_flag = False

        image = image.numpy()
        if image_meta_dict.get("resample_flag"):
            if not np.all(image_meta_dict.get("original_spacing") == self.target_spacing):
                # resample
                resample_shape = self.calculate_new_shape(
                    image_meta_dict.get("original_spacing"), image_meta_dict.get("original_shape")
                )
                anisotropy_flag = self.check_anisotrophy(image_meta_dict.get("original_spacing"))
                image = resample_image(image, resample_shape, anisotropy_flag)
                if "label" in self.keys:
                    label = resample_label(label, resample_shape, anisotropy_flag)

        image_meta_dict["anisotropy_flag"] = anisotropy_flag

        if self.do_normalize:
            for c in range(len(image)):
                scheme = self.modalities[c]
                if scheme == "CT":
                    # clip to lb and ub from train data foreground and use foreground mn and sd from
                    # training data
                    assert (
                        self.intensity_properties is not None
                    ), "ERROR: if there is a CT then we need intensity properties"
                    mean_intensity = self.intensity_properties[c]["mean"]
                    std_intensity = self.intensity_properties[c]["sd"]
                    lower_bound = self.intensity_properties[c]["percentile_00_5"]
                    upper_bound = self.intensity_properties[c]["percentile_99_5"]
                    image[c] = np.clip(image[c], lower_bound, upper_bound)
                    image[c] = (image[c] - mean_intensity) / std_intensity
                elif not scheme == "noNorm":
                    image[c] = (image[c] - image[c].mean()) / (image[c].std() + 1e-8)

        d["image"] = image

        if "label" in self.keys:
            d["label"] = label

        d["image_meta_dict"] = image_meta_dict

        return d


class LoadNpyd(MapTransform):
    """Load numpy array from .npy files.

    nnUNet's preprocessing concatenates image and label files before saving them to .npz files.
    The .npz files will be unpacked to .npy before training.

    data: .npy file containing an array of shape (c, w, h, d)
    data[:-1]: image
    data[-1:]: label
    """

    def __init__(
        self,
        keys: KeysCollection,
        test: bool = False,
        allow_missing_keys: bool = False,
        seg_label: bool = True,
    ) -> None:
        """Initialize class instance.

        Args:
            keys: Keys of the corresponding items to be transformed.
            test: Set to true to return image meta properties during test.
            allow_missing_keys: Don't raise exception if key is missing.
            seg_label: Set to true if the label is segmentation.
        """

        super().__init__(keys, allow_missing_keys)
        self.test = test
        self.seg_label = seg_label

    def __call__(self, data: dict[str, str]):
        """Load .npy image file.

        Args:
            data: Dict to transform.

        Returns:
            d: Dict containing either two {"image":, "label":} or three {"image":, "label":,
                "image_meta_dict":} keys

        Raises:
            ValueError: If the image or label is not 4D (c, w, h, d)
            NotImplementedError: If the data contains a path that is not a .npy file or a pkl file.
        """
        d = dict(data)
        for key in self.keys:
            if d[key].endswith(".npy"):
                case_all_data = LoadImage(image_only=True, channel_dim=0)(d[key])
                meta = case_all_data._meta
                d.pop(key, None)
                d["image"] = MetaTensor(case_all_data.array[:-1].astype(np.float32), meta=meta)
                if self.seg_label:
                    d["label"] = MetaTensor(case_all_data.array[-1:].astype(np.uint8), meta=meta)
                else:
                    d["label"] = MetaTensor(case_all_data.array[-1:].astype(np.float32), meta=meta)
                del case_all_data
                if not len(d["image"].shape) == 4:
                    raise ValueError("Image should be (c, w, h, d)")
                if not len(d["label"].shape) == 4:
                    raise ValueError("Label should be (c, w, h, d)")
            elif d[key].endswith(".pkl"):
                if self.test:
                    image_meta_dict = load_pickle(d["image_meta_dict"])
                    d["image_meta_dict"] = {}
                    d["image_meta_dict"]["case_identifier"] = image_meta_dict["case_identifier"]
                    d["image_meta_dict"]["original_shape"] = image_meta_dict["original_shape"]
                    d["image_meta_dict"]["original_spacing"] = image_meta_dict["original_spacing"]
                    d["image_meta_dict"]["shape_after_cropping"] = image_meta_dict[
                        "shape_after_cropping"
                    ]
                    d["image_meta_dict"]["crop_bbox"] = image_meta_dict["crop_bbox"]
                    d["image_meta_dict"]["resampling_flag"] = image_meta_dict["resampling_flag"]
                    if d["image_meta_dict"]["resampling_flag"]:
                        d["image_meta_dict"]["shape_after_resampling"] = image_meta_dict[
                            "shape_after_resampling"
                        ]
                        d["image_meta_dict"]["anisotropy_flag"] = image_meta_dict[
                            "anisotropy_flag"
                        ]
            else:
                raise NotImplementedError
        return d


class DealiasLoadNpyd(MapTransform):
    """Load numpy array from .npy files. (Specific to deep unfolding for dealiasing)

    nnUNet's preprocessing concatenates image and label files before saving them to .npz files.
    The .npz files will be unpacked to .npy before training.

    data: .npy file containing an array of shape (c, w, h, d)
    data[:-2]: image
    data[-2:-1]: label (Ground truth velocity)
    data[-1:]: label (Ground truth segmentation)
    """

    def __init__(
        self,
        keys: KeysCollection,
        test: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize class instance.

        Args:
            keys: Keys of the corresponding items to be transformed.
            test: Set to true to return image meta properties during test.
            allow_missing_keys: Don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.test = test

    def __call__(self, data):
        """Load .npy image file. (Specific to deep unfolding for dealiasing)
        Args:
            data: Dict to transform.

        Returns:
            d: Dict containing either two {"image":, "label":} or three {"image":, "label":,
                "image_meta_dict":} keys

        Raises:
            ValueError: If the image or label is not 4D (c, w, h, d)
            NotImplementedError: If the data contains a path that is not a .npy file or a pkl file.
        """
        d = dict(data)
        for key in self.keys:
            if d[key].endswith(".npy"):
                case_all_data = LoadImage(image_only=True, channel_dim=0)(d[key])
                meta = case_all_data._meta
                d.pop(key, None)
                d["image"] = MetaTensor(case_all_data.array[:-2].astype(np.float32), meta=meta)
                d["label"] = MetaTensor(case_all_data.array[-2:-1].astype(np.float32), meta=meta)
                d["seg"] = MetaTensor(case_all_data.array[-1:].astype(np.uint8), meta=meta)
                del case_all_data
                if not len(d["image"].shape) == 4:
                    raise ValueError("Image should be (c, w, h, d)")
                if not len(d["label"].shape) == 4:
                    raise ValueError("Label should be (c, w, h, d)")
                if not len(d["seg"].shape) == 4:
                    raise ValueError("Segmentation should be (c, w, h, d)")
            elif d[key].endswith(".pkl"):
                if self.test:
                    image_meta_dict = load_pickle(d["image_meta_dict"])
                    d["image_meta_dict"] = {}
                    d["image_meta_dict"]["case_identifier"] = image_meta_dict["case_identifier"]
                    d["image_meta_dict"]["original_shape"] = image_meta_dict["original_shape"]
                    d["image_meta_dict"]["original_spacing"] = image_meta_dict["original_spacing"]
                    d["image_meta_dict"]["shape_after_cropping"] = image_meta_dict[
                        "shape_after_cropping"
                    ]
                    d["image_meta_dict"]["crop_bbox"] = image_meta_dict["crop_bbox"]
                    d["image_meta_dict"]["resampling_flag"] = image_meta_dict["resampling_flag"]
                    if d["image_meta_dict"]["resampling_flag"]:
                        d["image_meta_dict"]["shape_after_resampling"] = image_meta_dict[
                            "shape_after_resampling"
                        ]
                        d["image_meta_dict"]["anisotropy_flag"] = image_meta_dict[
                            "anisotropy_flag"
                        ]
            else:
                raise NotImplementedError
        return d


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from ascent.utils.visualization import dopplermap, imagesc

    # image_path = [
    #     "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/data/DEALIAS/raw/imagesTr/Dealias_0001_0000.nii.gz",
    #     "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/data/DEALIAS/raw/imagesTr/Dealias_0001_0001.nii.gz",
    # ]
    # load = Preprocessd(
    #     "images", np.array([0.5, 0.5, 1]), None, True, True, {0: "noNorm", 1: "noNorm"}
    # )
    # batch = load({"image": LoadImage(image_path)})
    data_path = "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/data/DEALIAS/preprocessed/data_and_properties/Dealias_0035.npy"
    # data_path = "C:/Users/ling/Desktop/Thesis/REPO/ascent/data/CAMUS/cropped/NewCamus_0001.npz"
    # prop = "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/data/CAMUS/preprocessed/data_and_properties/NewCamus_0001.pkl"
    data = LoadNpyd(["data"], test=False, seg_label=False)({"data": data_path})
    aliased, gt_seg, gt_v = ArtfclAliasing(prob=1)(data["image"][..., 61], data["label"][..., 61])

    ori_vel = data["image"][0, :, :, 61].array.transpose()
    ori_gt = data["label"][0, :, :, 61].array.transpose()
    aliased_vel = aliased[0, ...].array.transpose()
    new_gt = gt_seg[0, ...].array.transpose()
    # gt_vel = batch["label"][0, :, :, 52].array.transpose()
    plt.figure("image", (18, 6))
    ax = plt.subplot(1, 5, 1)
    imagesc(ax, ori_vel, "ori", dopplermap(), [-1, 1])
    ax = plt.subplot(1, 5, 2)
    imagesc(ax, ori_gt, "ori_gt", dopplermap(), [0, 2])
    ax = plt.subplot(1, 5, 3)
    imagesc(ax, aliased_vel, "aliased", dopplermap(), [-1, 1])
    ax = plt.subplot(1, 5, 4)
    imagesc(ax, new_gt, "new_gt", dopplermap(), [0, 2])
    # ax = plt.subplot(1, 5, 5)
    # imagesc(ax, gt_vel, "gt_vel", dopplermap(), list(np.array([-1, 1]) * np.max(np.abs(gt_vel))))
    plt.show()
