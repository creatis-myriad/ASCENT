import numpy as np
from einops.einops import rearrange
from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.transforms import LoadImage, SpatialCrop, SqueezeDim
from monai.transforms.transform import MapTransform
from monai.transforms.utils import generate_spatial_bounding_box

from covid.preprocessing.preprocessing import resample_image, resample_label
from covid.utils.file_and_folder_operations import load_pickle


class Convert3Dto2Dd(MapTransform):
    """Converts a 3D volume to a stack of 2D images."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: Keys of the corresponding items to be transformed.
            allow_missing_keys: Don't raise exception if key is missing.
        """

        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
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
        """
        Args:
            keys: Keys of the corresponding items to be transformed.
            num_channel: Number of channels of the converted 3D volume.
            allow_missing_keys: Don't raise exception if key is missing.
        """

        super().__init__(keys, allow_missing_keys)
        self.num_channel = num_channel

    def __call__(self, data):
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
        """
        Args:
            keys: Keys of the corresponding items to be transformed.
            dim: Dimension to squeeze.
            allow_missing_keys: Don't raise exception if key is missing.
        """

        super().__init__(keys, allow_missing_keys)
        self.squeeze = SqueezeDim(dim=dim)
        self.dim = dim

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if len(d[key].shape) == 4 and d[key].shape[self.dim] == 1:
                d[key] = self.squeeze(d[key])
        return d


class LoadAndPreprocessd(MapTransform):
    """Load and preprocess data path given in dictionary keys.

    Dictionary must contain the following keys: "image" and "dataset_properties"
    """

    def __init__(self, keys) -> None:
        """
        Args:
            keys: Keys of the corresponding items to be transformed.
        """

        super().__init__(keys)
        self.keys = keys
        self.target_spacing = []
        self.intensity_properties = None
        self.do_resample = None
        self.do_normalize = None
        self.use_nonzero_mask = None

    def calculate_new_shape(self, spacing, shape):
        spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def __call__(self, data):
        # load data
        d = dict(data)
        image = LoadImage(image_only=True)(d["image"])
        image_spacings = image["meta_data"]["pixdim"][1:4].tolist()

        if "label" in self.keys:
            label = d["label"]
            label[label < 0] = 0

        dataset_properties = load_pickle(d["dataset_properties"])
        self.target_spacing = dataset_properties["spacing_after_resampling"]
        self.do_resample = dataset_properties["do_resample"]
        self.do_normalize = dataset_properties["do_normalize"]
        self.use_nonzero_mask = dataset_properties["use_nonzero_mask"]

        image_meta_dict = {}

        d["original_shape"] = np.array(image.shape[1:])
        box_start, box_end = generate_spatial_bounding_box(image)
        image = SpatialCrop(roi_start=box_start, roi_end=box_end)(image)
        d["bbox"] = np.vstack([box_start, box_end])
        d["crop_shape"] = np.array(image.shape[1:])

        original_shape = image.shape[1:]
        # calculate shape
        resample_flag = False
        anisotrophy_flag = False

        image = image.numpy()
        if self.target_spacing != image_spacings:
            # resample
            resample_flag = True
            resample_shape = self.calculate_new_shape(image_spacings, original_shape)
            anisotrophy_flag = self.check_anisotrophy(image_spacings)
            image = resample_image(image, resample_shape, anisotrophy_flag)
            if self.training:
                label = resample_label(label, resample_shape, anisotrophy_flag)

        d["resample_flag"] = resample_flag
        d["anisotrophy_flag"] = anisotrophy_flag
        # clip image for CT dataset
        if self.low != 0 or self.high != 0:
            image = np.clip(image, self.low, self.high)
            image = (image - self.mean) / self.std
        else:
            image = self.normalize_intensity(image.copy())

        d["image"] = image

        if "label" in self.keys:
            d["label"] = label

        return d


class LoadNpyd(MapTransform):
    """Load numpy array from .npz/.npy files.

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
    ) -> None:
        """
        Args:
            keys: Keys of the corresponding items to be transformed.
            test: Set to true to return image meta properties during test.
            allow_missing_keys: Don't raise exception if key is missing.
        """

        super().__init__(keys, allow_missing_keys)
        self.test = test

    def __call__(self, data):
        """
        Args:
            data: Dict to transform.

        Returns:
            d: Dict containing either two {"image":, "label":} or three {"image":, "label":,
                "image_meta_dict":} keys

        Raises:
            ValueError: Error when image or label is not 4D (c, w, h, d)
            NotImplementedError: Error when data contains a path that is not a numpy file or a pkl
                file.
        """

        d = dict(data)
        for key in self.keys:
            if d[key].endswith(".npy"):
                case_all_data = LoadImage(image_only=True, channel_dim=0)(d[key])
                meta = case_all_data._meta
                d.pop(key, None)
                d["image"] = MetaTensor(case_all_data.array[:-1].astype(np.float32), meta=meta)
                d["label"] = MetaTensor(case_all_data.array[-1:].astype(np.uint8), meta=meta)
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


if __name__ == "__main__":
    image_path = [
        "C:/Users/ling/Desktop/Thesis/REPO/CoVID/data/DEALIAS/raw/imagesTr/Dealias_0001_0000.nii.gz",
        "C:/Users/ling/Desktop/Thesis/REPO/CoVID/data/DEALIAS/raw/imagesTr/Dealias_0001_0001.nii.gz",
    ]

    image = LoadImage()([image_path[0]])
    data_path = "C:/Users/ling/Desktop/Thesis/REPO/CoVID/data/CAMUS/preprocessed/data_and_properties/NewCamus_0001.npy"
    # data_path = "C:/Users/ling/Desktop/Thesis/REPO/CoVID/data/CAMUS/cropped/NewCamus_0001.npz"
    prop = "C:/Users/ling/Desktop/Thesis/REPO/CoVID/data/CAMUS/preprocessed/data_and_properties/NewCamus_0001.pkl"
    data = LoadNpyd(["data", "image_meta_dict"], test=True)({"data": data_path})
