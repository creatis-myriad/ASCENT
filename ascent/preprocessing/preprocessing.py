import itertools
import os
import pickle  # nosec B403
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Union

import numpy as np
from einops.einops import rearrange
from joblib import Parallel, delayed
from monai.data import MetaTensor
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, SpatialCropd
from monai.transforms.utils import generate_spatial_bounding_box
from skimage.transform import resize
from torch import Tensor

from ascent import utils
from ascent.utils.file_and_folder_operations import load_json, subfiles

log = utils.get_pylogger(__name__)


def resample_image(
    image: np.ndarray, new_shape: Union[list, tuple], anisotropy_flag: bool
) -> np.ndarray:
    """Resample an image.

    Args:
        image: Image numpy array to be resampled.
        new_shape: Shape after resampling.
        anisotropy_flag: Whether the image is anisotropic.

    Returns:
        Resampled image.
    """

    shape = np.array(image[0].shape)
    if not np.all(shape == np.array(new_shape)):
        resized_channels = []
        if anisotropy_flag:
            print("Anisotropic image, using separate z resampling")
            for image_c in image:
                resized_slices = []
                for i in range(image_c.shape[-1]):
                    image_c_2d_slice = image_c[:, :, i]
                    image_c_2d_slice = resize(
                        image_c_2d_slice,
                        new_shape[:-1],
                        order=3,
                        mode="edge",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    resized_slices.append(image_c_2d_slice)
                resized = np.stack(resized_slices, axis=-1)
                resized = resize(
                    resized,
                    new_shape,
                    order=0,
                    mode="constant",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_channels.append(resized)
        else:
            print("Not using separate z resampling")
            for image_c in image:
                resized = resize(
                    image_c,
                    new_shape,
                    order=3,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_channels.append(resized)
        reshaped = np.stack(resized_channels, axis=0)
        return reshaped
    else:
        print("No resampling necessary")
        return image


def resample_label(
    label: np.ndarray, new_shape: Union[list, tuple], anisotropy_flag: bool
) -> np.ndarray:
    """Resample a label.

    Args:
        label: Label numpy array to be resampled.
        new_shape: Shape after resampling.
        anisotropy_flag: Whether the label is anisotropic.

    Returns:
        Resampled label.
    """

    shape = np.array(label[0].shape)
    if not np.all(shape == np.array(new_shape)):
        reshaped = np.zeros(new_shape, dtype=np.uint8)
        n_class = np.max(label)
        if anisotropy_flag:
            print("Anisotropic label, using separate z resampling")
            shape_2d = new_shape[:-1]
            depth = label.shape[-1]
            reshaped_2d = np.zeros((*shape_2d, depth), dtype=np.uint8)

            for class_ in range(1, int(n_class) + 1):
                for depth_ in range(depth):
                    mask = label[0, :, :, depth_] == class_
                    resized_2d = resize(
                        mask.astype(float),
                        shape_2d,
                        order=1,
                        mode="edge",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    reshaped_2d[:, :, depth_][resized_2d >= 0.5] = class_
            for class_ in range(1, int(n_class) + 1):
                mask = reshaped_2d == class_
                resized = resize(
                    mask.astype(float),
                    new_shape,
                    order=0,
                    mode="constant",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped[resized >= 0.5] = class_
        else:
            print("Not using separate z resampling")
            for class_ in range(1, int(n_class) + 1):
                mask = label[0] == class_
                resized = resize(
                    mask.astype(float),
                    new_shape,
                    order=1,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped[resized >= 0.5] = class_

        reshaped = np.expand_dims(reshaped, 0)
        return reshaped
    else:
        print("No resampling necessary")
        return label


class SegPreprocessor:
    """Preprocessor class that takes nnUNet's preprocessing method (https://github.com/MIC-
    DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py) for reference.

    Crop, resample and normalize data that is stored in ~/data/DATASET_NAME/raw. Cropped data is
    stored in ~/data/DATASET_NAME/cropped while preprocessed (normalized and resampled) data is
    stored in ~/data/DATASET_NAME/preprocessed/data_and_properties. Raw dataset should follow
    strictly the nnUNet raw data format. Refer to https://github.com/MIC-
    DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md for more information.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        do_resample: bool = True,
        do_normalize: bool = True,
        num_workers: int = 12,
        overwrite_existing: bool = False,
    ) -> None:
        """
        Args:
            dataset_path: Path to the dataset.
            do_resample: Whether to resample data.
            do_normalize: Whether to normalize data.
            num_workers: Number of workers to run the preprocessing.
            overwrite_existing: Whether to overwrite the preprocessed data if it exists.
        """

        self.dataset_path = os.path.join(dataset_path, "raw")
        self.cropped_folder = os.path.join(dataset_path, "cropped")
        self.preprocessed_folder = os.path.join(dataset_path, "preprocessed")
        self.preprocessed_npz_folder = os.path.join(
            dataset_path, "preprocessed", "data_and_properties"
        )
        self.do_resample = do_resample
        self.do_normalize = do_normalize
        self.num_workers = num_workers
        self.overwrite_existing = overwrite_existing
        self.target_spacing = None
        self.intensity_properties = OrderedDict()
        self.all_size_reductions = []

    def _create_datalist(self) -> tuple[list[dict[str, str]], list[str], dict[int, str]]:
        """Read the dataset.json in 'raw' directory and extract useful information.

        Returns:
            - List of dictionaries containing the paths to the image and its label.
            - Image keys in datalist.
            - Dictionary containing the modalities indicated in dataset.json.
        """

        datalist = []

        json_file = os.path.join(self.dataset_path, "dataset.json")

        d = load_json(json_file)
        training_files = d["training"]
        num_modalities = len(d["modality"].keys())
        for tr in training_files:
            cur_pat = OrderedDict()
            image_paths = []
            for mod in range(num_modalities):
                image_paths.append(
                    os.path.join(
                        self.dataset_path,
                        "imagesTr",
                        tr["image"].split("/")[-1][:-7] + "_%04.0d.nii.gz" % mod,
                    )
                )
            cur_pat["image"] = image_paths
            cur_pat["label"] = os.path.join(
                self.dataset_path, "labelsTr", tr["label"].split("/")[-1]
            )
            datalist.append(cur_pat)
            modalities = {int(i): d["modality"][str(i)] for i in d["modality"].keys()}

        return datalist, modalities

    def _get_target_spacing(
        self,
        datalist: list[dict[str, Union[Path, str]]],
        transforms: Callable[[dict[str, str]], dict[str, Union[MetaTensor, Tensor, str]]],
    ) -> list[float]:
        """Calculate the target spacing.

        The calculation of the target spacing involves:
            - Parallel runs to collect all the spacings.
            - Calculation of the median image spacing.
            - Handling of the case where the median image spacing is anisotropic.

        Args:
            datalist: List of paths to all the images and labels.
            transforms: Compose of sequences of monai's transformations to read the path provided
                in datalist and transform the data.

        Returns:
            Target spacing.
        """

        spacings = self._run_parallel_from_raw(self._get_spacing, datalist, transforms)
        spacings = np.array(spacings)
        target_spacing = np.median(spacings, axis=0)
        if max(target_spacing) / min(target_spacing) >= 3:
            lowres_axis = np.argmin(target_spacing)
            target_spacing[lowres_axis] = np.percentile(spacings[:, lowres_axis], 10)
        return list(target_spacing)

    def _get_spacing(
        self,
        data: dict[str, Union[Path, str]],
        transforms: Callable[[dict[str, str]], dict[str, Union[MetaTensor, Tensor, str]]],
        **kwargs,
    ) -> list[float]:
        """Get the image spacing from image in the data dictionary.

        Args:
            data: Paths to a pair of image and label.
            transforms: Compose of sequences of monai's transformations to read the path provided
                in datalist and transform the data.

        Returns:
            Image spacing.
        """

        data = transforms(data)
        return data["image"].meta["pixdim"][1:4].tolist()

    def _collect_intensities(
        self,
        datalist: list[dict[str, Union[Path, str]]],
        transforms: Callable[[dict[str, str]], dict[str, Union[MetaTensor, Tensor, str]]],
    ) -> dict[dict[str, float]]:
        """Collect the intensity properties of the whole dataset.

        Compute the min, max, mean, std, 0.5th percentile, and 99.5th percentile after gathering
        all the intensities of all data. Useful for CT images.

        Args:
            datalist: List of paths to all the images and labels.
            transforms: Compose of sequences of monai's transformations to read the path provided
                in datalist and transform the data.

        Returns:
            Intensity properties dictionary.
        """

        intensity_properties = OrderedDict()
        for i in range(len(self.modalities)):
            mod = {"modality": i}
            intensity_properties[i] = OrderedDict()
            intensities = self._run_parallel_from_raw(
                self._get_intensities, datalist, transforms, **mod
            )
            intensities = list(itertools.chain(*intensities))
            intensity_properties[i]["min"], intensity_properties[i]["max"] = np.min(
                intensities
            ), np.max(intensities)
            (
                intensity_properties[i]["percentile_00_5"],
                intensity_properties[i]["percentile_99_5"],
            ) = np.percentile(intensities, [0.5, 99.5])
            intensity_properties[i]["mean"], intensity_properties[i]["std"] = np.mean(
                intensities
            ), np.std(intensities)
        return intensity_properties

    def _get_intensities(
        self,
        data: dict[str, Union[Path, str]],
        transforms: Callable[[dict[str, str]], dict[str, Union[MetaTensor, Tensor, str]]],
        **kwargs,
    ) -> list:
        """Collect the intensities of a data.

        Gather the intensities of all the foreground pixels in an image.

        Args:
            data: Paths to a pair of image and label.
            transforms: Compose of sequences of monai's transformations to read the path provided
                in datalist and transform the data.
            kwargs: Indication of the desired modality for intensity collection.

        Returns:
            Image intensities of foreground pixels.
        """

        data = transforms(data)
        image = data["image"].astype(np.float32)
        label = data["label"].astype(np.uint8)
        foreground_idx = np.where(label[0] > 0)
        intensities = image[kwargs["modality"]][foreground_idx].tolist()
        return intensities

    def _crop_from_list_of_files(
        self,
        datalist: list[dict[str, Union[Path, str]]],
        transforms: Callable[[dict[str, str]], dict[str, Union[MetaTensor, Tensor, str]]],
    ) -> None:
        """Crop the dataset to non-zero region and save the cropped data.

        Some datasets like BraTS contains images that are surrounded by a lot of background pixels.
        Those images can have significant size reduction after being cropped to non-zero region.

        Args:
            datalist: List of paths to all the images and labels.
            transforms: Compose of sequences of monai's transformations to read the path provided
                in datalist and transform the data.
        """

        os.makedirs(self.cropped_folder, exist_ok=True)
        self._run_parallel_from_raw(self._crop, datalist, transforms)

    def _crop(
        self,
        data: dict[str, Union[Path, str]],
        transforms: Callable[[dict[str, str]], dict[str, Union[MetaTensor, Tensor, str]]],
        **kwargs,
    ) -> None:
        """Crop an image-label pair to non-zero region and save it.

        Args:
            data: Paths to a pair of image and label.
            transforms: Compose of sequences of monai's transformations to read the path provided
                in datalist and transform the data.
        """

        list_of_data_files = data["image"]
        case_identifier = os.path.basename(list_of_data_files[0]).split(".nii.gz")[0][:-5]
        if self.overwrite_existing or (
            not os.path.isfile(os.path.join(self.cropped_folder, "%s.npz" % case_identifier))
            or not os.path.isfile(os.path.join(self.cropped_folder, "%s.pkl" % case_identifier))
        ):
            properties = OrderedDict()
            data = transforms(data)
            properties["case_identifier"] = case_identifier
            properties["list_of_data_files"] = list_of_data_files
            properties["original_shape"] = np.array(data["image"].shape[1:])
            properties["original_spacing"] = np.array(data["image"].meta["pixdim"][1:4].tolist())
            box_start, box_end = generate_spatial_bounding_box(data["image"])
            properties["crop_bbox"] = np.vstack([box_start, box_end])
            print("\nCropping %s..." % properties["case_identifier"])
            data = SpatialCropd(
                keys=["image", "label"],
                roi_start=box_start,
                roi_end=box_end,
            )(data)
            properties["shape_after_cropping"] = np.array(data["image"].shape[1:])
            properties["cropping_size_reduction"] = np.prod(
                properties["shape_after_cropping"]
            ) / np.prod(properties["original_shape"])
            print(
                "before crop:",
                tuple([data["image"].shape[0], *properties["original_shape"].tolist()]),
                "after crop:",
                tuple([data["image"].shape[0], *properties["shape_after_cropping"].tolist()]),
                "spacing:",
                properties["original_spacing"],
                "\n",
            )

            cropped_filename = os.path.join(
                self.cropped_folder, "%s.npz" % properties["case_identifier"]
            )
            properties_name = os.path.join(
                self.cropped_folder, "%s.pkl" % properties["case_identifier"]
            )
            all_data = np.vstack([data["image"].array, data["label"].array])
            print("\nSaving to", cropped_filename)
            np.savez_compressed(cropped_filename, data=all_data)
            with open(properties_name, "wb") as f:
                pickle.dump(properties, f)  # nosec B301

    @staticmethod
    def get_case_identifier_from_raw_data(data: dict[str, Union[Path, str]]) -> str:
        """Extract the case identifier for files in raw dataset.

        Example:
            ~/abc/def/ghi/BraTS_0001_0000.nii.gz -> BraTS_0001

        Args:
            data: Paths to a pair of image and label.

        Returns:
            Case identifier.
        """

        return os.path.basename(data["image"].meta["filename_or_obj"]).split(".nii.gz")[0][:-5]

    @staticmethod
    def get_case_identifier_from_npz(case: Union[Path, str]) -> str:
        """Extract the case identifier for files in cropped dataset.

        Example:
            ~/abc/def/ghi/BraTS_0001_0000.npz -> BraTS_0001

        Args:
            data: Paths to a pair of image and label.

        Returns:
            Case identifier.
        """

        case_identifier = os.path.basename(case)[:-4]
        return case_identifier

    def _get_all_classes(self):
        dataset_json = load_json(os.path.join(self.dataset_path, "dataset.json"))
        classes = dataset_json["labels"]
        all_classes = [int(i) for i in classes.keys() if int(i) > 0]
        return all_classes

    def _check_anisotropy(
        self,
        spacing: list[
            float,
        ],
    ) -> bool:
        """Check whether the spacing is anisotropic.

        The anisotropy threshold of 3 is used as indicated in nnUNet.

        Args:
            spacing: New calculated spacing for data resampling.

        Returns:
            Anisotropy flag.
        """

        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def _calculate_new_shape(
        self,
        spacing: list[
            float,
        ],
        shape: Union[list, tuple],
    ) -> list:
        """Calculate the new shape after resampling.

        Args:
            spacing: New calculated spacing for data resampling.
            shape: Original image shape.

        Returns:
            New shape after resampling.
        """

        spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def _get_all_size_reductions(
        self,
        list_of_cropped_npz_files: list[
            Union[str, Path],
        ],
    ) -> list[float]:
        """Gather all the size reductions after cropping the dataset.

        Args:
            list_of_cropped_npz_files: Cropped npz files.

        Returns:
            List containing all the size reductions after cropping.
        """

        properties = self._run_parallel_from_cropped(
            self._get_size_reduction, list_of_cropped_npz_files
        )
        return properties

    def _get_size_reduction(self, case_identifier: str) -> float:
        """Extract the size reduction from the property .pkl file.

        Args:
            case_identifier: Case identifier of a data.

        Returns:
            Size reduction after cropping.
        """

        properties = self._load_properties_of_cropped(case_identifier)
        return properties["cropping_size_reduction"]

    def _load_cropped(self, case_identifier) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load image, label and properties of a cropped data.

        Args:
            case_identifier: Case identifier of a data.

        Returns:
            - Image numpy array
            - Label numpy array
            - Data properties
        """

        all_data = np.load(os.path.join(self.cropped_folder, "%s.npz" % case_identifier))["data"]
        data = all_data[:-1].astype(np.float32)
        seg = all_data[-1:]
        properties = self._load_properties_of_cropped(case_identifier)
        return data, seg, properties

    def _load_properties_of_cropped(self, case_identifier: str) -> dict:
        """Load the properties of a cropped data.

        Args:
            case_identifier: Case identifier of a data.

        Returns:
            Data properties.
        """

        with open(os.path.join(self.cropped_folder, "%s.pkl" % case_identifier), "rb") as f:
            properties = pickle.load(f)  # nosec B301
        return properties

    def _determine_whether_to_use_mask_for_norm(
        self,
    ) -> dict[bool]:
        """Determine whether to use non-zero mask for data normalization.

        Use non-zero mask for normalization when the image is not a CT and when the cropping to
        non-zero reduces the median image size to more than 25%.

        Returns:
            Boolean flags to indicate the use of non-zero mask for each modality.
        """

        # only use the nonzero mask for normalization of the cropping based on it resulted in a
        # decrease in image size (this is an indication that the data is something like brats/isles
        # and then we want to normalize in the brain region only)
        num_modalities = len(list(self.modalities.keys()))
        use_nonzero_mask_for_norm = OrderedDict()

        for i in range(num_modalities):
            if "CT" in self.modalities[i]:
                use_nonzero_mask_for_norm[i] = False
            else:

                if np.median(self.all_size_reductions) < 3 / 4.0:
                    print("Using nonzero mask for normalization")
                    use_nonzero_mask_for_norm[i] = True
                else:
                    print("Not using nonzero mask for normalization")
                    use_nonzero_mask_for_norm[i] = False

        use_nonzero_mask_for_normalization = use_nonzero_mask_for_norm
        return use_nonzero_mask_for_normalization

    def _preprocess(self, list_of_cropped_npz_files: list[Union[Path, str]]) -> None:
        """Preprocess (resample and normalize) the dataset.

        Args:
            list_of_cropped_npz_files: Cropped npz files.
        """

        os.makedirs(self.preprocessed_npz_folder, exist_ok=True)

        self._run_parallel_from_cropped(self._resample_and_normalize, list_of_cropped_npz_files)

    def _resample_and_normalize(self, case_identifier: str) -> None:
        """Resample and normalize a data.

        Resample, normalize and save the data to the preprocessed folder
        (~/data/DATASET_NAME/preprocessed/data_and_properties).

        Args:
            case_identifier: Case identifier of a data.
        """

        data, seg, properties = self._load_cropped(case_identifier)
        if not self.do_resample:
            print("\n", "Skip resampling...")
            properties["resampling_flag"] = False
            properties["shape_after_resampling"] = np.array(data[0].shape)
            properties["spacing_after_resampling"] = properties["original_spacing"]
        else:
            properties["resampling_flag"] = True

            before = {"spacing": properties["original_spacing"], "data.shape": data.shape}

            anisotropy_flag = bool(self._check_anisotropy(properties["original_spacing"]))
            new_shape = self._calculate_new_shape(
                properties["original_spacing"], properties["shape_after_cropping"]
            )
            data = resample_image(data, new_shape, anisotropy_flag)
            seg = resample_label(seg, new_shape, anisotropy_flag)
            properties["anisotropy_flag"] = anisotropy_flag
            properties["shape_after_resampling"] = np.array(data[0].shape)
            properties["spacing_after_resampling"] = np.array(self.target_spacing)

            after = {
                "spacing": properties["spacing_after_resampling"],
                "data.shape (data is resampled)": data.shape,
            }

            print("before:", before, "\nafter: ", after, "\n")

        if not self.do_normalize:
            print("\nSkip normalization...")
            properties["normalization_flag"] = False
        else:
            properties["normalization_flag"] = True
            properties["use_nonzero_mask_for_norm"] = self.use_nonzero_mask
            data, seg = self._normalize(data, seg)

        all_data = np.vstack((data, seg)).astype(np.float32)
        print(
            "Saving: ",
            os.path.join(
                self.preprocessed_folder, "data_and_properties", "%s.npz" % case_identifier
            ),
            "\n",
        )
        np.savez_compressed(
            os.path.join(
                self.preprocessed_folder, "data_and_properties", "%s.npz" % case_identifier
            ),
            data=all_data.astype(np.float32),
        )
        with open(
            os.path.join(
                self.preprocessed_folder, "data_and_properties", "%s.pkl" % case_identifier
            ),
            "wb",
        ) as f:
            pickle.dump(properties, f)  # nosec B301

    def _normalize(self, data: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalize data.

        Args:
            data: Image numpy array to be normalized.
            seg: Label numpy array. Serves for non-zero mask normalization.

        Returns:
            - Normalized image.
            - Label.
        """

        assert len(self.use_nonzero_mask) == len(data)
        print("Normalization...")
        for c in range(len(data)):
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
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if self.use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif not scheme == "noNorm":
                if self.use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (
                    data[c][mask].std() + 1e-8
                )
                data[c][mask == 0] = 0
        print("Normalization done")
        return data, seg

    def _get_all_shapes_after_resampling(
        self, list_of_preprocessed_npz_files: list[Union[Path, str]]
    ) -> list[np.array]:
        """Get all shapes after resampling.

        Args:
            list_of_preprocessed_npz_files: List of preprocessed npz data.

        Returns:
            List of all shapes after resampling.
        """
        shapes = self._run_parallel_from_cropped(
            self._get_shape_after_resampling, list_of_preprocessed_npz_files
        )
        return shapes

    def _get_shape_after_resampling(self, case_identifier: str) -> dict:
        """Get the shape after resampling of a preprocessed data.

        Args:
            case_identifier: Case identifier of a data.

        Returns:
            Data properties.
        """

        with open(
            os.path.join(self.preprocessed_npz_folder, "%s.pkl" % case_identifier), "rb"
        ) as f:
            properties = pickle.load(f)  # nosec B301
        return properties["shape_after_resampling"]

    def _save_dataset_properties(self, list_of_preprocessed_npz_files) -> None:
        """Save useful dataset properties that will be used during inference and for experiment
        planning."""

        prop = OrderedDict()
        prop["do_resample"] = self.do_resample
        prop["spacing_after_resampling"] = self.target_spacing
        prop["do_normalize"] = self.do_normalize
        prop["modalities"] = self.modalities
        prop["use_nonzero_mask"] = self.use_nonzero_mask
        prop["intensity_properties"] = self.intensity_properties
        all_cases = [
            self.get_case_identifier_from_npz(case) for case in list_of_preprocessed_npz_files
        ]
        prop["all_cases"] = all_cases
        prop["all_shapes_after_resampling"] = self._get_all_shapes_after_resampling(
            list_of_preprocessed_npz_files
        )
        prop["all_classes"] = self._get_all_classes()

        with open(os.path.join(self.preprocessed_folder, "dataset_properties.pkl"), "wb") as f:
            pickle.dump(prop, f)  # nosec B301

    def _run_parallel_from_raw(
        self,
        func: Callable,
        datalist: list[dict[str, Union[Path, str]]],
        transforms: Callable[[dict[str, str]], dict[str, Union[MetaTensor, Tensor, str]]],
        **kwargs,
    ):
        """Parallel runs to perform operations on raw data.

        Args:
            func: Function to be called.
            datalist: List of paths to all the images and labels.
            transforms: Compose of sequences of monai's transformations to read the path provided
                in datalist and transform the data.
            kwargs: Indication of the desired modality for intensity collection. To be passed to
                _get_intensities().
        """

        return Parallel(n_jobs=self.num_workers)(
            delayed(func)(data, transforms, **kwargs) for data in datalist
        )

    def _run_parallel_from_cropped(
        self, func: Callable, list_of_cropped_npz_files: list[Union[Path, str]]
    ):
        """Parallel runs to perform operations on cropped data.

        Args:
            func: Function to be called.
            list_of_cropped_npz_files: Cropped .npz files.
        """

        return Parallel(n_jobs=self.num_workers)(
            delayed(func)(self.get_case_identifier_from_npz(npz_file))
            for npz_file in list_of_cropped_npz_files
        )

    def run(self) -> None:
        """Perform the cropping, resampling, normalization and saving of the dataset."""

        # get all training data
        datalist, self.modalities = self._create_datalist()

        load_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
        ]
        transforms = Compose(load_transforms)

        log.info("Initializing to run preprocessing...")
        log.info(f"Cropped folder: {self.cropped_folder}")
        log.info(f"Preprocessed folder: {self.preprocessed_folder}")
        # get target spacing
        self.target_spacing = self._get_target_spacing(datalist, transforms)
        log.info("Target spacing: {np.array(self.target_spacing)}.")

        # get intensity properties if input contains CT data
        if "CT" in self.modalities.values():
            log.info("CT input, calculating intensity propoerties...")
            self.intensity_properties = self._collect_intensities(datalist, transforms)
        else:
            self.intensity_properties = None
            log.info("Non CT input, skipping the calculation of intensity properties...")

        # crop to non zero
        self._crop_from_list_of_files(datalist, transforms)
        list_of_cropped_npz_files = subfiles(self.cropped_folder, True, None, ".npz", True)

        # get all size reductions
        self.all_size_reductions = self._get_all_size_reductions(list_of_cropped_npz_files)

        # determine whether to use non zero mask for normalization
        self.use_nonzero_mask = self._determine_whether_to_use_mask_for_norm()

        # resample and normalize
        self._preprocess(list_of_cropped_npz_files)

        # save dataset properties
        list_of_preprocessed_npz_files = subfiles(
            self.preprocessed_npz_folder, True, None, ".npz", True
        )
        self._save_dataset_properties(list_of_preprocessed_npz_files)


class RegPreprocessor(SegPreprocessor):
    """Preprocessor for regression."""

    def _get_intensities(
        self,
        data: dict[str, Union[Path, str]],
        transforms: Callable[[dict[str, str]], dict[str, Union[MetaTensor, Tensor, str]]],
        **kwargs,
    ) -> list:
        """Collect the intensities of a data.

        Gather the intensities of all the foreground pixels in an image.

        Args:
            data: Paths to a pair of image and label.
            transforms: Compose of sequences of monai's transformations to read the path provided
                in datalist and transform the data.
            kwargs: Indication of the desired modality for intensity collection.

        Returns:
            Image intensities of foreground pixels.
        """

        # simply return all intensities of the image in a list
        data = transforms(data)
        image = data["image"].astype(np.float32)
        intensities = rearrange(image, "b c h w d -> (b c h w d)").tolist()
        return intensities

    def _crop(
        self,
        data: dict[str, Union[Path, str]],
        transforms: Callable[[dict[str, str]], dict[str, Union[MetaTensor, Tensor, str]]],
        **kwargs,
    ) -> None:
        """Crop an image-label pair to non-zero region and save it.

        Args:
            data: Paths to a pair of image and label.
            transforms: Compose of sequences of monai's transformations to read the path provided
                in datalist and transform the data.
        """

        # skip cropping but save some useful information
        list_of_data_files = data["image"]
        case_identifier = os.path.basename(list_of_data_files[0]).split(".nii.gz")[0][:-5]
        if self.overwrite_existing or (
            not os.path.isfile(os.path.join(self.cropped_folder, "%s.npz" % case_identifier))
            or not os.path.isfile(os.path.join(self.cropped_folder, "%s.pkl" % case_identifier))
        ):
            properties = OrderedDict()
            data = transforms(data)
            properties["case_identifier"] = case_identifier
            properties["list_of_data_files"] = list_of_data_files
            properties["original_shape"] = np.array(data["image"].shape[1:])
            properties["original_spacing"] = np.array(data["image"].meta["pixdim"][1:4].tolist())

            properties["crop_bbox"] = []
            print("\nSkip cropping %s..." % properties["case_identifier"])
            properties["shape_after_cropping"] = properties["original_shape"]
            properties["cropping_size_reduction"] = 1

            cropped_filename = os.path.join(
                self.cropped_folder, "%s.npz" % properties["case_identifier"]
            )
            properties_name = os.path.join(
                self.cropped_folder, "%s.pkl" % properties["case_identifier"]
            )
            all_data = np.vstack([data["image"].array, data["label"].array])
            print("\nSaving to", cropped_filename)
            np.savez_compressed(cropped_filename, data=all_data)
            with open(properties_name, "wb") as f:
                pickle.dump(properties, f)  # nosec B301

    def _determine_whether_to_use_mask_for_norm(
        self,
    ) -> dict[bool]:
        """Determine whether to use non-zero mask for data normalization.

        Use non-zero mask for normalization when the image is not a CT and when the cropping to
        non-zero reduces the median image size to more than 25%.

        Returns:
            Boolean flags to indicate the use of non-zero mask for each modality.
        """

        # disable non zero mask normalization as we don't have the segmentation
        num_modalities = len(list(self.modalities.keys()))
        use_nonzero_mask_for_norm = OrderedDict()

        for i in range(num_modalities):
            use_nonzero_mask_for_norm[i] = False

        return use_nonzero_mask_for_norm

    def _save_dataset_properties(self, list_of_preprocessed_npz_files) -> None:
        """Save useful dataset properties that will be used during inference and for experiment
        planning."""

        prop = OrderedDict()
        prop["do_resample"] = self.do_resample
        prop["spacing_after_resampling"] = self.target_spacing
        prop["do_normalize"] = self.do_normalize
        prop["modalities"] = self.modalities
        prop["use_nonzero_mask"] = self.use_nonzero_mask
        prop["intensity_properties"] = self.intensity_properties
        all_cases = [
            self.get_case_identifier_from_npz(case) for case in list_of_preprocessed_npz_files
        ]
        prop["all_cases"] = all_cases
        prop["all_shapes_after_resampling"] = self._get_all_shapes_after_resampling(
            list_of_preprocessed_npz_files
        )
        prop["all_classes"] = []

        with open(os.path.join(self.preprocessed_folder, "dataset_properties.pkl"), "wb") as f:
            pickle.dump(prop, f)  # nosec B301

    def _resample_and_normalize(self, case_identifier: str) -> None:
        """Resample and normalize a data.

        Resample, normalize and save the data to the preprocessed folder
        (~/data/DATASET_NAME/preprocessed/data_and_properties).

        Args:
            case_identifier: Case identifier of a data.
        """

        data, seg, properties = self._load_cropped(case_identifier)
        if not self.do_resample:
            print("\n", "Skip resampling...")
            properties["resampling_flag"] = False
            properties["shape_after_resampling"] = np.array(data[0].shape)
            properties["spacing_after_resampling"] = properties["original_spacing"]
        else:
            properties["resampling_flag"] = True

            before = {"spacing": properties["original_spacing"], "data.shape": data.shape}

            anisotropy_flag = bool(self._check_anisotropy(properties["original_spacing"]))
            new_shape = self._calculate_new_shape(
                properties["original_spacing"], properties["shape_after_cropping"]
            )
            data = resample_image(data, new_shape, anisotropy_flag)
            seg = resample_image(seg.astype(np.float32), new_shape, anisotropy_flag)
            properties["anisotropy_flag"] = anisotropy_flag
            properties["shape_after_resampling"] = np.array(data[0].shape)
            properties["spacing_after_resampling"] = np.array(self.target_spacing)

            after = {
                "spacing": properties["spacing_after_resampling"],
                "data.shape (data is resampled)": data.shape,
            }

            print("before:", before, "\nafter: ", after, "\n")

        if not self.do_normalize:
            print("\nSkip normalization...")
            properties["normalization_flag"] = False
        else:
            properties["normalization_flag"] = True
            properties["use_nonzero_mask_for_norm"] = self.use_nonzero_mask
            data, seg = self._normalize(data, seg)

        all_data = np.vstack((data, seg.astype(np.float32))).astype(np.float32)
        print(
            "Saving: ",
            os.path.join(
                self.preprocessed_folder, "data_and_properties", "%s.npz" % case_identifier
            ),
            "\n",
        )
        np.savez_compressed(
            os.path.join(
                self.preprocessed_folder, "data_and_properties", "%s.npz" % case_identifier
            ),
            data=all_data.astype(np.float32),
        )
        with open(
            os.path.join(
                self.preprocessed_folder, "data_and_properties", "%s.pkl" % case_identifier
            ),
            "wb",
        ) as f:
            pickle.dump(properties, f)  # nosec B301

    def run(self) -> None:
        """Perform the resampling, normalization and saving of the dataset."""

        # get all training data
        datalist, self.modalities = self._create_datalist()

        load_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
        ]
        transforms = Compose(load_transforms)

        log.info("Initializing to run preprocessing...")
        log.info(f"Cropped folder: {self.cropped_folder}")
        log.info(f"Preprocessed folder: {self.preprocessed_folder}")
        # get target spacing
        self.target_spacing = self._get_target_spacing(datalist, transforms)
        log.info(f"Target spacing: {np.array(self.target_spacing)}.")

        # get intensity properties if input contains CT data
        if "CT" in self.modalities.values():
            log.info("CT input, calculating intensity propoerties...")
            self.intensity_properties = self._collect_intensities(datalist, transforms)
        else:
            self.intensity_properties = None
            log.info("Non CT input, skipping the calculation of intensity properties...")

        # crop to non zero
        self._crop_from_list_of_files(datalist, transforms)

        list_of_cropped_npz_files = subfiles(self.cropped_folder, True, None, ".npz", True)

        # determine whether to use non zero mask for normalization
        self.use_nonzero_mask = self._determine_whether_to_use_mask_for_norm()

        # resample and normalize
        self._preprocess(list_of_cropped_npz_files)

        # save dataset properties
        list_of_preprocessed_npz_files = subfiles(
            self.preprocessed_npz_folder, True, None, ".npz", True
        )
        self._save_dataset_properties(list_of_preprocessed_npz_files)


class DealiasPreprocessor(RegPreprocessor):
    """Preprocessor for deep unfolding dealiasing."""

    def _create_datalist(self) -> tuple[list[dict[str, str]], list[str], dict[int, str]]:
        """Read the dataset.json in 'raw' directory and extract useful information.

        Returns:
            - List of dictionaries containing the paths to the image and its label.
            - Image keys in datalist.
            - Dictionary containing the modalities indicated in dataset.json.
        """

        datalist = []

        json_file = os.path.join(self.dataset_path, "dataset.json")

        d = load_json(json_file)
        training_files = d["training"]
        num_modalities = len(d["modality"].keys())
        for tr in training_files:
            cur_pat = OrderedDict()
            image_paths = []
            label_paths = []
            for mod in range(num_modalities):
                image_paths.append(
                    os.path.join(
                        self.dataset_path,
                        "imagesTr",
                        tr["image"].split("/")[-1][:-7] + "_%04.0d.nii.gz" % mod,
                    )
                )

            for i in range(0, 2):
                label_paths.append(
                    os.path.join(
                        self.dataset_path,
                        "labelsTr",
                        tr["image"].split("/")[-1][:-7] + "_%04.0d.nii.gz" % i,
                    )
                )
            cur_pat["image"] = image_paths
            cur_pat["label"] = label_paths
            datalist.append(cur_pat)
            modalities = {int(i): d["modality"][str(i)] for i in d["modality"].keys()}

        return datalist, modalities

    def _load_cropped(self, case_identifier) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load image, label and properties of a cropped data.

        Args:
            case_identifier: Case identifier of a data.

        Returns:
            - Image numpy array
            - Label numpy array
            - Data properties
        """

        all_data = np.load(os.path.join(self.cropped_folder, "%s.npz" % case_identifier))["data"]
        data = all_data[:-2].astype(np.float32)
        seg = all_data[-2:].astype(np.float32)
        properties = self._load_properties_of_cropped(case_identifier)
        return data, seg, properties

    def _resample_and_normalize(self, case_identifier: str) -> None:
        """Resample and normalize a data.

        Resample, normalize and save the data to the preprocessed folder
        (~/data/DATASET_NAME/preprocessed/data_and_properties).

        Args:
            case_identifier: Case identifier of a data.
        """

        # seg is now a double-channel array, seg[0] contains the dealiased velocity while seg[1]
        # contains the segmentation of aliased pixels
        data, seg, properties = self._load_cropped(case_identifier)
        if not self.do_resample:
            print("\n", "Skip resampling...")
            properties["resampling_flag"] = False
            properties["shape_after_resampling"] = np.array(data[0].shape)
            properties["spacing_after_resampling"] = properties["original_spacing"]
        else:
            properties["resampling_flag"] = True

            before = {"spacing": properties["original_spacing"], "data.shape": data.shape}

            anisotropy_flag = bool(self._check_anisotropy(properties["original_spacing"]))
            new_shape = self._calculate_new_shape(
                properties["original_spacing"], properties["shape_after_cropping"]
            )
            data = resample_image(data, new_shape, anisotropy_flag)
            seg[:-1] = resample_image(seg[:-1].astype(np.float32), new_shape, anisotropy_flag)
            seg[-1:] = resample_label(seg[-1:].astype(np.uint32), new_shape, anisotropy_flag)

            properties["anisotropy_flag"] = anisotropy_flag
            properties["shape_after_resampling"] = np.array(data[0].shape)
            properties["spacing_after_resampling"] = np.array(self.target_spacing)

            after = {
                "spacing": properties["spacing_after_resampling"],
                "data.shape (data is resampled)": data.shape,
            }

            print("before:", before, "\nafter: ", after, "\n")

        if not self.do_normalize:
            print("\nSkip normalization...")
            properties["normalization_flag"] = False
        else:
            properties["normalization_flag"] = True
            properties["use_nonzero_mask_for_norm"] = self.use_nonzero_mask
            data, seg = self._normalize(data, seg)

        all_data = np.vstack((data, seg)).astype(np.float32)
        print(
            "Saving: ",
            os.path.join(
                self.preprocessed_folder, "data_and_properties", "%s.npz" % case_identifier
            ),
            "\n",
        )
        np.savez_compressed(
            os.path.join(
                self.preprocessed_folder, "data_and_properties", "%s.npz" % case_identifier
            ),
            data=all_data.astype(np.float32),
        )
        with open(
            os.path.join(
                self.preprocessed_folder, "data_and_properties", "%s.pkl" % case_identifier
            ),
            "wb",
        ) as f:
            pickle.dump(properties, f)  # nosec B301


if __name__ == "__main__":
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    dataset_dir = os.path.join(root, "data", "DEALIAS")
    preprocessor = SegPreprocessor(dataset_dir)
    preprocessor.run()
