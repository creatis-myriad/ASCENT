import itertools
import json
import os
import pickle  # nosec B403
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
from joblib import Parallel, delayed
from monai.transforms import (
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    SpatialCropd,
)
from monai.transforms.utils import generate_spatial_bounding_box
from skimage.transform import resize

from covid.utils.file_and_folder_operations import subfiles


class Preprocessor:
    def __init__(
        self,
        dataset_path: Union[str, Path],
        do_resample: bool = True,
        do_normalize: bool = True,
        dilation: bool = False,
        num_workers: int = 12,
        overwrite_existing: bool = False,
    ):
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

    def create_datalist(self):
        lists = []

        json_file = os.path.join(self.dataset_path, "dataset.json")
        with open(json_file) as jsn:
            d = json.load(jsn)
            training_files = d["training"]
        num_modalities = len(d["modality"].keys())
        if num_modalities > 1:
            image_keys = ["image_%d" % mod for mod in range(num_modalities)]
        else:
            image_keys = ["image"]
        for tr in training_files:
            cur_pat = OrderedDict()
            for mod in range(num_modalities):
                cur_pat[image_keys[mod]] = os.path.join(
                    self.dataset_path,
                    "imagesTr",
                    tr["image"].split("/")[-1][:-7] + "_%04.0d.nii.gz" % mod,
                )
            cur_pat["label"] = os.path.join(
                self.dataset_path, "labelsTr", tr["label"].split("/")[-1]
            )
            lists.append(cur_pat)

        return lists, image_keys, {int(i): d["modality"][str(i)] for i in d["modality"].keys()}

    def get_target_spacing(self, datalist: list[dict], transforms: list) -> list:
        spacings = self.run_parallel_from_raw(self.get_spacing, datalist, transforms)
        spacings = np.array(spacings)
        target_spacing = np.median(spacings, axis=0)
        if max(target_spacing) / min(target_spacing) >= 3:
            lowres_axis = np.argmin(target_spacing)
            target_spacing[lowres_axis] = np.percentile(spacings[:, lowres_axis], 10)
        return list(target_spacing)

    def get_spacing(self, data: dict, transforms: list, **kwargs) -> list:
        data = transforms(data)
        return data["image"].meta["pixdim"][1:4].tolist()

    def collect_intensities(
        self, datalist: list[dict], transforms: list
    ) -> dict[dict,]:
        intensity_properties = OrderedDict()
        for i in range(len(self.modalities)):
            mod = {"modality": i}
            intensity_properties[i] = OrderedDict()
            intensities = self.run_parallel_from_raw(
                self.get_intensities, datalist, transforms, **mod
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

    def get_intensities(self, data: dict, transforms: list, **kwargs) -> list:
        data = transforms(data)
        image = data["image"].astype(np.float32)
        label = data["label"].astype(np.uint8)
        foreground_idx = np.where(label[0] > 0)
        intensities = image[kwargs["modality"]][foreground_idx].tolist()
        return intensities

    def crop_from_list_of_files(self, datalist: list[dict], transforms: list) -> None:
        os.makedirs(self.cropped_folder, exist_ok=True)
        self.run_parallel_from_raw(self.crop, datalist, transforms)

    def crop(self, data: dict, transforms: list, **kwargs) -> None:
        list_of_data_files = list(data.values())[:-1]
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
            self.all_size_reductions.append(properties["cropping_size_reduction"])
            print(
                "before crop:",
                properties["original_shape"],
                "after crop:",
                properties["shape_after_cropping"],
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
    def get_case_identifier_from_raw_data(data: dict):
        return os.path.basename(data["image"].meta["filename_or_obj"]).split(".nii.gz")[0][:-5]

    @staticmethod
    def get_case_identifier_from_npz(case):
        case_identifier = os.path.basename(case)[:-4]
        return case_identifier

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def calculate_new_shape(self, spacing, shape):
        spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def get_all_size_reductions(self, list_of_cropped_npz_files: list) -> None:
        properties = self.run_parallel_from_cropped(
            self.get_size_reduction, list_of_cropped_npz_files
        )
        return properties

    def get_size_reduction(self, case_identifier) -> float:
        properties = self.load_properties_of_cropped(case_identifier)
        return properties["cropping_size_reduction"]

    def load_cropped(self, case_identifier):
        all_data = np.load(os.path.join(self.cropped_folder, "%s.npz" % case_identifier))["data"]
        data = all_data[:-1].astype(np.float32)
        seg = all_data[-1:]
        properties = self.load_properties_of_cropped(case_identifier)
        return data, seg, properties

    def load_properties_of_cropped(self, case_identifier):
        with open(os.path.join(self.cropped_folder, "%s.pkl" % case_identifier), "rb") as f:
            properties = pickle.load(f)  # nosec B301
        return properties

    def determine_whether_to_use_mask_for_norm(
        self,
    ) -> dict[bool,]:
        # only use the nonzero mask for normalization of the cropping based on it resulted in a decrease in
        # image size (this is an indication that the data is something like brats/isles and then we want to
        # normalize in the brain region only)
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

    def preprocess(self, list_of_cropped_npz_files: list):
        os.makedirs(self.preprocessed_npz_folder, exist_ok=True)
        self.run_parallel_from_cropped(self.resample_and_normalize, list_of_cropped_npz_files)

    def resample_and_normalize(self, case_identifier: str):
        data, seg, properties = self.load_cropped(case_identifier)
        if not self.do_resample:
            print("\n", "Skip resampling...")
            properties["resampling_flag"] = False
            properties["shape_after_resampling"] = np.array(data[0].shape)
            properties["spacing_after_resampling"] = properties["original_spacing"]
        else:
            properties["resampling_flag"] = True

            before = {"spacing": properties["original_spacing"], "data.shape": data.shape}

            anisotrophy_flag = self.check_anisotrophy(properties["original_spacing"])
            new_shape = self.calculate_new_shape(
                properties["original_spacing"], properties["shape_after_cropping"]
            )
            data = self.resample_image(data, new_shape, anisotrophy_flag)
            seg = self.resample_image(seg, new_shape, anisotrophy_flag)
            properties["anisotrophy_flag"] = anisotrophy_flag
            properties["shape_after_resampling"] = np.array(data[0].shape)
            properties["spacing_after_resampling"] = np.array(self.target_spacing)

            after = {
                "spacing": properties["spacing_after_resampling"],
                "data.shape (data is resampled)": data.shape,
            }

            print("before:", before, "\nafter: ", after, "\n")

        if not self.do_normalize:
            print("\n", "Skip normalization...")
            properties["normalization_flag"] = False
        else:
            properties["normalization_flag"] = True
            properties["use_nonzero_mask_for_norm"] = self.use_nonzero_mask
            data, seg, properties = self.normalize(data, seg, properties)

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

    @staticmethod
    def resample_image(image, new_shape, anisotrophy_flag):
        shape = np.array(image[0].shape)
        if np.any(shape != np.array(new_shape)):
            resized_channels = []
            if anisotrophy_flag:
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

    @staticmethod
    def resample_label(label, new_shape, anisotrophy_flag):
        shape = np.array(label[0].shape)
        if np.any(shape != np.array(new_shape)):
            reshaped = np.zeros(new_shape, dtype=np.uint8)
            n_class = np.max(label)
            if anisotrophy_flag:
                print("Anisotropic image, using separate z resampling")
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

    def normalize(self, data, seg, properties):
        assert len(self.use_nonzero_mask) == len(data)
        print("Normalization...")
        for c in range(len(data)):
            scheme = self.modalities[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
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
            else:
                if self.use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (
                    data[c][mask].std() + 1e-8
                )
                data[c][mask == 0] = 0
        print("Normalization done")
        return data, seg, properties

    def run_parallel_from_raw(self, func, datalist, transforms, **kwargs):
        return Parallel(n_jobs=self.num_workers)(
            delayed(func)(data, transforms, **kwargs) for data in datalist
        )

    def run_parallel_from_cropped(self, func, list_of_cropped_npz_files):
        return Parallel(n_jobs=self.num_workers)(
            delayed(func)(self.get_case_identifier_from_npz(npz_file))
            for npz_file in list_of_cropped_npz_files
        )

    def run(self):
        # get all training data
        datalist, image_keys, self.modalities = self.create_datalist()

        load_transforms = [
            LoadImaged(keys=[*image_keys, "label"]),
            EnsureChannelFirstd(keys=[*image_keys, "label"]),
        ]

        if len(image_keys) > 1:
            concat_transform = [ConcatItemsd(keys=image_keys, name="image")]
        else:
            concat_transform = []
        transforms = Compose(load_transforms + concat_transform)

        print("Initializing to run preprocessing...")
        print("Cropped folder: ", self.cropped_folder)
        print("Preprocessed folder: ", self.preprocessed_folder)
        # get target spacing
        self.target_spacing = self.get_target_spacing(datalist, transforms)
        print("\nTarget spacing:", self.target_spacing)

        # get intensity properties if input contains CT data
        if "CT" in self.modalities.values():
            print("\nCT input, calculating intensity propoerties...")
            self.intensity_properties = self.collect_intensities(datalist, transforms)
        else:
            self.intensity_properties = None
            print("\nNon CT input, skipping the calculation of intensity properties...")

        # crop to non zero
        self.crop_from_list_of_files(datalist, transforms)
        list_of_cropped_npz_files = subfiles(self.cropped_folder, True, None, ".npz", True)

        # get all size reductions
        if not len(self.all_size_reductions):
            self.all_size_reductions = self.get_all_size_reductions(list_of_cropped_npz_files)

        # determine whether to use non zero mask for normalization
        self.use_nonzero_mask = self.determine_whether_to_use_mask_for_norm()

        # resample and normalize
        self.preprocess(list_of_cropped_npz_files)


if __name__ == "__main__":
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    dataset_dir = os.path.join(root, "data", "CAMUS")
    preprocessor = Preprocessor(dataset_dir)
    preprocessor.run()
