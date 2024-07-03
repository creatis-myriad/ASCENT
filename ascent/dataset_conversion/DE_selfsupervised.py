import argparse

import os

from pathlib import Path
from typing import Union, Optional

import SimpleITK as sitk
import numpy as np
import torch
from monai.transforms import CenterSpatialCrop, LoadImaged

from ascent.utils.file_and_folder_operations import subdirs, subfiles, save_json


def _normalize(data: np.ndarray) -> np.ndarray:
    """Normalize data.

    Args:
        data: Image numpy array to be normalized.

    Returns:
        - Normalized image.
    """
    data = (data - data.mean()) / (
                    data.std() + 1e-8
    )
    return data

def crop_around_label(
    image_file: Union[str, Path],
    label_file: Union[str, Path],
    margin_ratio: float = 0.1,
) -> list[torch.tensor, float, float]:
    """Crop data around the label with the given margin ratio.

    Args:
        image_file: Path to image file.
        label_file: Path to label file.
        margin_ratio: Margin to keep during the cropping.

    Returns:
        list[torch.tensor, float, float]: Cropped image and cropping size in y and x.
    """

    # Load the image and label files
    image_img = sitk.ReadImage(image_file)
    label_img = sitk.ReadImage(label_file)
    image_data = sitk.GetArrayFromImage(image_img)
    label_data = sitk.GetArrayFromImage(label_img)

    # Calculate the bounding box of the label
    non_zero_indices = np.where(label_data != 0)
    min_y, max_y = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    min_x, max_x = np.min(non_zero_indices[2]), np.max(non_zero_indices[2])

    # Calculate the margin around the label
    margin_y = int((max_y - min_y) * margin_ratio)
    margin_x = int((max_x - min_x) * margin_ratio)

    # Calculate the cropped bounding box
    cropped_min_y = max(min_y - margin_y, 0)
    cropped_max_y = min(max_y + margin_y, image_data.shape[1] - 1)
    cropped_min_x = max(min_x - margin_x, 0)
    cropped_max_x = min(max_x + margin_x, image_data.shape[2] - 1)

    # Crop the MRI data
    cropped_image_data = image_data[
        :, cropped_min_y: cropped_max_y + 1, cropped_min_x: cropped_max_x + 1
    ]

    return torch.tensor(cropped_image_data),  cropped_max_y-cropped_min_y, cropped_max_x-cropped_min_x


def crop_from_size(
        image_file: Union[str, Path],
        y_size: int,
        x_size: int
) -> torch.tensor:
    """Crop data in the center according to a given size.

    Args:
        image_file: Path to image file.
        y_size: Cropping size for y.
        x_size: Cropping size for x.

    Returns:
        sitk.Image: Cropped image.
    """
    # Load the image and label files
    image_img = sitk.ReadImage(image_file)
    image_data = sitk.GetArrayFromImage(image_img)

    crop = CenterSpatialCrop(roi_size=(y_size, x_size), lazy=False)
    cropped_image_data = crop(image_data)

    return cropped_image_data


def convert_to_nnUNet(
    data_dir: Union[Path, str],
    output_dir: Union[Path, str],
) -> None:
    """Convert DE dataset to nnUNet's format.

    Args:
        data_dir: Path to the dataset.
        output_dir: Path to the output folder to save the converted data.
    """

    images_out_dir = os.path.join(output_dir, "imagesTr")
    os.makedirs(images_out_dir, exist_ok=True)

    all_cases = []

    training_myosaiq = os.path.join(data_dir, "database/training")
    for folder in subdirs(training_myosaiq):
        for file in subfiles(os.path.join(folder, "images"), suffix=".nii.gz"):
            all_cases.append(file)
            case_identifier = os.path.basename(file)[:-7]

            image = sitk.ReadImage(file)
            image_data = torch.tensor(sitk.GetArrayFromImage(image).astype(float))

            image_data = np.asarray(image_data.permute(2, 1, 0).unsqueeze(0))
            image_data = _normalize(image_data)

            np.savez_compressed(os.path.join(images_out_dir, f"{case_identifier}.npz"), data=image_data)

    testing_myosaiq = os.path.join(data_dir, "database/testing")
    for folder in subdirs(testing_myosaiq):
        for file in subfiles(os.path.join(folder, "images"), suffix=".nii.gz"):
            all_cases.append(file)
            case_identifier = os.path.basename(file)[:-7]

            image = sitk.ReadImage(file)
            image_data = torch.tensor(sitk.GetArrayFromImage(image).astype(float))

            image_data = np.asarray(image_data.permute(2, 1, 0).unsqueeze(0))
            image_data = _normalize(image_data)

            np.savez_compressed(os.path.join(images_out_dir, f"{case_identifier}.npz"), data=image_data)

    training_emidec = os.path.join(data_dir, "emidec-dataset-1.0.1")
    max_y_size, max_x_size = 0, 0
    for folder in subdirs(training_emidec):
        for file in subfiles(os.path.join(folder, "Images"), suffix=".nii.gz"):
            all_cases.append(file)
            case_identifier = os.path.basename(file)[:-7]

            label_file = file.replace('Images', 'Contours')
            image_data, y_size, x_size = crop_around_label(file, label_file, 0.25)

            image_data = np.asarray(image_data.permute(2, 1, 0).unsqueeze(0))
            image_data = _normalize(image_data)

            np.savez_compressed(os.path.join(images_out_dir, f"{case_identifier}.npz"), data=image_data)

            if x_size > max_x_size:
                max_x_size = x_size

            if y_size > max_y_size:
                max_y_size = y_size

    testing_emidec = os.path.join(data_dir, "emidec-segmentation-testset-1.0.0")
    for folder in subdirs(testing_emidec):
        for file in subfiles(os.path.join(folder, "Images"), suffix=".nii.gz"):
            all_cases.append(file)
            case_identifier = os.path.basename(file)[:-7]

            image_data = crop_from_size(file, max_y_size, max_x_size)

            image_data = np.asarray(image_data.permute(2, 1, 0).unsqueeze(0))
            image_data = _normalize(image_data)

            np.savez_compressed(os.path.join(images_out_dir, f"{case_identifier}.npz"), data=image_data)

    pepr = os.path.join(data_dir, "NPZ")

    for folder in subdirs(pepr):
        for subfolder in subdirs(folder):
            for file in subfiles(subfolder, suffix="DE_SA.npz"):
                all_cases.append(file)
                case_identifier = os.path.basename(file)[:-4]
                data = np.load(file)
                data_shape = data['images'].shape
                crop = CenterSpatialCrop(roi_size=(data_shape[1]//3, data_shape[2]//3), lazy=False)
                image_data = crop(data['images'])

                image_data = np.asarray(image_data.permute(2, 1, 0).unsqueeze(0))
                image_data = _normalize(image_data)

                np.savez_compressed(os.path.join(images_out_dir, f"{case_identifier}.npz"), data=image_data)


def generate_dataset_json(
    output_file: str,
    imagesTr_dir: str,
    modalities: tuple[str, ...],
    dataset_name: str,
    sort_keys: bool = True,
    license: Optional[str] = "hands off!",
    dataset_description: Optional[str] = "",
    dataset_reference: Optional[str] = "",
    dataset_release: Optional[str] = "0.0",
) -> None:
    """Generate dataset.json file.

    Args:
        output_file: Full path to the dataset.json you intend to write, so output_file='DATASET_PATH/dataset.json'
            where the folder DATASET_PATH points to is the one with the imagesTr and labelsTr subfolders.
        imagesTr_dir: Path to the imagesTr folder of that dataset.
        modalities: Tuple of strings with modality names. Must be in the same order as the images
            (first entry corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
        dataset_name: Name of the dataset.
        sort_keys: Whether to sort the keys in dataset.json.
        license: License of the dataset.
        dataset_description: Quick description of the dataset.
        dataset_reference: Website of the dataset, if available.
        dataset_release: Version of the dataset.
    """
    uniques_nii = np.unique([i[:-12]+".nii.gz" for i in subfiles(imagesTr_dir, suffix=".nii.gz", join=False)]).tolist()
    uniques_npz = np.unique([i[:-9] + ".npz" for i in subfiles(imagesTr_dir, suffix=".npz", join=False)]).tolist()
    train_identifiers = uniques_nii + uniques_npz

    json_dict = {}
    json_dict["name"] = dataset_name
    json_dict["description"] = dataset_description
    json_dict["tensorImageSize"] = "3D"
    json_dict["reference"] = dataset_reference
    json_dict["licence"] = license
    json_dict["release"] = dataset_release
    json_dict["modality"] = {str(i): modalities[i] for i in range(len(modalities))}

    json_dict["training"] = [
            {"image": "./imagesTr/%s" % i}
            for i in train_identifiers
    ]
    json_dict["numTraining"] = len(train_identifiers)

    if not output_file.endswith("dataset.json"):
        print(
            "WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
            "Proceeding anyways..."
        )
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)

def main():
    """Main function where we define the argument for the script."""
    parser = argparse.ArgumentParser(
        description=(
            "Script to convert MYOSAIQ challenge dataset to 'nnHPU' format and create balanced "
            "train/val or train/val/test split."
        )
    )
    parser.add_argument(
        "--data_dir", type=str, default="../../data/DE", help="Path to the root directory of the downloaded MYOSAIQ data"
    )
    parser.add_argument("--output_dir", type=str, default="../../data", help="Path to the dataset/raw directory")

    args = parser.parse_args()

    dataset_name = "DE"
    output_dir = os.path.join(args.output_dir, dataset_name, "raw")
    os.makedirs(args.output_dir, exist_ok=True)

    imagesTr = os.path.join(output_dir, "imagesTr")

    os.makedirs(imagesTr, exist_ok=True)

    # Convert train data to nnUNet's format
    convert_to_nnUNet(args.data_dir, output_dir)

    # Generate dataset.json
    generate_dataset_json(
        output_file=os.path.join(output_dir, "dataset.json"),
        imagesTr_dir=imagesTr,
        modalities=("MRI",),
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    main()