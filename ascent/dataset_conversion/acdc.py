import os
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from ascent.utils.file_and_folder_operations import subdirs, subfiles
from utils import generate_dataset_json


def crop_around_label(
    image_file: Union[str, Path],
    label_file: Union[str, Path],
    margin_ratio: float = 0.25,
) -> tuple[sitk.Image, sitk.Image]:
    """Crop data around the label with the given margin ratio.

    Args:
        image_file: Path to image file.
        label_file: Path to label file.
        margin_ratio: Margin to keep during the cropping.

    Returns:
        tuple[sitk.Image, sitk.Image]: Cropped image and label.
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
        :, cropped_min_y : cropped_max_y + 1, cropped_min_x : cropped_max_x + 1
    ]
    cropped_label_data = label_data[
        :, cropped_min_y : cropped_max_y + 1, cropped_min_x : cropped_max_x + 1
    ]

    # Create a new SimpleITK images with the cropped data
    cropped_image_img = sitk.GetImageFromArray(cropped_image_data)
    cropped_image_img.SetSpacing(image_img.GetSpacing())
    cropped_image_img.SetOrigin(image_img.GetOrigin())
    cropped_image_img.SetDirection(image_img.GetDirection())

    cropped_label_img = sitk.GetImageFromArray(cropped_label_data.astype(np.uint8))
    cropped_label_img.SetSpacing(label_img.GetSpacing())
    cropped_label_img.SetOrigin(label_img.GetOrigin())
    cropped_label_img.SetDirection(label_img.GetDirection())

    return cropped_image_img, cropped_label_img


def convert_to_nnUNet(
    data_dir: Union[Path, str],
    output_dir: Union[Path, str],
    crop_foreground: bool = False,
    crop_margin_ratio: float = 0.25,
) -> None:
    """Convert Camus dataset to nnUNet's format.

    Args:
        data_dir: Path to the dataset.
        output_dir: Path to the output folder to save the converted data.
        crop_foreground: Whether to crop around the foreground.
        crop_margin_ratio: Margin ratio to keep during the cropping.
    """
    train_images_out_dir = os.path.join(output_dir, "imagesTr")
    train_labels_out_dir = os.path.join(output_dir, "labelsTr")
    test_images_out_dir = os.path.join(output_dir, "imagesTs")
    test_labels_out_dir = os.path.join(output_dir, "labelsTs")

    os.makedirs(train_images_out_dir, exist_ok=True)
    os.makedirs(train_labels_out_dir, exist_ok=True)
    os.makedirs(test_images_out_dir, exist_ok=True)
    os.makedirs(test_labels_out_dir, exist_ok=True)

    train_cases = subdirs(os.path.join(data_dir, "training"), join=True)
    test_cases = subdirs(os.path.join(data_dir, "testing"), join=True)

    for case in tqdm(train_cases, desc="Converting train data", unit="case"):
        case_path = subfiles(case, suffix=".nii.gz", join=True)
        image_file_list = [f for f in case_path if "4d" not in f and "gt" not in f]
        for image_file in image_file_list:
            file_identifier = os.path.basename(image_file)[:-7]
            label_file = image_file[:-7] + "_gt.nii.gz"
            if crop_foreground:
                image_img, label_img = crop_around_label(image_file, label_file, crop_margin_ratio)
            else:
                image_img = sitk.ReadImage(image_file)
                label_img = sitk.ReadImage(label_file)
            sitk.WriteImage(
                image_img, os.path.join(train_images_out_dir, f"{file_identifier}_0000.nii.gz")
            )
            sitk.WriteImage(
                label_img, os.path.join(train_labels_out_dir, f"{file_identifier}.nii.gz")
            )

    for case in tqdm(test_cases, desc="Converting test data", unit="case"):
        case_path = subfiles(case, suffix=".nii.gz", join=True)
        image_file_list = [f for f in case_path if "4d" not in f and "gt" not in f]
        for image_file in image_file_list:
            file_identifier = os.path.basename(image_file)[:-7]
            label_file = image_file[:-7] + "_gt.nii.gz"
            if crop_foreground:
                image_img, label_img = crop_around_label(image_file, label_file, crop_margin_ratio)
            else:
                image_img = sitk.ReadImage(image_file)
                label_img = sitk.ReadImage(label_file)
            sitk.WriteImage(
                image_img, os.path.join(test_images_out_dir, f"{file_identifier}_0000.nii.gz")
            )
            sitk.WriteImage(
                label_img, os.path.join(test_labels_out_dir, f"{file_identifier}.nii.gz")
            )


def main(
    data_dir: Union[Path, str],
    output_dir: Union[Path, str],
    dataset_name: str,
    crop_foreground: bool = False,
    crop_margin_ratio: float = 0.25,
) -> None:
    """Run the script.

    Args:
        data_dir: Path to the dataset.
        output_dir: Path to the output folder to save the converted data.
        dataset_name: Name of the dataset.
        crop_foreground: Whether to crop around the foreground.
        crop_margin_ratio: Margin ratio to keep during the cropping.
    """
    output_dir = os.path.join(output_dir, dataset_name, "raw")
    imagesTr = os.path.join(output_dir, "imagesTr")
    imagesTs = os.path.join(output_dir, "imagesTs")
    os.makedirs(output_dir, exist_ok=True)
    # Convert to nnUNet's format
    convert_to_nnUNet(data_dir, output_dir, crop_foreground, crop_margin_ratio)
    # Generate dataset.json
    generate_dataset_json(
        os.path.join(output_dir, "dataset.json"),
        imagesTr,
        imagesTs,
        ("MRI",),
        {0: "background", 1: "RV", 2: "MYO", 3: "LV"},
        dataset_name,
    )


if __name__ == "__main__":
    import argparse

    # Ignore SimpleITK warnings issued by the ACDC dataset
    sitk.ProcessObject_SetGlobalWarningDisplay(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="Path to the dataset.")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output folder to save the converted data.",
    )
    parser.add_argument(
        "-n", "--dataset_name", type=str, required=True, help="Name of the dataset."
    )
    parser.add_argument(
        "-cf",
        "--crop_foreground",
        action="store_true",
        help="Whether to crop around the foreground.",
    )
    parser.add_argument(
        "-cm",
        "--crop_margin_ratio",
        type=float,
        default=0.25,
        help="Margin ratio to keep during the cropping.",
    )
    args = parser.parse_args()
    main(
        args.data_dir,
        args.output_dir,
        args.dataset_name,
        args.crop_foreground,
        args.crop_margin_ratio,
    )
