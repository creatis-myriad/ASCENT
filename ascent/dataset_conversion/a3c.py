import os
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from ascent.utils.file_and_folder_operations import subdirs, subfiles
from utils import generate_dataset_json


def mhd2nifti(
    image_file: Union[str, Path],
    label_file: Union[str, Path],
    file_identifier: str,
    save_dir: Union[str, Path],
):
    """Convert mhd files to nifti files.

    Args:
        image_file: Path to image file.
        label_file: Path to label file.
        file_identifier: Identifier for the output files.
        save_dir: Path to save the converted files.
    """

    # Create the save directory if it does not exist
    os.makedirs(os.path.join(save_dir, file_identifier), exist_ok=True)

    # Load the image and label files
    image_img = sitk.ReadImage(image_file)
    label_img = sitk.ReadImage(label_file)

    # Save the image and label files
    sitk.WriteImage(
        image_img, os.path.join(save_dir, file_identifier, f"{file_identifier}.nii.gz")
    )
    sitk.WriteImage(
        label_img, os.path.join(save_dir, file_identifier, f"{file_identifier}_gt.nii.gz")
    )


def convert_to_nnUNet(
    data_dir: Union[Path, str],
    save_dir: Union[Path, str],
    output_dir: Union[Path, str],
) -> None:
    """Convert Camus dataset to nnUNet's format.

    Args:
        data_dir: Path to the dataset.
        save_dir: Path to save the converted data in nifti format.
        output_dir: Path to the output folder to save the converted data in nnUNet format.
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

    os.makedirs(save_dir, exist_ok=True)

    cases = subdirs(data_dir, join=True)
    for case in tqdm(cases, desc="Converting data to nifti", unit="case"):
        case_path = subfiles(case, suffix=".mhd", join=True)
        image_file_list = [f for f in case_path if "gt" not in f]
        for image_file in image_file_list:
            file_identifier = os.path.basename(image_file)[:-4]
            label_file = image_file[:-4] + "_gt.mhd"
            mhd2nifti(image_file, label_file, file_identifier, save_dir)

    cases = subdirs(save_dir, join=True)
    for case in tqdm(cases, desc="Converting data to nnunet", unit="case"):
        case_path = subfiles(case, suffix=".nii.gz", join=True)
        image_file_list = [f for f in case_path if "gt" not in f]
        for image_file in image_file_list:
            file_identifier = os.path.basename(image_file)[:-7]
            label_file = image_file[:-7] + "_gt.nii.gz"
            image_img = sitk.ReadImage(image_file)
            label_img = sitk.ReadImage(label_file)
            sitk.WriteImage(
                image_img, os.path.join(train_images_out_dir, f"{file_identifier}_0000.nii.gz")
            )
            sitk.WriteImage(
                label_img, os.path.join(train_labels_out_dir, f"{file_identifier}.nii.gz")
            )


def main(
    data_dir: Union[Path, str],
    save_dir: Union[Path, str],
    output_dir: Union[Path, str],
    dataset_name: str,
) -> None:
    """Run the script.

    Args:
        data_dir: Path to the dataset.
        save_dir: Path to save the converted data in nifti format.
        output_dir: Path to the output folder to save the converted data in nnUNet format.
        dataset_name: Name of the dataset.
    """
    output_dir = os.path.join(output_dir, dataset_name, "raw")
    imagesTr = os.path.join(output_dir, "imagesTr")
    imagesTs = os.path.join(output_dir, "imagesTs")
    os.makedirs(output_dir, exist_ok=True)
    # Convert to nnUNet's format
    convert_to_nnUNet(data_dir, save_dir, output_dir)
    # Generate dataset.json
    generate_dataset_json(
        os.path.join(output_dir, "dataset.json"),
        imagesTr,
        imagesTs,
        ("US",),
        {0: "background", 1: "LV", 2: "MYO"},
        dataset_name,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="Path to the dataset.")
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        required=True,
        help="Path to save the converted data in nifti format.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output folder to save the converted data in nnUNet format.",
    )
    parser.add_argument(
        "-n", "--dataset_name", type=str, required=True, help="Name of the dataset."
    )
    args = parser.parse_args()
    main(
        args.data_dir,
        args.save_dir,
        args.output_dir,
        args.dataset_name,
    )
