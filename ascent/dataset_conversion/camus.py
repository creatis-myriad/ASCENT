import os
from pathlib import Path
from typing import Union

import SimpleITK as sitk
from tqdm import tqdm
from utils import generate_dataset_json


def convert_to_nnUNet(
    data_dir: Union[Path, str],
    output_dir: Union[Path, str],
    test: bool = False,
) -> None:
    """Convert Camus dataset to nnUNet's format.

    Args:
        data_dir: Path to the dataset.
        output_dir: Path to the output folder to save the converted data.
        test: Whether is test dataset.
    """
    if not test:
        images_out_dir = os.path.join(output_dir, "imagesTr")
        labels_out_dir = os.path.join(output_dir, "labelsTr")
    else:
        images_out_dir = os.path.join(output_dir, "imagesTs")
        labels_out_dir = os.path.join(output_dir, "labelsTs")

    for case in tqdm(os.listdir(data_dir)):
        case_path = os.path.join(data_dir, case)
        if os.listdir(case_path):
            for view in ["2CH", "4CH"]:
                for instant in ["ED", "ES"]:
                    case_identifier = f"{case}_{view}_{instant}"
                    image = sitk.ReadImage(os.path.join(case_path, f"{case_identifier}.mhd"))
                    if os.path.isfile(os.path.join(case_path, f"{case_identifier}_gt.mhd")):
                        label = sitk.ReadImage(
                            os.path.join(case_path, f"{case_identifier}_gt.mhd")
                        )
                    else:
                        label = None
                    sitk.WriteImage(
                        image, os.path.join(images_out_dir, f"{case_identifier}_0000.nii.gz")
                    )
                    if label is not None:
                        sitk.WriteImage(
                            label, os.path.join(labels_out_dir, f"{case_identifier}.nii.gz")
                        )


if __name__ == "__main__":
    base = "C:/Users/ling/Downloads/training/training"
    test_data = "C:/Users/ling/Downloads/testing/testing"
    output_dir = "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/data/CAMUS_challenge/raw"
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = "CAMUS"

    imagesTr = os.path.join(output_dir, "imagesTr")
    labelsTr = os.path.join(output_dir, "labelsTr")
    imagesTs = os.path.join(output_dir, "imagesTs")
    labelsTs = os.path.join(output_dir, "labelsTs")

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(labelsTs, exist_ok=True)

    convert_to_nnUNet(base, output_dir)
    # generate_dataset_json(
    #     os.path.join(output_dir, "dataset.json"),
    #     imagesTr,
    #     imagesTs,
    #     ("US",),
    #     {0: "background", 1: "LV", 2: "MYO", 3: "LA"},
    #     dataset_name,
    # )
    convert_to_nnUNet(test_data, output_dir, test=True)
