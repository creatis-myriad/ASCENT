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
    sequence: bool = False,
    views: list = ["2CH", "4CH"],
) -> None:
    """Convert Camus dataset to nnUNet's format.

    Args:
        data_dir: Path to the dataset.
        output_dir: Path to the output folder to save the converted data.
        test: Whether is test dataset.
        sequence: Whether to convert the whole sequence or 2CH/4CH ED/ES only. (images only)
        views: Views to be converted.
    """
    if not test:
        images_out_dir = os.path.join(output_dir, "imagesTr")
        labels_out_dir = os.path.join(output_dir, "labelsTr")
    else:
        images_out_dir = os.path.join(output_dir, "imagesTs")
        labels_out_dir = os.path.join(output_dir, "labelsTs")

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    dirs = [
        folder
        for folder in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, folder))
        and folder not in ["patient0027", "patient0047", "patient0051", "patient0228"]
    ]
    for case in tqdm(dirs):
        case_path = os.path.join(data_dir, case)
        if os.listdir(case_path):
            if not sequence:
                for view in views:
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
            else:
                for view in views:
                    case_identifier = f"{case}_{view}_sequence"
                    image = sitk.ReadImage(os.path.join(case_path, f"{case_identifier}.mhd"))
                    sitk.WriteImage(
                        image, os.path.join(images_out_dir, f"{case_identifier}_0000.nii.gz")
                    )
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


def convert_to_CAMUS_submission(
    predictions_dir: Union[Path, str], output_dir: Union[Path, str]
) -> None:
    """Convert predictions to correct format for submission.

    Args:
        predictions_dir: Path to the prediction folder.
        output_dir: Path to the output folder to save the converted predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    for case in tqdm(os.listdir(predictions_dir)):
        case_path = os.path.join(predictions_dir, case)
        case_identifier = case[:-7]
        image = sitk.ReadImage(case_path)
        sitk.WriteImage(image, os.path.join(output_dir, f"{case_identifier}.mhd"))


if __name__ == "__main__":
    base = "C:/Users/ling/Desktop/Camus/4CH_full_cycle"
    output_dir = "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/data/TED/raw"
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = "TED"

    imagesTr = os.path.join(output_dir, "imagesTr")
    labelsTr = os.path.join(output_dir, "labelsTr")
    imagesTs = os.path.join(output_dir, "imagesTs")
    labelsTs = os.path.join(output_dir, "labelsTs")

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(labelsTs, exist_ok=True)

    # Convert train data to nnUNet's format
    convert_to_nnUNet(base, output_dir, sequence=True, views=["4CH"])

    # Generate dataset.json
    generate_dataset_json(
        os.path.join(output_dir, "dataset.json"),
        imagesTr,
        imagesTs,
        ("US",),
        {0: "background", 1: "LV", 2: "MYO"},
        dataset_name,
    )

    # Convert predictions in Nifti format to raw/mhd
    prediction_dir = "C:/Users/ling/Desktop/camus_test/inference_raw"
    submission_dir = "C:/Users/ling/Desktop/camus_test/submission"
    convert_to_CAMUS_submission(prediction_dir, submission_dir)
