import os
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from ascent.preprocessing.preprocessing import resample_image, resample_label
from utils import generate_dataset_json


def convert_to_nnUNet(
    data_dir: Union[Path, str],
    output_dir: Union[Path, str],
    test: bool = False,
    views: list = ["A2C", "A4C"],
    resize: bool = False,
    txt_file: Union[Path, str] = None,
) -> None:
    """Convert Cardinal dataset to nnUNet's format.

    Args:
        data_dir: Path to the dataset.
        output_dir: Path to the output folder to save the converted data.
        test: Whether is test dataset.
        views: Views to be converted.
        resize: Whether to resize images to 256x256xT.
        txt_file: Text file that contains patient indexes to select.
    """
    if not test:
        images_out_dir = os.path.join(output_dir, "imagesTr")
        labels_out_dir = os.path.join(output_dir, "labelsTr")
    else:
        images_out_dir = os.path.join(output_dir, "imagesTs")
        labels_out_dir = os.path.join(output_dir, "labelsTs")

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    all_cases = [
        file[:4]
        for file in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, file)) and file.endswith(".nii.gz")
    ]
    all_cases = np.unique(np.array(all_cases)).tolist()

    if txt_file is not None:
        if os.path.isfile(txt_file):
            with open(txt_file) as f:
                selected_cases = f.read().splitlines()
    else:
        selected_cases = [
            file[:4]
            for file in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, file)) and file.endswith(".nii.gz")
        ]
        selected_cases = np.unique(np.array(selected_cases)).tolist()

    for case in tqdm(selected_cases):
        for view in views:
            case_identifier = f"{case}_{view}"
            image = sitk.ReadImage(os.path.join(data_dir, f"{case_identifier}_bmode.nii.gz"))
            image.SetSpacing([*image.GetSpacing()[:-1], 1.0])
            if os.path.isfile(os.path.join(data_dir, f"{case_identifier}_mask.nii.gz")):
                label = sitk.ReadImage(os.path.join(data_dir, f"{case_identifier}_mask.nii.gz"))
                label.SetSpacing([*label.GetSpacing()[:-1], 1.0])
            else:
                label = None

            if resize:
                ori_shape = image.GetSize()
                ori_spacing = image.GetSpacing()
                new_shape = [256, 256, ori_shape[-1]]
                new_spacing = np.array(ori_spacing) * np.array(ori_shape) / np.array(new_shape)
                image_array = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
                image_array = image_array[None]
                resized_image_array = resample_image(image_array, new_shape, True)
                image = sitk.GetImageFromArray(resized_image_array[0].transpose(2, 1, 0))
                image.SetSpacing(new_spacing)

            sitk.WriteImage(image, os.path.join(images_out_dir, f"{case_identifier}_0000.nii.gz"))
            if label is not None:
                if resize:
                    label_array = sitk.GetArrayFromImage(label).transpose(2, 1, 0)
                    label_array = label_array[None]
                    resized_label_array = resample_label(label_array, new_shape, True)
                    label = sitk.GetImageFromArray(resized_label_array[0].transpose(2, 1, 0))
                    label.SetSpacing(new_spacing)
                sitk.WriteImage(label, os.path.join(labels_out_dir, f"{case_identifier}.nii.gz"))


def convert_to_CAMUS_submission(
    predictions_dir: Union[Path, str],
    output_dir: Union[Path, str],
) -> None:
    """Convert predictions to correct format for submission.

    Args:
        predictions_dir: Path to the prediction folder.
        output_dir: Path to the output folder to save the converted predictions.
        la_predictions_dir: Path to 2D nnUNet's predictions to retrieve the left atrium segmentations.
    """
    os.makedirs(output_dir, exist_ok=True)
    for case in tqdm(os.listdir(predictions_dir)):
        case_path = os.path.join(predictions_dir, case)
        case_identifier = case[:-7]
        image = sitk.ReadImage(case_path)
        if "sequence" in case_identifier:
            case_identifier = case_identifier[:-9]
            image_array = sitk.GetArrayFromImage(image)
            spacing = image.GetSpacing()
            ori_spacing = [0.308, 0.154, 1.54]
            if not np.all(np.array(np.round(spacing, 3))[:-1] == np.array(ori_spacing)[:-1]):
                shape = image.GetSize()
                image_array = image_array.transpose(2, 1, 0)
                image_array = image_array[None]
                ori_shape = (
                    np.round(np.array(spacing) * np.array(shape) / np.array(ori_spacing))
                    .astype(int)
                    .tolist()
                )
                resized = resample_label(image_array, ori_shape, True)
                image_array = resized[0]
                image_array = image_array.transpose(2, 1, 0)
            spacing = ori_spacing

            ed_frame_array = image_array[0:1].astype(np.uint8)
            es_frame_array = image_array[-1:].astype(np.uint8)

            ed_frame = sitk.GetImageFromArray(ed_frame_array)
            es_frame = sitk.GetImageFromArray(es_frame_array)
            ed_frame.SetSpacing(spacing)
            es_frame.SetSpacing(spacing)
            sitk.WriteImage(ed_frame, os.path.join(output_dir, f"{case_identifier}_ED.mhd"))
            sitk.WriteImage(es_frame, os.path.join(output_dir, f"{case_identifier}_ES.mhd"))
        else:
            sitk.WriteImage(image, os.path.join(output_dir, f"{case_identifier}.mhd"))


def convert_to_CARDINAL_evaluation(
    predictions_dir: Union[Path, str],
    output_dir: Union[Path, str],
) -> None:
    """Convert predictions to correct format for submission.

    Args:
        predictions_dir: Path to the prediction folder.
        output_dir: Path to the output folder to save the converted predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    for case in tqdm(os.listdir(predictions_dir)):
        case_path = os.path.join(predictions_dir, case)
        image = sitk.ReadImage(case_path)

        image_array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        ori_spacing = [0.308, 0.308, 1.0]
        if not np.all(np.array(np.round(spacing, 3))[:-1] == np.array(ori_spacing)[:-1]):
            shape = image.GetSize()
            image_array = image_array.transpose(2, 1, 0)
            image_array = image_array[None]
            ori_shape = (
                np.round(np.array(spacing) * np.array(shape) / np.array(ori_spacing))
                .astype(int)
                .tolist()
            )
            resized = resample_label(image_array, ori_shape, True)
            image_array = resized[0]
            image_array = image_array.transpose(2, 1, 0)
        spacing = ori_spacing

        resized_image = sitk.GetImageFromArray(image_array)
        resized_image.SetSpacing(spacing)
        sitk.WriteImage(resized_image, os.path.join(output_dir, case))


if __name__ == "__main__":
    base = "C:/Users/ling/Desktop/Dataset/Bmode/Cardinal"
    output_dir = "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/data/CARDINAL/raw"
    # txt_file = "C:/Users/ling/Desktop/Dataset/Bmode/Cardinal/A2C_100_patients.txt"
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = "Cardinal"

    imagesTr = os.path.join(output_dir, "imagesTr")
    labelsTr = os.path.join(output_dir, "labelsTr")
    imagesTs = os.path.join(output_dir, "imagesTs")
    labelsTs = os.path.join(output_dir, "labelsTs")

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(labelsTs, exist_ok=True)

    # Convert train data to nnUNet's format
    convert_to_nnUNet(base, output_dir, views=["A2C", "A4C"], resize=True, txt_file=None)

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
    prediction_dir = "C:/Users/ling/Desktop/camus_sequence_test/lstm_processed"
    submission_dir = "C:/Users/ling/Desktop/camus_sequence_test/submission_lstm"
    convert_to_CAMUS_submission(prediction_dir, submission_dir)

    # Resize the predictions in Nifti format if they do not have their original spacing
    prediction_dir = "C:/Users/ling/Desktop/camus_sequence_test/inference_patient88"
    submission_dir = "C:/Users/ling/Desktop/camus_sequence_test/inference_patient88"
    convert_to_CARDINAL_evaluation(prediction_dir, submission_dir)
