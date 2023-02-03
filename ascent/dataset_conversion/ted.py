import os
from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from utils import generate_dataset_json

from ascent.preprocessing.preprocessing import resample_image, resample_label


def convert_to_nnUNet(
    data_dir: Union[Path, str],
    output_dir: Union[Path, str],
    test: bool = False,
    sequence: bool = False,
    views: list = ["2CH", "4CH"],
    resize: bool = False,
) -> None:
    """Convert TED dataset to nnUNet's format.

    Args:
        data_dir: Path to the dataset.
        output_dir: Path to the output folder to save the converted data.
        test: Whether is test dataset.
        sequence: Whether to convert the whole sequence or 2CH/4CH ED/ES only. (images only)
        views: Views to be converted.
        resize: Whether to resize images to 256x256xT.
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
        # and folder not in ["patient0027", "patient0047", "patient0051", "patient0228"]
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

                        if resize:
                            ori_shape = image.GetSize()
                            ori_spacing = image.GetSpacing()
                            new_shape = [256, 256, ori_shape[-1]]
                            new_spacing = (
                                np.array(ori_spacing) * np.array(ori_shape) / np.array(new_shape)
                            )
                            image_array = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
                            image_array = image_array[None]
                            resized_image_array = resample_image(image_array, new_shape, True)
                            image = sitk.GetImageFromArray(
                                resized_image_array[0].transpose(2, 1, 0)
                            )
                            image.SetSpacing(new_spacing)

                        sitk.WriteImage(
                            image, os.path.join(images_out_dir, f"{case_identifier}_0000.nii.gz")
                        )
                        if label is not None:
                            if resize:
                                label_array = sitk.GetArrayFromImage(label).transpose(2, 1, 0)
                                label_array = label_array[None]
                                resized_label_array = resample_label(label_array, new_shape, True)
                                label = sitk.GetImageFromArray(
                                    resized_label_array[0].transpose(2, 1, 0)
                                )
                                label.SetSpacing(new_spacing)
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

                    if resize:
                        ori_shape = image.GetSize()
                        ori_spacing = image.GetSpacing()
                        new_shape = [256, 256, ori_shape[-1]]
                        new_spacing = (
                            np.array(ori_spacing) * np.array(ori_shape) / np.array(new_shape)
                        )
                        image_array = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
                        image_array = image_array[None]
                        resized_image_array = resample_image(image_array, new_shape, True)
                        image = sitk.GetImageFromArray(resized_image_array[0].transpose(2, 1, 0))
                        image.SetSpacing(new_spacing)

                    sitk.WriteImage(
                        image, os.path.join(images_out_dir, f"{case_identifier}_0000.nii.gz")
                    )
                    if label is not None:
                        if resize:
                            label_array = sitk.GetArrayFromImage(label).transpose(2, 1, 0)
                            label_array = label_array[None]
                            resized_label_array = resample_label(label_array, new_shape, True)
                            label = sitk.GetImageFromArray(
                                resized_label_array[0].transpose(2, 1, 0).astype(np.uint8)
                            )
                            label.SetSpacing(new_spacing)
                        sitk.WriteImage(
                            label, os.path.join(labels_out_dir, f"{case_identifier}.nii.gz")
                        )


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
                    (np.array(spacing) * np.array(shape) / np.array(ori_spacing))
                    .astype(np.int8)
                    .tolist()
                )
                resized = resample_label(image_array, ori_shape, True)
                image_array = resized[0]
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
    convert_to_nnUNet(base, output_dir, sequence=True, views=["4CH"], resize=False)

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
    prediction_dir = "C:/Users/ling/Desktop/camus_sequence_test/ted"
    submission_dir = "C:/Users/ling/Desktop/camus_sequence_test/submission_ted"
    convert_to_CAMUS_submission(prediction_dir, submission_dir)
