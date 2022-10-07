import os

import SimpleITK as sitk
from tqdm import tqdm
from utils import generate_dataset_json


def rename_case(case_folder, imagesTr, labelsTr, case_identifier):
    case = os.path.basename(case_folder)
    velocity = sitk.ReadImage(os.path.join(case_folder, f"{case}_3CH.nii.gz"))
    power = sitk.ReadImage(os.path.join(case_folder, f"{case}_3CH_power.nii.gz"))
    unwrapped = sitk.ReadImage(os.path.join(case_folder, f"{case}_3CH_unwrapped.nii.gz"))
    gt = sitk.ReadImage(os.path.join(case_folder, f"{case}_3CH_gt.nii.gz"))

    sitk.WriteImage(velocity, os.path.join(imagesTr, f"{case_identifier}_0000.nii.gz"))
    sitk.WriteImage(power, os.path.join(imagesTr, f"{case_identifier}_0001.nii.gz"))
    sitk.WriteImage(unwrapped, os.path.join(imagesTr, f"{case_identifier}_0002.nii.gz"))
    sitk.WriteImage(gt, os.path.join(labelsTr, f"{case_identifier}.nii.gz"))


def convert_to_nnUNet(data_dir, dataset_name, output_dir):
    id = 1
    for case in tqdm(os.listdir(data_dir)):
        case_identifier = dataset_name + "_%04.0d" % id
        rename_case(os.path.join(data_dir, case), imagesTr, labelsTr, case_identifier)
        id += 1


if __name__ == "__main__":
    base = "C:/Users/ling/Desktop/unwrap2"
    data_path = "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/data/UNWRAP/raw"
    os.makedirs(data_path, exist_ok=True)

    dataset_name = "Dealias"

    imagesTr = os.path.join(data_path, "imagesTr")
    labelsTr = os.path.join(data_path, "labelsTr")
    imagesTs = os.path.join(data_path, "imagesTs")
    labelsTs = os.path.join(data_path, "labelsTs")

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(labelsTs, exist_ok=True)

    convert_to_nnUNet(base, dataset_name, data_path)
    generate_dataset_json(
        os.path.join(data_path, "dataset.json"),
        imagesTr,
        imagesTs,
        ("noNorm", "noNorm", "noNorm"),
        {0: "background", 1: "Doppler_Velocity"},
        dataset_name,
    )
