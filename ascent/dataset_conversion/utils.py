import os
from typing import Optional

import numpy as np

from ascent.utils.file_and_folder_operations import save_json, subfiles


def get_identifiers_from_split_files(folder: str) -> np.ndarray:
    """Get unique case identifiers from split imagesTr or imagesTs folders.

    Args:
        folder: Path to folder containing train or test images.

    Returns:
        Sorted unique case identifiers array.
    """
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix=".nii.gz", join=False)])
    return uniques


def generate_dataset_json(
    output_file: str,
    imagesTr_dir: str,
    imagesTs_dir: Optional[str],
    modalities: tuple[
        str,
    ],
    labels: dict[int, str],
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
        imagesTs_dir: Path to the imagesTs folder of that dataset. Can be None
        modalities: Tuple of strings with modality names. Must be in the same order as the images
            (first entry corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
        labels: Dict mapping the label IDs to label names. Note that 0 is always supposed to be background!
            Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}.
        dataset_name: Name of the dataset.
        sort_keys: Whether to sort the keys in dataset.json.
        license: License of the dataset.
        dataset_description: Quick description of the dataset.
        dataset_reference: Website of the dataset, if available.
        dataset_release: Version of the dataset.
    """
    train_identifiers = get_identifiers_from_split_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_split_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict["name"] = dataset_name
    json_dict["description"] = dataset_description
    json_dict["tensorImageSize"] = "4D"
    json_dict["reference"] = dataset_reference
    json_dict["licence"] = license
    json_dict["release"] = dataset_release
    json_dict["modality"] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict["labels"] = {str(i): labels[i] for i in labels.keys()}

    json_dict["numTraining"] = len(train_identifiers)
    json_dict["numTest"] = len(test_identifiers)
    json_dict["training"] = [
        {"image": "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
        for i in train_identifiers
    ]
    json_dict["test"] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print(
            "WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
            "Proceeding anyways..."
        )
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)
