import argparse
import itertools
import os
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Sequence, Union

import numpy as np
import seaborn as sns
import SimpleITK as sitk
from pandas import DataFrame, Series, concat
from seaborn import PairGrid
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ascent.dataset_conversion.utils import generate_dataset_json
from ascent.utils.file_and_folder_operations import save_pickle, subdirs, subfiles


def convert_to_nnUNet(
    data_dir: Union[Path, str],
    output_dir: Union[Path, str],
    test: bool = False,
    keep_lv_myo_only: bool = False,
) -> None:
    """Convert MYOSAIQ dataset to nnUNet's format.

    Args:
        data_dir: Path to the dataset.
        output_dir: Path to the output folder to save the converted data.
        test: Whether is test dataset.
        keep_lv_myo_only: Whether to keep the left ventricle and myocardium's labels only
    """
    if not test:
        images_out_dir = os.path.join(output_dir, "imagesTr")
        labels_out_dir = os.path.join(output_dir, "labelsTr")
    else:
        images_out_dir = os.path.join(output_dir, "imagesTs")
        labels_out_dir = os.path.join(output_dir, "labelsTs")

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    all_cases = []
    for folder in subdirs(data_dir):
        for file in subfiles(os.path.join(folder, "images"), suffix=".nii.gz"):
            all_cases.append(file)

    for case in tqdm(all_cases, desc="Converting cases", unit="case"):
        case_identifier = os.path.basename(case)[:-7]
        image = sitk.ReadImage(case)
        label = sitk.ReadImage(case.replace("images", "labels"))
        if keep_lv_myo_only:
            label_data = sitk.GetArrayFromImage(label)
            label_data[label_data > 2] = 2
            label_data = label_data.astype(np.uint8)
            processed_label = sitk.GetImageFromArray(label_data)
            processed_label.SetSpacing(label.GetSpacing())
            processed_label.SetOrigin(label.GetOrigin())
            processed_label.SetDirection(label.GetDirection())
            label = processed_label
        sitk.WriteImage(image, os.path.join(images_out_dir, f"{case_identifier}_0000.nii.gz"))
        sitk.WriteImage(label, os.path.join(labels_out_dir, f"{case_identifier}.nii.gz"))


def compute_attributes(labels_dir: str) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Compute attributes for training labels, i.e., number of pixels of the left ventricle,
    myocardium, infart, and MVO.

    Args:
        labels_dir: Path to labelsTr of the converted MYOSAIQ dataset.

    Returns:
        Tuple of dataframe containing all the attributes for D8, M1, and M12 patients, respectively.
    """
    attributes = {}
    for case in subfiles(labels_dir, suffix=".nii.gz"):
        case_identifier = os.path.basename(case)[:-7]
        mask = sitk.ReadImage(case)
        mask_array = sitk.GetArrayFromImage(mask)
        LV_pixels = np.count_nonzero(mask_array[mask_array == 1])
        MYO_pixels = np.count_nonzero(mask_array[mask_array == 2])
        Infart_pixels = np.count_nonzero(mask_array[mask_array == 3])
        MVO_pixels = np.count_nonzero(mask_array[mask_array == 4])
        attributes[case_identifier] = {
            "LV_pixels": LV_pixels,
            "MYO_pixels": MYO_pixels,
            "Infart_pixels": Infart_pixels,
            "MVO_pixels": MVO_pixels,
        }

    patient_attributes = DataFrame.from_dict(attributes, orient="index")
    all_patients = list(patient_attributes.index)
    d8_idx = [i for i, s in enumerate(all_patients) if "D8" in s]
    m12_idx = [i for i, s in enumerate(all_patients) if "M12" in s]
    m1_idx = list({i for i in range(len(all_patients))} - set(d8_idx + m12_idx))

    d8_patients_attributes = patient_attributes.iloc[d8_idx]
    m1_patients_attributes = patient_attributes.iloc[m1_idx]
    m12_patients_attributes = patient_attributes.iloc[m12_idx]
    return d8_patients_attributes, m1_patients_attributes, m12_patients_attributes


def generate_patients_splits(
    patient_attribute_df: DataFrame,
    stratify: Literal["LV", "MYO", "Infart", "MVO"],
    bins: int = 5,
    test_size: Union[int, float] = None,
    train_size: Union[int, float] = None,
    seed: int = None,
) -> tuple[list, list]:
    """Splits patients into train and test subsets, preserving the distribution of `stratify`
    variable across subsets.

    Notes:
        - Wrapper around `sklearn.model_selection.train_test_split` that performs binning on continuous
        variables, since out-of-the-box `sklearn`'s `train_test_split` only works with categorical
        `stratify` variables.

    Args:
        patient_attribute_df: Dataframe containing patients ID and their attributes.
        stratify: Name of the continuous clinical attribute whose distribution in each of the subset
            should match the global distribution.
        bins: Number of bins into which to categorize the continuous `stratify` attribute's values,
            to ensure each bin is distributed representatively in the split.
        test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
            to include in the test split. If int, represents the absolute number of test samples.
            If None, the value is set to the complement of the train size.
        train_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
            to include in the train split. If int, represents the absolute number of train samples.
            If None, the value is automatically set to the complement of the test size.
        seed: Seed to control the shuffling applied to the data before applying the split.

    Returns:
        Lists of patients in the train and tests subsets, respectively.
    """

    # Collect the data of the attribute by which to stratify the split from the patient
    patients_stratify = {
        index: row[f"{stratify}_pixels"] for index, row in patient_attribute_df.iterrows()
    }

    # Compute categorical stratify variable from scalar attribute
    stratify_vals = list(patients_stratify.values())
    stratify_bins = np.linspace(min(stratify_vals), max(stratify_vals), num=bins + 1)

    # Add epsilon to the last bin's upper bound since it's excluded by `np.digitize`
    stratify_bins[-1] += 1e-6

    # Subtract 1 because bin indexing starts at 1
    stratify_labels = np.digitize(stratify_vals, stratify_bins) - 1

    patient_ids_train, patient_ids_test = train_test_split(
        list(patients_stratify),
        test_size=test_size,
        train_size=train_size,
        random_state=seed,
        stratify=stratify_labels,
    )
    return sorted(patient_ids_train), sorted(patient_ids_test)


def check_subsets(patients: Sequence[str], subsets: dict[str, Sequence[str]]) -> None:
    """Checks the lists of patients overall and in each subset to ensure each patient belongs to
    one and only subset.

    Args:
        patients: List of patients.
        subsets: Lists of patients making up subsets.
    """
    for (subset1, subset1_patients), (subset2, subset2_patients) in itertools.combinations(
        subsets.items(), 2
    ):
        if intersect := set(subset1_patients) & set(subset2_patients):
            raise RuntimeError(
                f"All subsets provided in `patient_subsets` should be disjoint from each other, but "
                f"subsets '{subset1}' and '{subset2}' have the following patients in common: "
                f"{sorted(intersect)}."
            )

    patient_ids_in_subsets = set().union(*subsets.values())
    if unassigned_patients := set(patients) - patient_ids_in_subsets:
        raise RuntimeError(
            f"All patients should be part of one of the subset in `patient_subsets`. However, the "
            f"following patients are not included in any subset: {sorted(unassigned_patients)}."
        )


def plot_patients_distribution(
    patient_attribute_df: DataFrame,
    plot_attributes: list[Literal["LV", "MYO", "Infart", "MVO"]],
    subsets: dict[str, Sequence[str]] = None,
) -> PairGrid:
    """Plots the pairwise relationships between clinical attributes for a collection (or multiple
    subsets) of patients.

    Args:
        patient_attribute_df: Dataframe containing patients ID and their attributes.
        plot_attributes: Patients' clinical attributes whose distributions to compare pairwise.
        subsets: Lists of patients making up each subset, to plot with different hues. The subsets
            should be disjoint from one another.

    Returns:
        PairGrid representing the pairwise relationships between clinical attributes for the patients.
    """
    if subsets is not None:
        check_subsets(list(patient_attribute_df.index), subsets)

    plot_attributes = [s + "_pixels" for s in plot_attributes]

    patients_data = patient_attribute_df.loc[:, plot_attributes]

    # Add additional subset information to the dataframe, if provided
    plot_kwargs = {}
    if subsets:
        plot_kwargs.update({"hue": "subset", "hue_order": list(subsets)})
        patients_data["subset"] = Series(dtype=str)
        for subset, patient_ids_in_subset in subsets.items():
            patients_data.loc[
                list(set(patient_ids_in_subset) & set(patients_data.index)), "subset"
            ] = subset

    # Plot pairwise relationships between the selected attributes in the dataset
    g = sns.pairplot(patients_data, **plot_kwargs)
    g.map_lower(sns.kdeplot)
    return g


def do_splits(splits_file: str, train_keys: Sequence[str], val_keys: Sequence[str]) -> None:
    """Create split based on the train/val keys.

    Args:
        splits_file: Path to the splits file.
        train_keys: List containing keys for training.
        val_keys: List containing keys for validation.
    """
    print("Creating new split...")
    splits = []
    splits.append(OrderedDict())
    splits[-1]["train"] = np.sort(np.array(train_keys))
    splits[-1]["val"] = np.sort(np.array(val_keys))
    save_pickle(splits, splits_file)


def main():
    """Main function where we define the argument for the script."""
    parser = argparse.ArgumentParser(
        description=(
            "Script to convert MYOSAIQ challenge dataset to 'nnHPU' format and create balanced "
            "train/val split."
        )
    )
    parser.add_argument(
        "data_dir", type=str, help="Path to the root directory of the downloaded MYOSAIQ data"
    )
    parser.add_argument("output_dir", type=str, help="Path to the dataset/raw directory")
    parser.add_argument(
        "--create_split", type=bool, default=True, help="Whether to create train/val split"
    )
    parser.add_argument(
        "--save_fig",
        type=bool,
        default=True,
        help="Whether to save the distribution plot of train/val split",
    )

    args = parser.parse_args()

    base = os.path.join(args.data_dir, "training")
    dataset_name = "MYOSAIQ"
    output_dir = os.path.join(args.output_dir, dataset_name, "raw")
    os.makedirs(args.output_dir, exist_ok=True)

    imagesTr = os.path.join(output_dir, "imagesTr")
    labelsTr = os.path.join(output_dir, "labelsTr")
    imagesTs = os.path.join(output_dir, "imagesTs")
    labelsTs = os.path.join(output_dir, "labelsTs")

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(labelsTs, exist_ok=True)

    # Convert train data to nnUNet's format
    convert_to_nnUNet(base, output_dir)

    # Generate dataset.json
    generate_dataset_json(
        os.path.join(output_dir, "dataset.json"),
        imagesTr,
        imagesTs,
        ("MRI",),
        {0: "background", 1: "LV", 2: "MYO", 3: "Infart", 4: "MVO"},
        dataset_name,
    )

    if args.create_split:
        # Compute attributes for d8, m1, and m12 patients
        (
            d8_patients_attributes,
            m1_patients_attributes,
            m12_patients_attributes,
        ) = compute_attributes(labelsTr)

        all_patients_attributes = concat(
            [d8_patients_attributes, m1_patients_attributes, m12_patients_attributes]
        )

        # Generate train/val split for d8, m1, and m12 patients
        d8_train, d8_val = generate_patients_splits(
            d8_patients_attributes, "MVO", test_size=0.2, seed=12345
        )
        m1_train, m1_val = generate_patients_splits(
            m1_patients_attributes, "Infart", test_size=0.2, seed=12345
        )
        m12_train, m12_val = generate_patients_splits(
            m12_patients_attributes, "Infart", test_size=0.2, seed=12345
        )

        myosaiq_train = d8_train + m1_train + m12_train
        myosaiq_val = d8_val + m1_val + m12_val

        subsets = {"train": myosaiq_train, "val": myosaiq_val}

        preprocessed_dir = os.path.join(output_dir, "../", "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True)
        splits_file = os.path.join(preprocessed_dir, "splits_final.pkl")
        do_splits(splits_file, myosaiq_train, myosaiq_val)

        if args.save_fig:
            g = plot_patients_distribution(
                all_patients_attributes, ["Infart", "MVO"], subsets=subsets
            )
            print("Saving distribution figure...")
            fig = g.fig
            fig.savefig(os.path.join(output_dir, "distribution.png"))


if __name__ == "__main__":
    main()
