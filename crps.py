import os

import numpy as np
import nibabel as nib
import scipy.stats as stats
import pandas as pd
from typing import Union
from ascent.utils.file_and_folder_operations import subfiles, subdirs


def compute_volume(
    mask: np.ndarray, voxel_spacing: Union[list[float], tuple[float, ...]]
) -> float:
    """
    Compute the volume of a segmentation mask.

    Args:
        mask: Binary mask of the segmented region.
        voxel_spacing: Tuple or list of voxel spacing in mm.

    Returns:
        Calculated volume in mL.
    """
    # Compute voxel volume in mL
    voxel_volume = np.prod(voxel_spacing) / 1e3
    return float(np.sum(mask) * voxel_volume)


def compute_statistics(volumes: list[float]) -> tuple[float, float, tuple[float, float]]:
    """
    Compute the mean, standard deviation, and range of a list of volumes.

    Args:
        volumes: List of volumes.

    Returns:
        Mean, standard deviation, and range (min, max) of the volumes.
    """
    mean_volume = np.mean(volumes, dtype=np.float32)
    std_volume = np.std(volumes)
    range_volume = (np.min(volumes), np.max(volumes))
    return mean_volume, std_volume, range_volume


def compute_cdf(mean_volume: float, std_volume: float, num_points: int = 600) -> np.ndarray:
    """
    Compute the cumulative distribution function (CDF) for a normal distribution.

    Args:
        mean_volume: Mean of the volumes.
        std_volume: Standard deviation of the volumes.
        num_points: Number of points for the CDF. Default is 600.

    Returns:
        np.ndarray: Array of CDF values.
    """
    volume_range = np.arange(num_points)
    if std_volume == 0:
        cdf_values = np.zeros(num_points).astype(float)
        cdf_values[volume_range >= mean_volume] = 1.0
    else:
        cdf_values = stats.norm.cdf(volume_range, loc=mean_volume, scale=std_volume).astype(float)
    return cdf_values


if __name__ == "__main__":
    # Set paths and parameters
    base_path = "C:/Users/goujat/Documents/thesis/ASCENT/inference/stage3/classifTrue"  # Base path containing the 10 folders
    save_path = "C:/Users/goujat/Documents/thesis/ASCENT/inference/stage3/stats_myosaiq"  # Path to save the CSV files
    num_points = 600  # Number of points for the CDF

    # Prepare data structures
    volume_data: dict[str, dict[int, list[float]]] = {}

    os.makedirs(save_path, exist_ok=True)

    folders = subdirs(base_path)

    # Iterate through folders and volumes
    for folder in folders:
        if folder.endswith("mean"):
            print(folder)
        else:
            folder_path = os.path.join(folder, "inference_rawclean_classifTrue")
            image_paths = subfiles(folder_path, suffix=".nii.gz")
            for image_path in image_paths:
                image_id = os.path.basename(image_path)[:-7]
                nii = nib.load(image_path)
                data = nii.get_fdata()
                voxel_spacing = nii.header["pixdim"][1:4]

                unique_classes = np.unique(data)
                for class_id in unique_classes:
                    if class_id == 0:  # Ignore background
                        continue
                    if class_id == 1:
                        mask = data == class_id
                    else:
                        mask = data >= class_id
                    volume = compute_volume(mask, voxel_spacing)
                    if image_id not in volume_data:
                        volume_data[image_id] = {}
                    if class_id not in volume_data[image_id]:
                        volume_data[image_id][class_id] = []
                    volume_data[image_id][class_id].append(volume)

    # Compute statistics and CDF, and save to CSV
    classes = ["BG", "LV", "MYO", "MI", "MVO"]
    for class_id in range(1, len(classes) + 1):
        class_volumes = {
            img_id: volumes[class_id]
            for img_id, volumes in volume_data.items()
            if class_id in volumes
        }
        if not class_volumes:
            continue
        ids, all_volumes = zip(*[(img_id, vols) for img_id, vols in class_volumes.items()])
        mean_volumes = []
        std_volumes = []
        range_volumes = []
        cdf_values = []

        results = []
        for vol_id, volumes in zip(ids, all_volumes):
            mean, std, range_vol = compute_statistics(volumes)
            cdf_values = compute_cdf(mean, std, num_points=num_points)

            # Prepare row for DataFrame
            row = {
                "ID": vol_id,
                "VOL": mean,
            }  # For challenge submissions
            # row = {
            #     "Id": vol_id,
            #     "MeanVol": mean,
            #     "StdVol": std,
            #     "RangeVol": range_vol[1] - range_vol[0],
            # }  # For saving more info
            for i in range(num_points):
                row[f"P{i}"] = cdf_values[i]
            results.append(row)

        # Create DataFrame from the results
        df = pd.DataFrame(results)

        # Save to CSV
        csv_path = os.path.join(save_path, f"{classes[class_id]}_volumes.csv")
        df.to_csv(csv_path, index=False)
