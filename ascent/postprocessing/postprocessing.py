import json

import cv2

from ascent import utils
import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt

log = utils.get_pylogger(__name__)


def has_myo_neighbor(point, slice_mask):
    x, y = point

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < slice_mask.shape[1] and 0 <= ny < slice_mask.shape[0]:
            if slice_mask[ny, nx] >= 2:  # Vérifier si le voisin est dans le myocarde
                return True
    return False


class SegPostprocessor:
    """Postprocessor class that takes segmentation result stored in Result_Folder,
    applies postprocessing and return segmentation result of the post processed datas.
    """

    def __init__(
            self,
            dataset_path: str,
            segmentation_path: str,
            verbose: bool = False,
    ) -> None:
        """Initialize class instance.

        Args:
            dataset_path: Path to the dataset.
            segmentation_path: Path to the segmentation results.
            verbose: Whether to log the preprocessing message.
        """

        self.gt_folder = os.path.join(dataset_path, "raw/labelsTr")
        self.segmentation_path = segmentation_path
        self.postprocess_path = os.path.join(self.segmentation_path, "../postprocessing")
        if not os.path.exists(self.postprocess_path):
            os.makedirs(self.postprocess_path)

    def remove_over_seg_slices_from_gt(self):
        results_path = os.path.join(self.postprocess_path, "remove_from_gt")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for file in os.listdir(self.segmentation_path):
            gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.gt_folder, file)))

            seg_image = sitk.ReadImage(os.path.join(self.segmentation_path, file))
            seg = sitk.GetArrayFromImage(seg_image)

            case_identifier = file.split(".nii.gz")[0]

            for slice_idx in range(len(seg)):
                is_heart_gt = (gt[slice_idx] > 0).astype(np.uint8)

                if np.sum(is_heart_gt) == 0:
                    seg[slice_idx] = 0
                    print("Removing slice", slice_idx, "from the volume", case_identifier)
            new_seg_image = sitk.GetImageFromArray(seg)

            orig = seg_image.GetOrigin()
            spacing = seg_image.GetSpacing()
            dir = seg_image.GetDirection()
            new_seg_image.SetOrigin(orig)
            new_seg_image.SetSpacing(spacing)
            new_seg_image.SetDirection(dir)

            sitk.WriteImage(new_seg_image, os.path.join(results_path, file))

    def remove_over_seg_slices_from_rule(self):
        results_path = os.path.join(self.postprocess_path, "remove_from_rule")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for file in os.listdir(self.segmentation_path):
            seg_image = sitk.ReadImage(os.path.join(self.segmentation_path, file))
            seg = sitk.GetArrayFromImage(seg_image)
            case_identifier = file.split(".nii.gz")[0]

            for slice_idx in range(len(seg)):
                slice_mask = seg[slice_idx]
                # Identify left ventricle
                lv = (slice_mask == 1).astype(np.uint8)

                if np.sum(lv) > 0:
                    # Find LV contours
                    contours, _ = cv2.findContours(lv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    largest_contour = max(contours, key=cv2.contourArea)
                    sum_contour_lv = len(largest_contour)
                    sum_contour_myo = 0

                    # Iterate over points of the contour
                    for point in largest_contour:
                        if has_myo_neighbor(point[0], slice_mask):
                            sum_contour_myo += 1

                    if sum_contour_lv != sum_contour_myo:
                        res = sum_contour_myo * 100 / sum_contour_lv

                        if res <= 40:
                            print("Removing slice", slice_idx, "from the volume", case_identifier)
                            seg[slice_idx] = 0
            new_seg_image = sitk.GetImageFromArray(seg)

            orig = seg_image.GetOrigin()
            spacing = seg_image.GetSpacing()
            dir = seg_image.GetDirection()
            new_seg_image.SetOrigin(orig)
            new_seg_image.SetSpacing(spacing)
            new_seg_image.SetDirection(dir)

            sitk.WriteImage(new_seg_image, os.path.join(results_path, file))

    def remove_over_seg_slices_from_classif(self):
        results_path = os.path.join(self.postprocess_path, "remove_from_classif")
        classif_path = "C:/Users/goujat/Documents/thesis/Challenge/EXPLICIT/logs/Classif_MYOSAIQ/nnUNet/2D/Fold_0/runs/2024-06-06_18-59-41/testing_raw"

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for file in os.listdir(self.segmentation_path):
            seg_image = sitk.ReadImage(os.path.join(self.segmentation_path, file))
            seg = sitk.GetArrayFromImage(seg_image)
            case_identifier = file.split(".nii.gz")[0]

            classif_res = json.load(open(os.path.join(classif_path, case_identifier + ".json")))["class"]

            for slice_idx in range(len(seg)):
                if classif_res[slice_idx] == 0:
                    print("Removing slice", slice_idx, "from the volume", case_identifier)
                    seg[slice_idx] = 0

            new_seg_image = sitk.GetImageFromArray(seg)

            orig = seg_image.GetOrigin()
            spacing = seg_image.GetSpacing()
            dir = seg_image.GetDirection()
            new_seg_image.SetOrigin(orig)
            new_seg_image.SetSpacing(spacing)
            new_seg_image.SetDirection(dir)

            sitk.WriteImage(new_seg_image, os.path.join(results_path, file))


if __name__ == "__main__":
    dataset_dir = "../../data/MYOSAIQ"
    segmentation_dir = "../../logs/MYOSAIQ/nnUNet/2D/Fold_0/runs/1000batch50lr0.01/testing_raw"
    postProcessor = SegPostprocessor(dataset_dir, segmentation_dir)
    postProcessor.remove_over_seg_slices_from_classif()
