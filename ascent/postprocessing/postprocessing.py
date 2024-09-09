import json

import cv2
import torch

from ascent import utils
import SimpleITK as sitk
import os
import numpy as np

from ascent.utils.file_and_folder_operations import subfiles

log = utils.get_pylogger(__name__)


def correct_innacurate_classif(liste):
    n = len(liste)
    if n == 0:
        return liste

    # Find index for first and last 1
    debut = -1
    fin = -1
    for i in range(n):
        if liste[i] == 1:
            debut = i
            break

    for i in range(n - 1, -1, -1):
        if liste[i] == 1:
            fin = i
            break

    if debut != -1 and fin != -1 and debut < fin:
        for i in range(debut, fin + 1):
            if liste[i] == 0:
                liste[i] = 1

    return liste


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


def get_majority_class(neighbors):
    neighbors = neighbors[neighbors != 255]
    if len(neighbors) == 0:
        return 0
    return np.bincount(neighbors).argmax()


def load_segmentation(file):
    """
        Read and convert nii.gz file to a NumPy array.
        Return the Image, NumPy array and case identifier corresponding to the read image.
    """
    seg_image = sitk.ReadImage(file)
    seg = sitk.GetArrayFromImage(seg_image)
    case_identifier = os.path.split(file)[-1].split(".nii.gz")[0]
    return seg_image, seg, case_identifier


def save_cleaned_segmentation(cleaned_seg, seg_image, results_path, case_identifier):
    """
            Convert NumPy array of the corrected segmentation into an Image and save it as .nii.gz.
            Origin, spacing and direction of the Image is base on the Image of not corrected segmentation.
    """
    new_seg_image = sitk.GetImageFromArray(cleaned_seg)
    new_seg_image.SetOrigin(seg_image.GetOrigin())
    new_seg_image.SetSpacing(seg_image.GetSpacing())
    new_seg_image.SetDirection(seg_image.GetDirection())
    sitk.WriteImage(new_seg_image, os.path.join(results_path, case_identifier + ".nii.gz"))


class SegPostprocessor:
    """
        Postprocessor class that takes segmentation result stored in Result_Folder,
        applies postprocessing and return segmentation result of the post processed datas.
    """

    def __init__(
            self,
            segmentation_path: str,
            verbose: bool = False,
    ) -> None:
        """Initialize class instance.

        Args:
            segmentation_path: Path to the segmentation results.
            verbose: Whether to log the preprocessing message.
        """

        self.segmentation_path = segmentation_path

    """def morphological_cleaning_2d(self, results_path, num_classes=5, kernel_size=5):

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for file in subfiles(self.segmentation_path, suffix=".nii.gz"):

            seg_image = sitk.ReadImage(file)
            seg = sitk.GetArrayFromImage(seg_image)

            case_identifier = os.path.split(file)[-1].split(".nii.gz")[0]

            cleaned_seg = seg.copy()

            for slice_idx in range(len(seg)):

                class_seg = (seg[slice_idx] == 0).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(class_seg)

                max_label = 1 + np.argmax(np.bincount(labels.flat)[1:])

                for i in range(1, num_labels):
                    if i != max_label:
                        component_seg = (labels == i)
                        seg[slice_idx][component_seg] = 255

                        dilated_seg = cv2.dilate(component_seg.astype(np.uint8), np.ones((3, 3), np.uint8))
                        neighbors = seg[slice_idx][dilated_seg.astype(bool)]

                        majority_class = get_majority_class(neighbors)
                        cleaned_seg[slice_idx][component_seg] = majority_class

                for cls in range(1, num_classes):
                    class_seg = (seg[slice_idx] == cls).astype(np.uint8)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_seg, connectivity=8)

                    for i in range(1, num_labels):
                        if stats[i, cv2.CC_STAT_AREA] < 6:
                            component_seg = (labels == i)
                            seg[slice_idx][component_seg] = 255

                            dilated_seg = cv2.dilate(component_seg.astype(np.uint8), np.ones((3, 3), np.uint8))
                            neighbors = seg[slice_idx][dilated_seg.astype(bool)]

                            majority_class = get_majority_class(neighbors)
                            cleaned_seg[slice_idx][component_seg] = majority_class

            new_seg_image = sitk.GetImageFromArray(cleaned_seg)

            orig = seg_image.GetOrigin()
            spacing = seg_image.GetSpacing()
            dir = seg_image.GetDirection()
            new_seg_image.SetOrigin(orig)
            new_seg_image.SetSpacing(spacing)
            new_seg_image.SetDirection(dir)

            sitk.WriteImage(new_seg_image, os.path.join(results_path, case_identifier + ".nii.gz"))
            
            
        def morphological_cleaning_3d(self, results_path, num_classes=5, min_component_size=6):

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for file in subfiles(self.segmentation_path, suffix=".nii.gz"):

            seg_image = sitk.ReadImage(file)
            seg = sitk.GetArrayFromImage(seg_image)

            case_identifier = os.path.split(file)[-1].split(".nii.gz")[0]

            cleaned_seg = seg.copy()

            for slice_idx in range(len(seg)):

                class_seg = (seg[slice_idx] == 0).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(class_seg)

                max_label = 1 + np.argmax(np.bincount(labels.flat)[1:])

                for i in range(1, num_labels):
                    if i != max_label:
                        component_seg = (labels == i)
                        seg[slice_idx][component_seg] = 255

                        dilated_seg = cv2.dilate(component_seg.astype(np.uint8), np.ones((3, 3), np.uint8))
                        neighbors = seg[slice_idx][dilated_seg.astype(bool)]

                        majority_class = get_majority_class(neighbors)
                        cleaned_seg[slice_idx][component_seg] = majority_class

            for cls in range(1, num_classes):
                class_seg = (seg == cls).astype(np.uint8)
                sitk_class_seg = sitk.GetImageFromArray(class_seg)

                cc = sitk.ConnectedComponent(sitk_class_seg)
                stats = sitk.LabelShapeStatisticsImageFilter()
                stats.Execute(cc)

                for label in stats.GetLabels():
                    if stats.GetPhysicalSize(label) < min_component_size:
                        component_seg = (sitk.GetArrayFromImage(cc) == label).astype(np.uint8)
                        sitk_component_seg = sitk.GetImageFromArray(component_seg)
                        dilated_seg = sitk.BinaryDilate(sitk_component_seg, [1, 1, 1])
                        dilated_array = sitk.GetArrayFromImage(dilated_seg)
                        neighbors = seg[dilated_array.astype(bool)]

                        majority_class = get_majority_class(neighbors)
                        cleaned_seg[component_seg.astype(bool)] = majority_class

            new_seg_image = sitk.GetImageFromArray(cleaned_seg)

            orig = seg_image.GetOrigin()
            spacing = seg_image.GetSpacing()
            dir = seg_image.GetDirection()
            new_seg_image.SetOrigin(orig)
            new_seg_image.SetSpacing(spacing)
            new_seg_image.SetDirection(dir)

            sitk.WriteImage(new_seg_image, os.path.join(results_path, case_identifier + ".nii.gz"))"""

    def morphological_cleaning(self, results_path, num_classes=5, min_component_size=6):
        """
            Implements a morphological cleaning process
        """

        # Creating the directory specified by results_path it does not exist.
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for file in subfiles(self.segmentation_path, suffix=".nii.gz"):
            seg_image, seg, case_identifier = load_segmentation(file)

            cleaned_seg = seg.copy()
            num_slices_mvo = 0

            # 3D cleaning for Inf and MVO
            for cls in range(3, num_classes):
                cleaned_seg = self._clean_class_components(seg, cleaned_seg, cls, min_component_size)

            # Clean by slice for other classes
            for slice_idx in range(len(seg)):
                cleaned_seg = self._clean_class_components2D(cleaned_seg, cleaned_seg, slice_idx, 2, min_component_size=10)
                for cls in range(0, 1):
                    cleaned_seg = self._clean_slice(cleaned_seg, cleaned_seg, slice_idx, cl=0)

                if np.sum((cleaned_seg[slice_idx] == 4).astype(np.uint8)) != 0:
                    num_slices_mvo += 1

            # Replace MVO by infarct if only detected in one slice
            if num_slices_mvo == 1:
                print(file)
                cleaned_seg[cleaned_seg == 4] = 3
            save_cleaned_segmentation(cleaned_seg, seg_image, results_path, case_identifier)

    def _clean_slice(self, seg, cleaned_seg, slice_idx, cl=0):
        """
            Cleans a single slice of the segmentation by removing smaller connected 2D components
            that are not the largest one, and reassigning their class based on the majority
            class of their neighbors.
        """

        class_seg = (seg[slice_idx] == cl).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(class_seg)
        max_label = 1 + np.argmax(np.bincount(labels.flat)[1:])

        for i in range(1, num_labels):
            if i != max_label:
                component_seg = (labels == i)
                seg[slice_idx][component_seg] = 255
                dilated_seg = cv2.dilate(component_seg.astype(np.uint8), np.ones((3, 3), np.uint8))
                neighbors = seg[slice_idx][dilated_seg.astype(bool)]
                majority_class = get_majority_class(neighbors)
                cleaned_seg[slice_idx][component_seg] = majority_class
        return cleaned_seg

    def _clean_class_components2D(self, seg, cleaned_seg, slice_idx, cls, min_component_size):
        """
            Cleans the slice 2D segmentation by removing small connected 3D components of a specified
            class and reassigning their class based on the majority class of their neighbors.
        """

        class_seg = (seg[slice_idx] == cls).astype(np.uint8)
        sitk_class_seg = sitk.GetImageFromArray(class_seg)
        cc = sitk.ConnectedComponent(sitk_class_seg)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)

        for label in stats.GetLabels():
            if stats.GetPhysicalSize(label) < min_component_size:
                component_seg = (sitk.GetArrayFromImage(cc) == label).astype(np.uint8)
                sitk_component_seg = sitk.GetImageFromArray(component_seg)
                dilated_seg = sitk.BinaryDilate(sitk_component_seg, [1, 1])
                dilated_array = sitk.GetArrayFromImage(dilated_seg)
                neighbors = seg[slice_idx][dilated_array.astype(bool)]
                majority_class = get_majority_class(neighbors)
                if majority_class != 0:
                    cleaned_seg[slice_idx][component_seg.astype(bool)] = majority_class
        return cleaned_seg


    def _clean_class_components(self, seg, cleaned_seg, cls, min_component_size):
        """
            Cleans the entire 3D segmentation by removing small connected 3D components of a specified
            class and reassigning their class based on the majority class of their neighbors.
        """

        class_seg = (seg == cls).astype(np.uint8)
        sitk_class_seg = sitk.GetImageFromArray(class_seg)
        cc = sitk.ConnectedComponent(sitk_class_seg)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)

        for label in stats.GetLabels():
            if stats.GetPhysicalSize(label) < min_component_size:
                component_seg = (sitk.GetArrayFromImage(cc) == label).astype(np.uint8)
                sitk_component_seg = sitk.GetImageFromArray(component_seg)
                dilated_seg = sitk.BinaryDilate(sitk_component_seg, [1, 1, 1])
                dilated_array = sitk.GetArrayFromImage(dilated_seg)
                neighbors = seg[dilated_array.astype(bool)]
                majority_class = get_majority_class(neighbors)
                cleaned_seg[component_seg.astype(bool)] = majority_class

        return cleaned_seg

    def remove_over_seg_slices_from_gt(self, results_path, gt_folder):
        """
            Implements a cleaning process that remove over segmented slices based on gt.
        """

        # Creating the directory specified by results_path it does not exist.
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for file in subfiles(self.segmentation_path, suffix=".nii.gz"):
            seg_image, seg, case_identifier = load_segmentation(file)
            gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_folder, case_identifier + ".nii.gz")))

            cleaned_seg = seg.copy()
            for slice_idx in range(len(seg)):
                is_heart_gt = (gt[slice_idx] > 0).astype(np.uint8)

                if np.sum(is_heart_gt) == 0:
                    if np.sum(seg[slice_idx]) == 0:
                        print("Slice", slice_idx, "from the volume", case_identifier, "already classify as not heart.")
                    else:
                        cleaned_seg[slice_idx] = 0
                        print("Removing slice", slice_idx, "from the volume", case_identifier)

            save_cleaned_segmentation(cleaned_seg, seg_image, results_path, case_identifier)

    def remove_over_seg_slices_from_rule(self, results_path):
        """
            Implements a cleaning process that remove over segmented slices for MYOSAIQ dataset.
            Based on a 40% rule for the proportion of MYO compared to the LV.
        """

        # Creating the directory specified by results_path it does not exist.
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for file in subfiles(self.segmentation_path, suffix=".nii.gz"):
            seg_image, seg, case_identifier = load_segmentation(file)

            cleaned_seg = seg.copy()

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
                            cleaned_seg[slice_idx] = 0

                else:
                    print("Slice", slice_idx, "from the volume", case_identifier, "does not contains LV")

            save_cleaned_segmentation(cleaned_seg, seg_image, results_path, case_identifier)

    def remove_over_seg_slices_from_classif(self, results_path, classif_path):
        """
            Implements a cleaning process that remove over segmented slices based a classification bootstrap pipeline.
            The classification is done by a bootstrap method with 5 classification model.
        """

        # Creating the directory specified by results_path it does not exist.
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for file in subfiles(self.segmentation_path, suffix=".nii.gz"):
            seg_image, seg, case_identifier = load_segmentation(file)
            cleaned_seg = seg.copy()

            classif_pred = 0
            for path in classif_path:
                classif_pred += torch.tensor(json.load(open(os.path.join(path, case_identifier + ".json")))["class"])

            # Implements bootstrap method for classification
            bootstrap_min = torch.zeros_like(classif_pred)
            bootstrap_min[0] = 3
            bootstrap_min[1] = 3
            bootstrap_min[2] = 2
            bootstrap_min[3] = 2

            bootstrap_min[-1] = 4
            bootstrap_min[-2] = 3
            bootstrap_min[-3] = 3
            bootstrap_min[-4] = 2

            bootstrap_pred = torch.zeros_like(classif_pred)
            bootstrap_pred[classif_pred > bootstrap_min] = 1

            classif_res = correct_innacurate_classif(bootstrap_pred)

            """classif_res = torch.tensor(
                json.load(open(os.path.join(classif_path[0], case_identifier + ".json")))["class"])
            classif_res = correct_innacurate_classif(classif_res)"""

            for slice_idx in range(len(seg)):
                if classif_res[slice_idx] == 0:
                    if seg[slice_idx].sum() != 0:
                        print("Removing slice", slice_idx, "from the volume", case_identifier)
                        cleaned_seg[slice_idx] = 0
                    else:
                        print("Slice", slice_idx, "from the volume", case_identifier, "already classify as not heart.")

            save_cleaned_segmentation(cleaned_seg, seg_image, results_path, case_identifier)

    def main(self, classif_path, do_classif=True) -> None:
        results_path_classif = self.segmentation_path + "classif"
        results_path_cleaning = self.segmentation_path + f"clean_classif{do_classif}"

        if do_classif:

            self.remove_over_seg_slices_from_classif(results_path_classif, classif_path)

            self.segmentation_path = results_path_classif

        self.morphological_cleaning(results_path_cleaning)


if __name__ == "__main__":
    classif_path=[
            "../../../Challenge/EXPLICIT/logs/Classif_MYOSAIQ/nnUNet/2D/Fold_0/runs/2024-07-04_01-09-48/testing_raw",
            "../../../Challenge/EXPLICIT/logs/Classif_MYOSAIQ/nnUNet/2D/Fold_1/runs/2024-07-04_00-28-50/testing_raw",
            "../../../Challenge/EXPLICIT/logs/Classif_MYOSAIQ/nnUNet/2D/Fold_2/runs/2024-07-03_23-01-05/testing_raw",
            "../../../Challenge/EXPLICIT/logs/Classif_MYOSAIQ/nnUNet/2D/Fold_3/runs/2024-07-03_23-43-22/testing_raw",
            "../../../Challenge/EXPLICIT/logs/Classif_MYOSAIQ/nnUNet/2D/Fold_4/runs/2024-07-03_21-07-35/testing_raw"]

    classif_path = [
        "C:/Users/goujat/Documents/thesis/ASCENT/inference/classification/0/inference_raw",
        "C:/Users/goujat/Documents/thesis/ASCENT/inference/classification/1/inference_raw",
        "C:/Users/goujat/Documents/thesis/ASCENT/inference/classification/2/inference_raw",
        "C:/Users/goujat/Documents/thesis/ASCENT/inference/classification/3/inference_raw",
        "C:/Users/goujat/Documents/thesis/ASCENT/inference/classification/4/inference_raw"
    ]

    postProcessor_mean = SegPostprocessor("C:/Users/goujat/Documents/thesis/ASCENT/inference/stage3/classifTrue/mean_9/inference_raw")
    postProcessor_mean.main(classif_path, do_classif=True)