import errno
import itertools
import json
import os
from typing import Optional, Sequence, Union

import numpy as np
from scipy import ndimage
from scipy.ndimage import center_of_mass
from scipy.signal import find_peaks
from skimage.measure import find_contours
from skimage.morphology import disk

from ascent.utils.type_definitions import PathLike


def cart2pol(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts (x,y) cartesian coordinates to (theta,rho) polar coordinates.

    Notes:
        - Made to mimic Matlab's `cart2pol` function: https://www.mathworks.com/help/matlab/ref/pol2cart.html

    Args:
        x: x component of cartesian coordinates.
        y: y component of cartesian coordinates.

    Returns:
        (theta,rho) polar coordinates corresponding to the input cartesian coordinates.

    Example:
        >>> x = np.array([5, 3.5355, 0, -10])
        >>> y = np.array([0, 3.5355, 10, 0])
        >>> cart2pol(x,y)
        (array([0, 0.7854, 1.5708, 3.1416]), array([5.0000, 5.0000, 10.0000, 10.0000]))
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, rho


def _endo_epi_base(
    structure_mask: np.ndarray,
    contour: np.ndarray,
    smooth: bool = False,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Finds the left/right markers at the base of the endo/epi using Shi-Tomasi corner detection
    algorithm.

    Args:
        segmentation: (H, W), Segmentation map.
        labels: Labels of the classes that are part of the endocardium/epicardium.

    Returns:
        Coordinates of the left/right markers at the base of the endo/epi.
    """

    if smooth:
        structure_mask = ndimage.binary_closing(structure_mask, structure=disk(3), iterations=5)

    # Shift the grid center to the center of mass
    contour_shifted = contour - center_of_mass(structure_mask)

    # Convert contour points from Cartesian grid to Polar grid
    theta, rho = cart2pol(contour_shifted[:, 1], contour_shifted[:, 0])

    # Sort theta and rho arrays
    theta_sorted_idx = theta.argsort()
    rho_sorted = rho[theta_sorted_idx]
    theta_sorted = theta[theta_sorted_idx]

    # Compute appropriate distance that corresponds to [pi/4, 2.1*pi/4]
    angle_interval = [np.pi / 4, 2.1 * np.pi / 4]
    min_distance_idx = np.argmin(np.abs(theta_sorted - angle_interval[0]))
    max_distance_idx = np.argmin(np.abs(theta_sorted - angle_interval[1]))
    distance = max_distance_idx - min_distance_idx

    if debug:
        plt.scatter(theta, rho, c="b", marker="o", s=4)
        plt.show()

    # Detect peaks that correspond to endo/epi base and apex
    peaks, _ = find_peaks(rho_sorted, distance=distance, prominence=0.5)

    # Keep only peaks that have positive angle to eliminate apex peak
    peaks = peaks[theta_sorted[peaks] > 0]

    while len(peaks) < 2:
        distance = distance - 5

        # Detect peaks that correspond to endo/epi base and apex
        peaks, _ = find_peaks(rho_sorted, distance=distance, prominence=0.5)

        # Keep only peaks that have positive angle to eliminate apex peak
        peaks = peaks[theta_sorted[peaks] > 0]

    if len(peaks) >= 2:
        peaks = peaks[:2]

    # Retrieve the corner indices
    corner_idx = theta_sorted_idx[peaks][::-1]

    left_corner, right_corner = contour[corner_idx[0], :], contour[corner_idx[1], :]

    if debug:
        plt.imshow(structure_mask)
        plt.scatter(left_corner[1], left_corner[0], c="r", marker="o", s=1)
        plt.scatter(right_corner[1], right_corner[0], c="r", marker="o", s=1)
        plt.show()
    return left_corner, right_corner


def _endo_epi_contour(segmentation: np.ndarray, labels: Union[int, Sequence[int]]) -> np.ndarray:
    """Lists points on the contour of the endo/epi (excluding the base), from the left of the base
    to its right.

    Args:
        segmentation: (H, W), Segmentation map.
        labels: Labels of the classes that are part of the endocardium/epicardium.

    Returns:
        Coordinates of points on the contour of the endo/epi (excluding the base), from the left of the base to its
        right.
    """
    structure_mask = np.isin(segmentation, labels)

    # Extract all the points on the contour of the structure of interest
    # Use `level=0.9` to force the contour to be closer to the structure of interest than the background
    contour = find_contours(structure_mask, level=0.9)[0]

    # Identify the left/right markers at the base of the endo/epi
    left_corner, right_corner = _endo_epi_base(structure_mask, contour, smooth=False, debug=False)

    # Shift the contour so that they start at the left corner
    # To detect the contour coordinates that match the corner, we use the closest match since skimage's
    # `find_contours` coordinates are interpolated between pixels, so they won't match exactly corner coordinates
    dist_to_left_corner = np.linalg.norm(left_corner - contour, axis=1)
    left_corner_contour_idx = np.argmin(dist_to_left_corner)
    contour = np.roll(contour, -left_corner_contour_idx, axis=0)

    # Filter the full contour to discard points along the base
    # We implement this by slicing the contours from the left corner to the right corner, since the contour returned
    # by skimage's `find_contours` is oriented clockwise
    dist_to_right_corner = np.linalg.norm(right_corner - contour, axis=1)
    right_corner_contour_idx = np.argmin(dist_to_right_corner)
    contour_without_base = contour[: right_corner_contour_idx + 1]

    return contour_without_base


def endo_epi_control_points(
    segmentation: np.ndarray,
    labels: Union[int, Sequence[int]],
    num_control_points: int,
    voxelspacing: tuple[float, float] = None,
) -> np.ndarray:
    """Lists uniformly distributed control points along the contour of the endocardium/epicardium.

    Args:
        segmentation: (H, W), Segmentation map.
        labels: Labels of the classes that are part of the endocardium/epicardium.
        num_control_points: Number of control points to sample along the contour of the endocardium/epicardium.
        voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

    Returns:
        Coordinates of the control points along the contour of the endocardium/epicardium.
    """
    if voxelspacing is None:
        voxelspacing = (1, 1)
    voxelspacing = np.array(voxelspacing)

    # Find the points along the contour of the endo/epi excluding the base
    contour = _endo_epi_contour(segmentation, labels)

    # Round the contour's coordinates, so they don't fall between pixels anymore
    contour = contour.round().astype(int)

    # Compute the geometric distances between each point along the contour and the previous one.
    # This allows to then simply compute the cumulative distance from the left corner to each contour point
    contour_dist_to_prev = [0.0] + [
        np.linalg.norm((p1 - p0) * voxelspacing) for p0, p1 in itertools.pairwise(contour)
    ]
    contour_cum_dist = np.cumsum(contour_dist_to_prev)

    # Select points along the contour that are equidistant along the contour (by selecting points that are closest
    # to where steps of `perimeter / num_control_points` would expect to find points)
    control_points_step = np.linspace(0, contour_cum_dist[-1], num=num_control_points)
    control_points_indices = [
        np.argmin(np.abs(point_cum_dist - contour_cum_dist))
        for point_cum_dist in control_points_step
    ]
    return np.roll(contour[control_points_indices], 1, 1)


def extract_control_points_and_save_as_json(
    seg_path: PathLike,
    output_folder: PathLike,
    num_points: int = 15,
    json_name: Optional[str] = None,
) -> None:
    """Extract endocardium and/or epicardium control points from segmentation and export the point
    coordinates to a .json file.

    Args:
        seg_path: Path to segmentation file (.nii.gz).
        output_folder: Path of output folder to save the json file.
        num_points: Number of control points to extract.
        json_name: Name of the json file to save. Segmentation filename will be used if set as None.

    Raises:
        FileNotFoundError: When segmentation file is not found.
        NotImplementedError: When the segmentation file is not in nifti format.
    """
    if not os.path.isfile(seg_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), seg_path)

    if not seg_path.endswith(".nii.gz"):
        raise NotImplementedError("Currently, only nifti segmentation files are supported.")

    seg_itk = sitk.ReadImage(seg_path)
    seg_array = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
    spacing = seg_itk.GetSpacing()

    endo_points = []
    epi_points = []
    dummy_points = []

    if not len(seg_array.shape) == 3:
        seg_array = seg_array[
            None,
        ]
        spacing = spacing
    else:
        spacing = spacing[:-1]

    for _, seg in enumerate(seg_array):
        endo_points.append(
            (endo_epi_control_points(seg, 1, num_points, spacing) * spacing).tolist()
        )

        if 2 in np.unique(seg):
            epi_points.append(
                (endo_epi_control_points(seg, [1, 2], num_points, spacing) * spacing).tolist()
            )
        else:
            epi_points.append([])

        dummy_points.append([])

    os.makedirs(output_folder, exist_ok=True)

    if json_name is None:
        json_name = os.path.basename(seg_path)[:-7] + ".json"
    else:
        json_name = json_name + ".json"

    json_path = os.path.join(output_folder, json_name)

    json_dict = {}
    json_dict["contourCheck"] = 0
    json_dict["imageQuality"] = 0
    json_dict["ecg"] = []
    json_dict["left_ventricle_endo"] = endo_points
    json_dict["left_ventricle_epi"] = epi_points
    json_dict["right_ventricle"] = dummy_points

    if os.path.isfile(json_path):
        os.remove(json_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, separators=(",", ":"))


def nifti2mhd(
    nifti_file: PathLike, output_folder: PathLike, mhd_name: Optional[str] = None
) -> None:
    """Convert nifti file to raw mhd file.

    Args:
        nifti_file: Path to nifti file.
        output_folder: Path to output folder for saving.
        mhd_name: Filename to save the mhd file.
    """
    if not seg_path.endswith(".nii.gz"):
        raise ValueError(f"{nifti_file} is not a nifti file.")

    if not os.path.isfile(seg_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), seg_path)

    output_folder = os.path.join(output_folder, os.path.basename(seg_path)[:8])
    os.makedirs(output_folder, exist_ok=True)

    if mhd_name is None:
        mhd_name = os.path.basename(seg_path)[:8] + ".mhd"
    else:
        mhd_name = mhd_name + ".mhd"

    nifti_itk = sitk.ReadImage(nifti_file)
    nifti_array = sitk.GetArrayFromImage(nifti_itk)
    spacing = nifti_itk.GetSpacing()
    mhd_itk = sitk.GetImageFromArray(nifti_array)
    mhd_itk.SetSpacing(spacing)
    sitk.WriteImage(mhd_itk, os.path.join(output_folder, mhd_name))


if __name__ == "__main__":
    import SimpleITK as sitk
    from matplotlib import pyplot as plt

    from ascent.utils.visualization import imagesc, overlay_mask_on_image

    bmode_path = "C:/Users/ling/Desktop/A3C-nnUNet-results/nifti/bmode/0027_A3C_bmode.nii.gz"
    seg_path = "C:/Users/ling/Desktop/A3C-nnUNet-results/nifti/post_processed_masks/0027_A3C_post_mask.nii.gz"
    output_folder = "C:/Users/ling/Desktop/new_A3C_data/control_points"
    json_name = os.path.basename(seg_path)[:8]
    # extract_control_points_and_save_as_json(seg_path, output_folder, json_name=json_name)
    # nifti2mhd(bmode_path, output_folder)
    frame = 20

    seg_itk = sitk.ReadImage(seg_path)
    seg_array = sitk.GetArrayFromImage(seg_itk)

    bmode_itk = sitk.ReadImage(bmode_path)
    bmode_array = sitk.GetArrayFromImage(bmode_itk)

    overlaid = overlay_mask_on_image(bmode_array, seg_array)

    spacing = seg_itk.GetSpacing()

    for frame in range(5):
        endo_points = endo_epi_control_points(seg_array[frame], 1, 12, spacing[:-1])
        epi_points = endo_epi_control_points(seg_array[frame], [1, 2], 12, spacing[:-1])

        plt.figure(figsize=(18, 6), dpi=300)
        ax = plt.subplot(1, 3, 1)
        imagesc(ax, overlaid[frame], show_colorbar=False)
        ax.scatter(endo_points[:, 0], endo_points[:, 1], c="r", s=4)
        ax.scatter(epi_points[:, 0], epi_points[:, 1], c="b", s=4)
        plt.show()
