import errno
import functools
import itertools
import json
import os
from typing import Callable, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.measure import find_contours
from skimage.morphology import disk

from ascent.utils.type_definitions import PathLike
from ascent.utils.visualization import array2gif, overlay_mask_on_image


def cart2pol(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def _endo_epi_contour(
    segmentation: np.ndarray,
    labels: int | Sequence[int],
    base_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Lists points on the contour of the endo/epi (excluding the base), from the left of the base
    to its right.

    Args:
        segmentation: (H, W), Segmentation map.
        labels: Labels of the classes that are part of the endocardium/epicardium.
        base_fn: Function that identifies the left and right corners at the base of the endocardium/epicardium in a
            segmentation mask.

    Returns:
        Coordinates of points on the contour of the endo/epi (excluding the base), from the left of the base to its
        right.
    """
    structure_mask = np.isin(segmentation, labels)

    # Identify the left/right markers at the base of the endo/epi
    left_corner, right_corner = base_fn(segmentation)

    # Extract all the points on the contour of the structure of interest
    # Use `level=0.9` to force the contour to be closer to the structure of interest than the background
    contour = find_contours(structure_mask, level=0.9)[0]

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


def _endo_epi_apex(
    segmentation: np.ndarray, labels: int | Sequence[int]
) -> np.ndarray | Tuple[np.ndarray, float]:
    """Identifies the apex of the endo/epi as the point farthest from the center of the endo/epi.

    Args:

    Returns:
        The coordinates of the endocardium/epicardium apex.
    """
    structure_mask = np.isin(segmentation, labels)

    # Find the center point of the segmentation
    center = ndimage.center_of_mass(structure_mask)

    # Extract all the points on the contour of the structure of interest
    # Use `level=0.9` to force the contour to be closer to the structure of interest than the background
    contour = find_contours(structure_mask, level=0.9)[0]

    # Shift the grid center to the center of the mask
    contour_centered = contour - center

    # Convert contour points from Cartesian grid to Polar grid
    theta, rho = cart2pol(contour_centered[:, 1], contour_centered[:, 0])

    # Sort theta and rho arrays
    theta_sort_indices = theta.argsort()
    theta_sorted, rho_sorted = theta[theta_sort_indices], rho[theta_sort_indices]

    # Smooth the signal to avoid finding peaks for small localities
    rho_sorted_gaussian_filtered = gaussian_filter1d(rho_sorted, len(contour) * 5e-2)

    # Detect peaks that correspond to endo/epi base and apex
    peaks, properties = find_peaks(rho_sorted_gaussian_filtered, height=0)

    # Discard base peaks by only keeping peaks found in the upper half of the mask
    # (by discarding peaks found where theta < 0)
    upper_half_peaks_mask = theta_sorted[peaks] < 0
    peaks = peaks[upper_half_peaks_mask]

    ####################
    # Debugging code to display contour curve in polar coordinates
    #####################
    if "DEBUG" in os.environ:
        import seaborn as sns
        from matplotlib import pyplot as plt

        plot_data = pd.melt(
            pd.DataFrame(
                {
                    "theta": theta_sorted,
                    "none": rho_sorted,
                    "gaussian": rho_sorted_gaussian_filtered,
                }
            ),
            id_vars=["theta"],
            value_vars=["none", "gaussian"],
            var_name="filter",
            value_name="rho",
        )
        with sns.axes_style("darkgrid"):
            plot = sns.lineplot(data=plot_data, x="theta", y="rho", style="filter")

        # Annotate the peaks with their respective index
        for peak_idx, peak in enumerate(peaks):
            plot.annotate(
                f"{peak_idx}",
                (theta_sorted[peak], rho_sorted[peak]),
                xytext=(1, 4),
                textcoords="offset points",
            )

        # Plot lines pointing to the peaks to make them more visible
        plot.vlines(
            x=theta_sorted[peaks],
            ymin=rho[theta > 0].min(),
            ymax=rho_sorted[peaks],
            linestyles="dashed",
        )

        plt.show()
    ####################
    # End of debugging block
    ####################

    if not len(peaks):
        raise RuntimeError("Unable to identify the apex of the endo/epi.")

    # Extract the heights of each peak and discard the values associated with discarded peaks
    peaks_heights = properties["peak_heights"]
    peaks_heights = peaks_heights[upper_half_peaks_mask]

    # Keep only the highest peak as the peak corresponding to the apex
    peak = peaks[peaks_heights.argmax()]

    # Map the index of the peak in polar coordinates back to the indices in the list of contour points
    contour_idx = theta_sort_indices[peak]
    apex = contour[contour_idx]

    ####################
    # Debugging code to display the selected corners on the segmentation mask
    #####################
    if "DEBUG" in os.environ:
        plt.imshow(structure_mask)
        plt.scatter(apex[1], apex[0], c="r", marker="o", s=3)
        plt.show()
    ####################
    # End of debugging block
    ####################

    return apex


def endo_base(
    segmentation: np.ndarray, lv_labels: int | Sequence[int], myo_labels: int | Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the left/right markers at the base of the endocardium.

    Args:
        segmentation: (H, W), Segmentation map.
        lv_labels: Labels of the classes that are part of the left ventricle.
        myo_labels: Labels of the classes that are part of the left ventricle.

    Returns:
        Coordinates of the left/right markers at the base of the endocardium.
    """
    struct = ndimage.generate_binary_structure(2, 2)
    left_ventricle = np.isin(segmentation, lv_labels)
    myocardium = np.isin(segmentation, myo_labels)
    others = ~(left_ventricle + myocardium)
    dilated_myocardium = ndimage.binary_dilation(myocardium, structure=struct)
    dilated_others = ndimage.binary_dilation(others, structure=struct)
    y_coords, x_coords = np.nonzero(left_ventricle * dilated_myocardium * dilated_others)

    if (num_markers := len(y_coords)) < 2:
        raise RuntimeError(
            f"Identified {num_markers} marker(s) at the edges of the left ventricle/myocardium frontier. We need "
            f"to identify at least 2 such markers to determine the base of the left ventricle."
        )

    if np.all(x_coords == x_coords.mean()):
        # Edge case where the base points are aligned vertically
        # Divide frontier into bottom and top halves.
        coord_mask = y_coords > y_coords.mean()
        left_point_idx = y_coords[coord_mask].argmin()
        right_point_idx = y_coords[~coord_mask].argmax()
    else:
        # Normal case where there is a clear divide between left and right markers at the base
        # Divide frontier into left and right halves.
        coord_mask = x_coords < x_coords.mean()
        left_point_idx = y_coords[coord_mask].argmax()
        right_point_idx = y_coords[~coord_mask].argmax()
    return (
        np.array([y_coords[coord_mask][left_point_idx], x_coords[coord_mask][left_point_idx]]),
        np.array([y_coords[~coord_mask][right_point_idx], x_coords[~coord_mask][right_point_idx]]),
    )


def epi_base(
    segmentation: np.ndarray, labels: int | Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the left/right markers at the base of the epicardium.

    Args:
        segmentation: (H, W), Segmentation map.
        labels: Labels of the classes that are part of the epicardium.

    Returns:
        Coordinates of the left/right markers at the base of the epicardium.
    """
    structure_mask = np.isin(segmentation, labels)

    # Find the center point of the segmentation
    center = ndimage.center_of_mass(structure_mask)

    # Extract all the points on the contour of the structure of interest
    # Use `level=0.9` to force the contour to be closer to the structure of interest than the background
    contour = find_contours(structure_mask, level=0.9)[0]

    # Shift the grid center to the center of the mask
    contour_centered = contour - center

    # Convert contour points from Cartesian grid to Polar grid
    theta, rho = cart2pol(contour_centered[:, 1], contour_centered[:, 0])

    # Sort theta and rho arrays
    theta_sort_indices = theta.argsort()
    theta_sorted, rho_sorted = theta[theta_sort_indices], rho[theta_sort_indices]

    # Smooth the signal to avoid finding peaks for small localities
    rho_sorted_gaussian_filtered = gaussian_filter1d(rho_sorted, len(contour) * 5e-3)

    # Detect peaks that correspond to endo/epi base and apex
    peaks, properties = find_peaks(rho_sorted_gaussian_filtered, height=0)

    # Discard apex peak by only keeping peaks found in the lower half of the mask
    # (by discarding peaks found where theta > 0)
    lower_half_peaks_mask = theta_sorted[peaks] > 0
    peaks = peaks[lower_half_peaks_mask]

    if (num_corners := len(peaks)) > 2:
        # Extract the heights of each peak and discard the values associated with discarded peaks
        peaks_heights = properties["peak_heights"]
        peaks_heights = peaks_heights[lower_half_peaks_mask]

        # Identify the indices of the 2 highest peaks in the list of peaks
        highest_peaks = peaks_heights.argsort()[-2:]
        # Sort the indices of the 2 highest peaks to make sure they stay ordered by ascending theta
        # (so that the peak of the right corner comes first) regardless of their heights
        highest_peaks = sorted(highest_peaks)

        # Extract the peaks corresponding to the right and left corners
        peaks = peaks[highest_peaks]
    elif num_corners < 2:
        raise RuntimeError(
            f"Identified {num_corners} corner(s) for the endo/epi. We needed to identify at least "
            f"2 corners to determine control points along the contour of the endo/epi."
        )

    # Map the indices of the peaks in polar coordinates back to the indices in the list of contour points
    contour_indices = theta_sort_indices[peaks]
    # Since the peaks were ordered by ascending theta in polar coordinates, the peak corresponding to the right
    # corner is always first
    right_corner, left_corner = contour[contour_indices]
    return left_corner, right_corner


def endo_epi_control_points(
    segmentation: np.ndarray,
    lv_labels: int | Sequence[int],
    myo_labels: int | Sequence[int],
    structure: Literal["endo", "epi"],
    num_control_points: int,
    voxelspacing: np.ndarray | Tuple[float, float] = (1, 1),
) -> np.ndarray:
    """Lists uniformly distributed control points along the contour of the endocardium/epicardium.

    Args:
        segmentation: (H, W), Segmentation map.
        lv_labels: Labels of the classes that are part of the left ventricle.
        myo_labels: Labels of the classes that are part of the myocardium.
        structure: Structure for which to identify the control points.
        num_control_points: Number of control points to sample along the contour of the endocardium/epicardium. The
            number of control points should be odd to be divisible evenly between the base -> apex and apex -> base
            segments.
        voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

    Returns:
        Coordinates of the control points along the contour of the endocardium/epicardium.
    """
    voxelspacing = np.array(voxelspacing)

    # Find the points along the contour of the endo/epi excluding the base
    # "Backend" function used to find the control points at the base of the structure depends on the structure
    if structure == "endo":
        struct_labels = lv_labels
        contour = _endo_epi_contour(
            segmentation,
            struct_labels,
            functools.partial(endo_base, lv_labels=lv_labels, myo_labels=myo_labels),
        )
    elif structure == "epi":
        struct_labels = [lv_labels, myo_labels]
        contour = _endo_epi_contour(
            segmentation,
            struct_labels,
            functools.partial(epi_base, labels=struct_labels),
        )
    else:
        raise ValueError(f"Unexpected value for 'mode': {structure}. Use either 'endo' or 'epi'.")

    # Identify the apex from the points within the contour
    apex = _endo_epi_apex(segmentation, struct_labels)

    # Round the contour's coordinates, so they don't fall between pixels anymore
    contour = contour.round().astype(int)

    # Break the contour down into independent segments (base -> apex, apex -> base) along which to uniformly
    # distribute control points
    apex_idx_in_contour = np.linalg.norm((contour - apex) * voxelspacing, axis=1).argmin()
    segments = [0, apex_idx_in_contour, len(contour) - 1]

    if (num_control_points - 1) % (num_segments := len(segments) - 1):
        raise ValueError(
            f"The number of requested control points: {num_control_points}, cannot be divided evenly across the "
            f"{num_segments} contour segments. Please set a number of control points that, when subtracted by 1, "
            f"is divisible by {num_segments}."
        )
    num_control_points_per_segment = (num_control_points - 1) // num_segments

    # Simplify the general case for handling th
    control_points_indices = [0]
    for segment_start, segment_stop in itertools.pairwise(segments):
        # Slice segment so that both the start and stop points are included in the segment
        segment = contour[segment_start : segment_stop + 1]

        # Compute the geometric distances between each point along the segment and the previous point.
        # This allows to then simply compute the cumulative distance from the left corner to each segment point
        segment_dist_to_prev = [0.0] + [
            np.linalg.norm((p1 - p0) * voxelspacing) for p0, p1 in itertools.pairwise(segment)
        ]
        segment_cum_dist = np.cumsum(segment_dist_to_prev)

        # Select points along the segment that are equidistant (by selecting points that are closest to where
        # steps of `perimeter / num_control_points` would expect to find points)
        control_points_step = np.linspace(
            0, segment_cum_dist[-1], num=num_control_points_per_segment + 1
        )
        segment_control_points = [
            segment_start + np.argmin(np.abs(point_cum_dist - segment_cum_dist))
            for point_cum_dist in control_points_step
        ]
        # Skip the first control point in the current segment, because its already included as the last control
        # point of the previous segment
        control_points_indices += segment_control_points[1:]

    return np.roll(contour[control_points_indices], 1, 1)


def extract_control_points_and_save_as_json(
    seg_path: PathLike,
    output_folder: PathLike,
    num_points: int = 11,
    json_name: Optional[str] = None,
    export_gif: bool = True,
    bmode_path: Optional[PathLike] = None,
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
            (endo_epi_control_points(seg, 1, 2, "endo", num_points, spacing)).tolist()
        )

        if 2 in np.unique(seg):
            epi_points.append(
                (endo_epi_control_points(seg, 1, 2, "epi", num_points, spacing)).tolist()
            )
        else:
            epi_points.append([])

        dummy_points.append([])

    json_folder = os.path.join(output_folder, "control_points")
    os.makedirs(json_folder, exist_ok=True)

    if json_name is None:
        json_name = os.path.basename(seg_path)[:-7] + ".json"
    else:
        json_name = json_name + ".json"

    json_path = os.path.join(json_folder, json_name)

    json_dict = {}
    json_dict["contourCheck"] = 0
    json_dict["imageQuality"] = 0
    json_dict["ecg"] = []
    json_dict["left_ventricle_endo"] = (np.array(endo_points) * spacing).tolist()
    json_dict["left_ventricle_epi"] = (np.array(epi_points) * spacing).tolist()
    json_dict["right_ventricle"] = dummy_points

    if os.path.isfile(json_path):
        os.remove(json_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, separators=(",", ":"))

    if export_gif:
        endo_points = np.array(endo_points)
        epi_points = np.array(epi_points)
        point_colors = np.array([[144, 238, 144], [255, 182, 193]])
        # Create mask for control points
        control_points_mask = np.zeros([2, *seg_array.shape])
        for frame in range(seg_array.shape[0]):
            # Mark endo points in the mask
            control_points_mask[0, frame, endo_points[frame, :, 1], endo_points[frame, :, 0]] = 1
            if 2 in np.unique(seg_array):
                # Mark epi points in the mask
                control_points_mask[1, frame, epi_points[frame, :, 1], epi_points[frame, :, 0]] = 1
        # Create pseudo 3D disk structure for dilatation
        points_struct = np.stack([np.zeros(disk(3).shape), disk(3), np.zeros(disk(3).shape)])
        for i in range(control_points_mask.shape[0]):
            # Dilate the control points so that they occupy a few pixels (to make them more visible)
            control_points_mask[i] = ndimage.binary_dilation(control_points_mask[i], points_struct)
        control_points_mask = control_points_mask.astype(bool)

        # Get bmode data
        if bmode_path is not None:
            bmode_itk = sitk.ReadImage(bmode_path)
            bmode_array = sitk.GetArrayFromImage(bmode_itk)
        else:
            bmode_array = np.zeros(seg_array.shape)
        overlay = overlay_mask_on_image(bmode_array, seg_array) * 255

        # Update the pixel values of control points
        for structure in np.unique(seg_array)[1:]:
            overlay[control_points_mask[structure - 1]] = point_colors[structure - 1]

        # Export overlay to gif
        gif_folder = os.path.join(output_folder, "gif")
        os.makedirs(gif_folder, exist_ok=True)
        gif_path = os.path.join(gif_folder, json_name[:-5] + ".gif")
        array2gif(overlay, gif_path)


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

    from ascent.utils.visualization import imagesc

    for i in [27]:
        bmode_path = (
            f"C:/Users/ling/Desktop/A3C-nnUNet-results/nifti/bmode/{i:04}_A3C_bmode.nii.gz"
        )
        seg_path = f"C:/Users/ling/Desktop/A3C-nnUNet-results/nifti/post_processed_masks/{i:04}_A3C_post_mask.nii.gz"
        output_folder = "C:/Users/ling/Desktop/new_A3C_data"
        mhd_folder = "C:/Users/ling/Desktop/new_A3C_data/bmode"
        json_name = os.path.basename(seg_path)[:8]
        extract_control_points_and_save_as_json(
            seg_path, output_folder, json_name=json_name, bmode_path=bmode_path
        )
        nifti2mhd(bmode_path, mhd_folder)
    # frame = 20

    # seg_itk = sitk.ReadImage(seg_path)
    # seg_array = sitk.GetArrayFromImage(seg_itk)

    # bmode_itk = sitk.ReadImage(bmode_path)
    # bmode_array = sitk.GetArrayFromImage(bmode_itk)

    # overlaid = overlay_mask_on_image(bmode_array, seg_array)

    # spacing = seg_itk.GetSpacing()

    # for frame in range(5):
    #     endo_points = endo_epi_control_points(seg_array[frame], 1, 2, "endo", 11, spacing[:-1])
    #     epi_points = endo_epi_control_points(seg_array[frame], 1, 2, "epi", 11, spacing[:-1])

    #     plt.figure(figsize=(18, 6), dpi=300)
    #     ax = plt.subplot(1, 3, 1)
    #     imagesc(ax, overlaid[frame], show_colorbar=False)
    #     ax.scatter(endo_points[:, 0], endo_points[:, 1], c="r", s=4)
    #     ax.scatter(epi_points[:, 0], epi_points[:, 1], c="b", s=4)
    #     plt.show()
