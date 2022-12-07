from itertools import pairwise
from typing import Sequence, Union

import numpy as np
from skimage.feature import corner_peaks, corner_shi_tomasi
from skimage.measure import find_contours


def endo_epi_control_points(
    segmentation: np.ndarray,
    labels: Union[int, Sequence[int]],
    num_control_points: int,
    spacing: tuple[float, float],
) -> np.ndarray:
    """Lists uniformly distributed control points along the contour of the endocardium/epicardium.

    Args:
        segmentation: (H, W), Segmentation map.
        labels: Labels of the classes that are part of the endocardium/epicardium.
        num_control_points: Number of control points to sample along the contour of the endocardium/epicardium.
        spacing: Pixel spacing in ``mm``.

    Returns:
        Coordinates of the control points along the contour of the endocardium/epicardium.

    Raises:
        RuntimeError: When not exactly two corners are detected.
    """
    structure_mask = np.isin(segmentation, labels)

    # Find the two points at the base of the structure of interest, using Shi-Tomasi corner detection algorithm
    base_corners = corner_peaks(corner_shi_tomasi(structure_mask), min_distance=10, num_peaks=2)
    if (num_corners := len(base_corners)) != 2:
        raise RuntimeError(
            f"Identified {num_corners} corner(s) for the endo/epi. We needed to identify exactly 2 corners to "
            f"determine control points along the contour of the endo/epi."
        )
    left_corner_idx, right_corner_idx = np.argmin(base_corners[:, 1]), np.argmax(
        base_corners[:, 1]
    )
    left_corner, right_corner = base_corners[left_corner_idx], base_corners[right_corner_idx]

    # Extract all the points on the contour of the structure of interest
    # Use level=0.9 to force the contour to touch the endo/epicardial wall
    contours = find_contours(structure_mask, level=0.9)[0]

    # Shift the contours so that they start at the left corner
    # To detect the contour coordinates that match the corner, we use the closest match since skimage's
    # `find_contours` coordinates are interpolated between pixels, so they won't match exactly corner coordinates
    dist_to_left_corner = np.linalg.norm(left_corner - contours, axis=1)
    left_corner_contour_idx = np.argmin(dist_to_left_corner)
    contours = np.roll(contours, -left_corner_contour_idx, axis=0)

    # Filter the full contour to discard points along the base
    # We implement this by slicing the contours from the left corner to the right corner, since the contour returned
    # by skimage's `find_contours` is oriented clockwise
    dist_to_right_corner = np.linalg.norm(right_corner - contours, axis=1)
    right_corner_contour_idx = np.argmin(dist_to_right_corner)
    outer_contours = contours[: right_corner_contour_idx + 1]

    # Round the contours' coordinates so they don't fall between pixels anymore
    outer_contours = np.round(outer_contours).astype(int)

    # Compute accumulated distance from left corner to right corner
    dist = [0]
    for p0, p1 in pairwise(outer_contours):
        dist.append(dist[-1] + np.linalg.norm((p1 - p0) * np.array(spacing)))

    dist = np.array(dist)

    # Get equally spaced distance steps along the contour's perimeter
    dist_step = np.linspace(dist[0], dist[-1], num=num_control_points)

    # Get the closest points on the contour that respect the distance steps
    control_points_idx = [np.argmin(np.abs(d - dist)) for d in dist_step]

    return np.roll(outer_contours[control_points_idx], 1, axis=1)


if __name__ == "__main__":
    import SimpleITK as sitk
    from matplotlib import pyplot as plt

    from ascent.utils.visualization import imagesc

    seg_path = (
        "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/data/CAMUS/raw/labelsTr/NewCamus_0012.nii.gz"
    )
    seg_itk = sitk.ReadImage(seg_path)
    seg_array = sitk.GetArrayFromImage(seg_itk)
    spacing = seg_itk.GetSpacing()
    endo_points = endo_epi_control_points(seg_array[0], 1, 15, spacing[:-1])
    epi_points = endo_epi_control_points(seg_array[0], [1, 2], 15, spacing[:-1])

    plt.figure(figsize=(18, 6), dpi=1200)
    ax = plt.subplot(1, 3, 1)
    imagesc(ax, seg_array[0], clim=[0, 2], show_colorbar=False)
    ax.scatter(endo_points[:, 0], endo_points[:, 1], c="r", s=4)
    ax.scatter(epi_points[:, 0], epi_points[:, 1], c="b", s=4)
    plt.show()
