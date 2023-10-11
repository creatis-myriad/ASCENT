import errno
import os
from pathlib import Path
from typing import Optional, Type, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import find_contours
from torch import Tensor
from tqdm.auto import tqdm

from ascent.utils.file_and_folder_operations import remove_suffixes


def imagesc(
    ax: matplotlib.axes,
    image: Union[Tensor, np.ndarray],
    title: Optional[str] = None,
    colormap: matplotlib.colormaps = plt.cm.gray,
    clim: Optional[tuple[float, float]] = None,
    show_axis: bool = False,
    show_colorbar: bool = True,
    **kwargs,
) -> None:
    """Display image with scaled colors. Similar to Matlab's imagesc.

    Args:
        ax: Axis to plot on.
        image: Array to plot.
        title: Title of plotting.
        colormap: Colormap of plotting.
        clim: Colormap limits.
        show_axis: Whether to show axis when plotting.
        show_colorbar: Whether to show colorbar when plotting.
        **kwargs: Keyword arguments to be passed to `imshow`.

    Example:
        >>> plt.figure("image", (18, 6))
        >>> ax = plt.subplot(1, 2, 1)
        >>> imagesc(ax, np.random.rand(100,100), "image", clim=(-1, 1))
        >>> plt.show()
    """

    if clim is not None and isinstance(clim, (list, tuple)):
        if len(clim) == 2 and (clim[0] < clim[1]):
            clim_args = {"vmin": float(clim[0]), "vmax": float(clim[1])}
        else:
            raise ValueError(
                f"clim should be a list or tuple containing 2 floats with clim[0] < clim[1], "
                f"got {clim} instead.",
            )
    else:
        clim_args = {}

    if isinstance(image, Tensor):
        image = image.cpu().detach().numpy()

    im = ax.imshow(image, colormap, **clim_args, **kwargs)
    plt.title(title)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(im, cax)

    if not show_axis:
        ax.set_axis_off()


def dopplermap(m: int = 256) -> Type[ListedColormap]:
    """Color Doppler-based color map.

    Args:
        m: Number of samples in the colormap.

    Returns:
        Doppler colormap in Matplotlib format.
    """

    x = np.linspace(0, 1, m)
    R = 1.8 * np.emath.sqrt(x - 0.5)
    R[np.abs(R) > 1] = 1
    R = R * (x >= 0.5)
    R = np.real(R)
    G = -8 * (x - 0.5) ** 4 + 6 * (x - 0.5) ** 2
    B = np.real((1.1 * np.emath.sqrt(0.5 - x)) * (x < 0.5))
    J = []
    for rgb in zip(R, G, B):
        J.append([*rgb, 1.0])
    J = np.array(J)
    cmap = ListedColormap(J)
    return cmap


def plot_LV_endo_epi_contour(
    bmode: Union[Path, str], mask: Union[Path, str], save_dir: Union[Path, str], dpi: int = 300
) -> None:
    """Plot and overlay the endo and epicardial contour of the left ventricle mask on the
    corresponding bmode image.

    Args:
        bmode: Path to bmode image.
        mask: Path to segmentation mask.
        save_dir: Path to save the plot.
        dpi: DPI to save the figure(s).
    """

    if not os.path.isfile(bmode):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), bmode)

    if not os.path.isfile(mask):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), mask)

    case_identifier = remove_suffixes(Path(bmode)).name

    os.makedirs(save_dir, exist_ok=True)

    bmode_itk = sitk.ReadImage(bmode)
    mask_itk = sitk.ReadImage(mask)

    if not np.all(np.array(bmode_itk.GetSize()) == np.array(mask_itk.GetSize())):
        raise ValueError(
            f"B-mode {bmode_itk.GetSize()} does not have the same shape as the mask {mask_itk.GetSize()}",
        )

    voxel_spacing = bmode_itk.GetSpacing()

    image_array = sitk.GetArrayFromImage(bmode_itk)
    mask_array = sitk.GetArrayFromImage(mask_itk)
    mask_array[mask_array > 2] = 0

    for frame in tqdm(
        range(image_array.shape[0]),
        desc="Plotting LV endo and epi contours",
        unit="frame",
        position=0,
        leave=True,
    ):
        lv_endo = mask_array[frame] == 1
        lv_epi = mask_array[frame] >= 1

        fig, ax = plt.subplots(1)
        height, width = image_array.shape[1:]
        ax.axis("off")

        instance = np.stack((lv_endo, lv_epi), 0)
        colors = ["#00f000", "#fc0000"]

        # Reverse the order here to plot the epi contour first
        for i in reversed(range(0, 2)):
            contours = find_contours(instance[i], 0.5)
            padded_mask = np.zeros(
                (instance[i].shape[0] + 2, instance[i].shape[1] + 2), dtype=np.uint8
            )
            padded_mask[1:-1, 1:-1] = instance[i]
            contours = find_contours(padded_mask, 0.5)
            for contour in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                contour = np.fliplr(contour) - 1
                p = Polygon(contour, facecolor="none", edgecolor=colors[i], linewidth=2)
                ax.add_patch(p)

        ax.imshow(
            image_array[frame],
            cmap="gray",
            aspect=float(voxel_spacing[1]) / float(voxel_spacing[0]),
        )
        filename_tosave = f"{case_identifier}_{frame+1:03d}.png"
        fig.savefig(Path(save_dir) / filename_tosave, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    cmap = dopplermap()
    plt.figure("image", (18, 6))
    ax = plt.subplot(1, 2, 1)
    imagesc(ax, np.random.rand(100, 100), "image", cmap, clim=[-1, 1])
    ax = plt.subplot(1, 2, 2)
    imagesc(ax, np.random.rand(100, 100), "image", cmap)
    plt.show()
