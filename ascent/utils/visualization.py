from typing import Optional, Type, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor


def imagesc(
    ax: matplotlib.axes,
    image: Union[Tensor, np.ndarray],
    title: Optional[str] = None,
    colormap: matplotlib.colormaps = plt.cm.gray,
    clim: Optional[Union[tuple[float, float], list[float, float]]] = None,
    show_axis: bool = False,
) -> None:
    """Display image with scaled colors. Similar to Matlab's imagesc.

    Args:
        ax: Axis to plot on.
        image: Array to plot.
        title: Title of plotting.
        colormap: Colormap of plotting.
        clim: Colormap limits.
        show_axis: Whether to show axis when plotting.

    Example:
        >>> plt.figure("image", (18, 6))
        >>> ax = plt.subplot(1, 2, 1)
        >>> imagesc(ax, np.random.rand(100,100), "image", clim=[-1, 1])
        >>> plt.show()
    """

    if clim is not None and isinstance(clim, (list, tuple)):
        if len(clim) == 2 and (clim[0] < clim[1]):
            clim_args = {"vmin": float(clim[0]), "vmax": float(clim[1])}
        else:
            raise ValueError(
                f"clim should be a list or tuple containing 2 floats with clim[0] < clim[1], got {clim} instead."
            )
    else:
        clim_args = {}

    if isinstance(image, Tensor):
        image = image.cpu().detach().numpy()

    im = ax.imshow(image, colormap, **clim_args)
    plt.title(title)
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


if __name__ == "__main__":
    cmap = dopplermap()
    plt.figure("image", (18, 6))
    ax = plt.subplot(1, 2, 1)
    imagesc(ax, np.random.rand(100, 100), "image", cmap, clim=[-1, 1])
    ax = plt.subplot(1, 2, 2)
    imagesc(ax, np.random.rand(100, 100), "image", cmap)
    plt.show()
