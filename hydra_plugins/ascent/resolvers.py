from typing import Literal, Union

import numpy as np
from omegaconf import OmegaConf


def determine_rotation_range(
    patch_size: Union[list[int], tuple[int, ...]],
    do_dummy_2D_data_aug: bool,
    axis: Literal["x", "y", "z"],
) -> Union[list[float], float]:
    """Determine the rotation range for the given axis.

    Args:
        patch_size: Patch size used by the model.
        do_dummy_2D_data_aug: Whether to do dummy 2D data augmentation.
        axis: Axis to determine the rotation range for.

    Returns:
        Rotation range for the given axis.
    """
    if len(patch_size) == 3:
        range_x = range_y = range_z = [-30.0 / 180 * np.pi, 30.0 / 180 * np.pi]
        if do_dummy_2D_data_aug:
            range_x = [-180.0 / 180 * np.pi, 180.0 / 180 * np.pi]
            range_y = range_z = 0.0
    else:
        range_x = [-180.0 / 180 * np.pi, 180.0 / 180 * np.pi]
        range_y = range_z = 0.0
        if max(patch_size) / min(patch_size) > 1.5:
            range_x = [-15.0 / 180 * np.pi, 15.0 / 180 * np.pi]

    if axis == "x":
        return range_x
    elif axis == "y":
        return range_y
    elif axis == "z":
        return range_z


def determine_interpolation_mode(
    patch_size: Union[list[int], tuple[int, ...]],
    do_dummy_2D_data_aug: bool,
    transforms: Literal["rotation", "zoom"],
) -> str:
    """Determine the interpolation mode used for rotation or zoom.

    Args:
        patch_size: Patch size used by the model.
        do_dummy_2D_data_aug: Whether to do dummy 2D data augmentation.
        transforms: Type of transformation to determine the interpolation mode for.

    Returns:
        Interpolation mode.
    """
    if len(patch_size) == 3:
        rot_inter_mode = "bilinear"
        zoom_inter_mode = "trilinear"
        if do_dummy_2D_data_aug:
            zoom_inter_mode = rot_inter_mode = "bicubic"
    else:
        zoom_inter_mode = rot_inter_mode = "bicubic"

    if transforms == "rotation":
        return rot_inter_mode
    elif transforms == "zoom":
        return zoom_inter_mode


OmegaConf.register_new_resolver("get_rot_range", determine_rotation_range)
OmegaConf.register_new_resolver("get_interp_mode", determine_interpolation_mode)
