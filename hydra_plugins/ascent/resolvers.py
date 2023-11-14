from typing import Literal, Union

import numpy as np
import omegaconf
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
        # Cast list to omegaconf.ListConfig to avoid error when resolving with OmegaConf
        return omegaconf.ListConfig(range_x) if isinstance(range_x, list) else range_x
    elif axis == "y":
        # Cast list to omegaconf.ListConfig to avoid error when resolving with OmegaConf
        return omegaconf.ListConfig(range_y) if isinstance(range_y, list) else range_y
    elif axis == "z":
        # Cast list to omegaconf.ListConfig to avoid error when resolving with OmegaConf
        return omegaconf.ListConfig(range_z) if isinstance(range_z, list) else range_z


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


def determine_seg_key_based_on_keys(keys: list[str]) -> str:
    """Determine the segmentation key based on the keys.

    Args:
        keys: List of keys.

    Returns:
        Segmentation key.

    Raises:
        ValueError: If the keys do not contain either `seg` or `label`.
    """
    if "seg" in keys:
        return "seg"
    elif "label" in keys:
        return "label"
    else:
        raise ValueError("No segmentation key found in the given keys.")


def determine_separate_transform_based_on_input_channels(in_channels: int) -> bool:
    """Determine whether to apply separate noise and intensity transforms to multichannel image
    based on the input channels.

    Used for dealiasing where channel 0 is the aliased image and channel 1 is the Doppler power
    when input contains 2 channels. We want to keep Doppler data intact and only apply noise and
    intensity transforms to the aliased image.

    Args:
        in_channels: Number of input channels.

    Returns:
        Whether to use separate transforms.

    Raises:
        NotImplementedError: If more than 2 channels are given.
    """
    if in_channels > 2:
        raise NotImplementedError(
            "Separate transforms are not implemented for more than 2 channels."
        )
    return in_channels == 2


def determine_key_for_noise_and_intensity_transforms(
    separate_transform: bool, image_key: str
) -> str:
    """Determine the key for noise and intensity transforms based on the `separate transforms`
    flag.

    Args:
        separate_transform: Whether to apply separate transforms.
        image_key: Key for the image.

    Returns:
        Key for noise and intensity transforms.
    """
    if separate_transform:
        return f"{image_key}_0"
    else:
        return image_key


def get_crop_size(patch_size: Union[list[int], tuple[int, ...]]) -> omegaconf.ListConfig[int]:
    """Determine the crop size for the given patch size.

    Args:
        patch_size: Patch size used by the model.

    Returns:
        Crop size.
    """
    if len(patch_size) == 3:
        return omegaconf.ListConfig(patch_size)
    else:
        return omegaconf.ListConfig([*patch_size, 1])


def get_in_channels_from_model_net(net: omegaconf.DictConfig) -> int:
    """Get the number of input channels from the model net.

    Args:
        net: Model net.

    Returns:
        Number of input channels.

    Raises:
        ValueError: If the number of input channels could not be determined.
    """
    if "in_channels" in net:
        return net.in_channels
    elif "encoder" in net:
        return net.encoder.in_channels
    else:
        raise ValueError("Could not determine number of input channels.")


def get_dim_from_patch_size(
    patch_size: Union[list[int], tuple[int, ...], omegaconf.ListConfig]
) -> int:
    """Get the dimension from the patch size.

    Args:
        patch_size: Patch size used by the model.

    Returns:
        Dimension.

    Raises:
        NotImplementedError: If the patch size is not 2D or 3D.
    """
    if len(patch_size) == 3:
        return 3
    elif len(patch_size) == 2:
        return 2
    else:
        raise NotImplementedError("Only 2D and 3D patch size is supported.")


OmegaConf.register_new_resolver("get_rot_range", determine_rotation_range)
OmegaConf.register_new_resolver("get_interp_mode", determine_interpolation_mode)
OmegaConf.register_new_resolver("get_seg_key", determine_seg_key_based_on_keys)
OmegaConf.register_new_resolver(
    "do_separate_transform", determine_separate_transform_based_on_input_channels
)
OmegaConf.register_new_resolver(
    "get_noise_and_intensity_transform_key", determine_key_for_noise_and_intensity_transforms
)
OmegaConf.register_new_resolver("get_crop_size", get_crop_size)
OmegaConf.register_new_resolver("get_in_channels_from_model_net", get_in_channels_from_model_net)
OmegaConf.register_new_resolver("get_dim_from_patch_size", get_dim_from_patch_size)
