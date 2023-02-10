from copy import deepcopy
from typing import Union

import numpy as np


def get_pool_and_conv_props(
    spacing: tuple[int, ...],
    patch_size: tuple[int, ...],
    min_feature_map_size: int,
    max_numpool: int,
) -> tuple[list[int, ...]]:
    """Compute the optimum patch size, strides, number of pooling, and convolution kernel sizes
    based on the given arguments.

    Args:
        spacing: Pixel spacing of the patch size.
        patch_size: Input patch size of U-Net.
        min_feature_map_size: Minimum edge length of feature maps in the U-Net bottleneck.
        max_numpool: Maximum number of pooling for all the axes.

    Returns:
        A tuple containing:
            - number of pooling per axis
            - stride per axis for each convolution layer
            - kernel size per axis for each convolution layer
            - patch size that is divisible by the number of pooling
            - 2 ** number of pooling per axis
    """
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = []
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim

    while True:
        # This is a problem because sometimes we have spacing 20, 50, 50 and we want to still keep pooling.
        # Here we would stop however. This is not what we want! Fixed in get_pool_and_conv_propsv2
        min_spacing = min(current_spacing)
        valid_axes_for_pool = [i for i in range(dim) if current_spacing[i] / min_spacing < 2]
        axes = []
        for a in range(dim):
            my_spacing = current_spacing[a]
            partners = [
                i
                for i in range(dim)
                if current_spacing[i] / my_spacing < 2 and my_spacing / current_spacing[i] < 2
            ]
            if len(partners) > len(axes):
                axes = partners
        conv_kernel_size = [3 if i in axes else 1 for i in range(dim)]

        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        # before = len(valid_axes_for_pool)
        valid_axes_for_pool = [
            i for i in valid_axes_for_pool if current_size[i] >= 2 * min_feature_map_size
        ]
        # after = len(valid_axes_for_pool)
        # if after == 1 and before > 1:
        #    break

        valid_axes_for_pool = [
            i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool
        ]

        if len(valid_axes_for_pool) == 0:
            break

        # print(current_spacing, current_size)

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(conv_kernel_size)
        # print(conv_kernel_sizes)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3] * dim)
    return (
        num_pool_per_axis,
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        patch_size,
        must_be_divisible_by,
    )


def get_shape_must_be_divisible_by(num_pool_per_axis: tuple[int, ...]) -> np.ndarray:
    """Compute 2 ** number of pooling per axis to get the integers to divide each axis of the input
    patch with.

    Args:
        num_pool_per_axis: Number of pooling per axis.

    Returns:
        Numpy array containing 2 ** number of pooling per axis.
    """
    return 2 ** np.array(num_pool_per_axis)


def pad_shape(
    shape: tuple[int, ...],
    must_be_divisible_by: Union[
        tuple[int, ...],
        list[int, ...],
        np.ndarray[int, ...],
    ],
) -> tuple[int, ...]:
    """Pads ``shape`` so that it is divisibly by ``must_be_divisible_by``.

    Args:
        shape: Shaped to be padded
        must_be_divisible_by: List or tuple or numpy array containing integers to divide each axis
            of ``shape``.

    Returns:
        Padded shape that is divisible by ``must_be_divisible_by``.
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        if not len(must_be_divisible_by) == len(shape):
            raise ValueError(
                f"must_be_divisible_by {len(must_be_divisible_by)} does not have the length as ",
                f"shape {len(shape)}",
            )

    new_shp = [
        shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i]
        for i in range(len(shape))
    ]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp
