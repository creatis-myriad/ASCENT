from typing import Union

import numpy as np
import torch
from torch import Tensor


def sum_tensor(inputs: Tensor, axes: tuple[int], keepdim: bool = False) -> Tensor:
    """Reduce tensor across given dimensions by summing.

    Args:
        inputs: Input tensor.
        axes: Axes to reduce.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        Sum of each row of the input tensor in the given axes. If axes is a list of dimensions,
        reduce over all of them.

    Retrieved from:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    """
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inputs = inputs.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inputs = inputs.sum(int(ax))
    return inputs


def reshape_fortran(
    inputs: Union[np.ndarray, Tensor], shape: Union[tuple, list]
) -> Union[np.ndarray, Tensor]:
    """Reshape tensor/array in Fortran-like style.

    Args:
        inputs: Tensor/array to reshape.
        shape: Desired shape for reshapping.

    Returns:
        Reshaped tensor/array.
    """
    if isinstance(inputs, Tensor):
        inputs = inputs.permute(*reversed(range(len(inputs.shape))))
        return inputs.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))
    elif isinstance(inputs, np.ndarray):
        return np.reshape(inputs, shape, order="F")


def round_differentiable(inputs: Tensor) -> Tensor:
    """A differentiable version of round that returns identity tensor when backward is called.

    Args:
        inputs: Tensor to be rounded.

    Returns:
        Rounded tensor that is also differentiable.
    """
    # This is equivalent to replacing round function (non-differentiable) with
    # an identity function (differentiable) only when backward.
    forward_value = torch.round(inputs)
    out = inputs.clone()
    out.data = forward_value.data
    return out
