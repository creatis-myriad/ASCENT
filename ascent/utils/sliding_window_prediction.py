import warnings
from typing import List, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F


def pad_nd_image(
    image: Union[torch.Tensor, np.ndarray],
    new_shape: Tuple[int, ...] = None,
    mode: str = "constant",
    kwargs: dict = None,
    return_slicer: bool = False,
    shape_must_be_divisible_by: Union[int, Tuple[int, ...], List[int]] = None,
) -> Union[Union[torch.Tensor, np.ndarray], Tuple[Union[torch.Tensor, np.ndarray], Tuple]]:
    """
    One padder to pad them all. Documentation? Well okay. A little bit
    Padding is done such that the original content will be at the center of the padded image. If the amount of padding
    needed it odd, the padding 'above' the content is larger,
    Example:
    old shape: [ 3 34 55  3]
    new_shape: [3, 34, 96, 64]
    amount of padding (low, high for each axis): [[0, 0], [0, 0], [20, 21], [30, 31]]
    :param image: can either be a numpy array or a torch.Tensor. pad_nd_image uses np.pad for the former and
           torch.nn.functional.pad for the latter
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
           len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in
           any of the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
           Example:
           image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
           image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).
    :param mode: will be passed to either np.pad or torch.nn.functional.pad depending on what the image is. Read the
           respective documentation!
    :param return_slicer: if True then this function will also return a tuple of python slice objects that you can use
           to crop back to the original image (reverse padding)
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
           divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match
           that will be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation (numpy) or torch.nn.functional.pad (torch)
    :returns: if return_slicer=False, this function returns the padded numpy array / torch Tensor. If
              return_slicer=True it will also return a tuple of slice objects that you can use to revert the padding:
              output, slicer = pad_nd_image(input_array, new_shape=XXX, return_slicer=True)
              reversed_padding = output[slicer] ## this is now the same as input_array, padding was reversed
    """
    if kwargs is None:
        kwargs = {}

    old_shape = np.array(image.shape)

    if shape_must_be_divisible_by is not None:
        assert isinstance(shape_must_be_divisible_by, (int, list, tuple, np.ndarray))  # nosec
        if isinstance(shape_must_be_divisible_by, int):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(image.shape)
        else:
            if len(shape_must_be_divisible_by) < len(image.shape):
                shape_must_be_divisible_by = [1] * (
                    len(image.shape) - len(shape_must_be_divisible_by)
                ) + list(shape_must_be_divisible_by)

    if new_shape is None:
        assert shape_must_be_divisible_by is not None  # nosec
        new_shape = image.shape

    if len(new_shape) < len(image.shape):
        new_shape = list(image.shape[: len(image.shape) - len(new_shape)]) + list(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)

        if len(shape_must_be_divisible_by) < len(new_shape):
            shape_must_be_divisible_by = [1] * (
                len(new_shape) - len(shape_must_be_divisible_by)
            ) + list(shape_must_be_divisible_by)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array(
            [
                new_shape[i]
                + shape_must_be_divisible_by[i]
                - new_shape[i] % shape_must_be_divisible_by[i]
                for i in range(len(new_shape))
            ]
        )

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [list(i) for i in zip(pad_below, pad_above)]

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        if isinstance(image, np.ndarray):
            res = np.pad(image, pad_list, mode, **kwargs)
        elif isinstance(image, torch.Tensor):
            # torch padding has the weirdest interface ever. Like wtf? Y u no read numpy documentation? So much easier
            torch_pad_list = [i for j in pad_list for i in j[::-1]][::-1]
            res = F.pad(image, torch_pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = tuple(slice(*i) for i in pad_list)
        return res, slicer


def compute_gaussian(
    tile_size: Tuple[int, ...], sigma_scale: float = 1.0 / 8, dtype=np.float16
) -> np.ndarray:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0]
    )

    return gaussian_importance_map


def compute_steps_for_sliding_window(
    image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float
) -> List[List[int]]:
    assert [  # nosec
        i >= j for i, j in zip(image_size, tile_size)
    ], "image size must be as large or larger than patch_size"
    assert (  # nosec
        0 < tile_step_size <= 1
    ), "step_size must be larger than 0 and smaller or equal to 1"

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [
        int(np.ceil((i - k) / j)) + 1
        for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)
    ]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


def get_sliding_window_generator(
    image_size: Tuple[int, ...],
    tile_size: Tuple[int, ...],
    tile_step_size: float,
    verbose: bool = False,
):
    if len(tile_size) < len(image_size):
        assert len(tile_size) == len(image_size) - 1, (  # nosec
            "if tile_size has less entries than image_size, len(tile_size) "
            "must be one shorter than len(image_size) (only dimension "
            "discrepancy of 1 allowed)."
        )
        steps = compute_steps_for_sliding_window(image_size[1:], tile_size, tile_step_size)
        if verbose:
            print(
                f"n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is {image_size}, tile_size {tile_size}, "
                f"tile_step_size {tile_step_size}\nsteps:\n{steps}"
            )
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicer = tuple(
                        [
                            slice(None),
                            d,
                            *[slice(si, si + ti) for si, ti in zip((sx, sy), tile_size)],
                        ]
                    )
                    yield slicer
    else:
        steps = compute_steps_for_sliding_window(image_size, tile_size, tile_step_size)
        if verbose:
            print(
                f"n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {tile_size}, "
                f"tile_step_size {tile_step_size}\nsteps:\n{steps}"
            )
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicer = tuple(
                        [
                            slice(None),
                            *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), tile_size)],
                        ]
                    )
                    yield slicer


def maybe_mirror_and_predict(
    network: nn.Module, x: torch.Tensor, mirror_axes: Tuple[int, ...] = None
) -> torch.Tensor:
    prediction = network(x)

    if mirror_axes is not None:
        # check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert (  # nosec
            max(mirror_axes) <= len(x.shape) - 3
        ), "mirror_axes does not match the dimension of the input!"

        num_predictons = 2 ** len(mirror_axes)
        if 0 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2,))), (2,))
        if 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3,))), (3,))
        if 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (4,))), (4,))
        if 0 in mirror_axes and 1 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3))), (2, 3))
        if 0 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 4))), (2, 4))
        if 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (3, 4))), (3, 4))
        if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
        prediction /= num_predictons
    return prediction


def predict_sliding_window_return_logits(
    network: nn.Module,
    input_image: Union[np.ndarray, torch.Tensor],
    num_segmentation_heads: int,
    tile_size: Tuple[int, ...],
    mirror_axes: Tuple[int, ...] = None,
    tile_step_size: float = 0.5,
    use_gaussian: bool = True,
    precomputed_gaussian: torch.Tensor = None,
    perform_everything_on_gpu: bool = True,
    verbose: bool = True,
    device: str = "cuda",
) -> Union[np.ndarray, torch.Tensor]:
    network.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        with torch.autocast("cuda" if device.startswith("cuda") else "cpu"):
            assert (  # nosec
                len(input_image.shape) == 4
            ), "input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)"

            if not torch.cuda.is_available():
                if perform_everything_on_gpu:
                    print(
                        'WARNING! "perform_everything_on_gpu" was True but cuda is not available! Set it to False...'
                    )
                perform_everything_on_gpu = False

            results_device = "cuda" if perform_everything_on_gpu else "cpu"

            if verbose:
                print("step_size:", tile_step_size)
            if verbose:
                print("mirror_axes:", mirror_axes)

            if not isinstance(input_image, torch.Tensor):
                # pytorch will warn about the numpy array not being writable. This doesn't matter though because we
                # just want to read it. Suppress the warning in order to not confuse users...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    input_image = torch.from_numpy(input_image)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(
                input_image, tile_size, "constant", {"value": 0}, True, None
            )

            if use_gaussian:
                gaussian = (
                    torch.from_numpy(compute_gaussian(tile_size, sigma_scale=1.0 / 8))
                    if precomputed_gaussian is None
                    else precomputed_gaussian
                )
                gaussian = gaussian.half()
                # make sure nothing is rounded to zero or we get division by zero :-(
                mn = gaussian.min()
                if mn == 0:
                    gaussian.clip_(min=mn)

            slicers = get_sliding_window_generator(
                data.shape[1:], tile_size, tile_step_size, verbose=verbose
            )

            # preallocate results and num_predictions. Move everything to the correct device
            try:
                predicted_logits = torch.zeros(
                    (num_segmentation_heads, *data.shape[1:]),
                    dtype=torch.half,
                    device=results_device,
                )
                n_predictions = torch.zeros(
                    data.shape[1:], dtype=torch.half, device=results_device
                )
                gaussian = gaussian.to(results_device)
            except RuntimeError:
                # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                results_device = "cpu"
                predicted_logits = torch.zeros(
                    (num_segmentation_heads, *data.shape[1:]),
                    dtype=torch.half,
                    device=results_device,
                )
                n_predictions = torch.zeros(
                    data.shape[1:], dtype=torch.half, device=results_device
                )
                gaussian = gaussian.to(results_device)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for sl in slicers:
                workon = data[sl][None]
                workon = workon.to(device, non_blocking=False)

                prediction = maybe_mirror_and_predict(network, workon, mirror_axes)[0].to(
                    results_device
                )

                predicted_logits[sl] += prediction * gaussian if use_gaussian else prediction
                n_predictions[sl[1:]] += gaussian if use_gaussian else 1

            predicted_logits /= n_predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]
