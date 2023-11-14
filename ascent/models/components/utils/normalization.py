from typing import Literal, Union

import torch.nn.functional as F
from einops import rearrange
from torch import Size, Tensor, nn


class LayerNorm(nn.LayerNorm):
    """LayerNorm that converts channels_first to channels_last before normalization and back
    after."""

    def __init__(
        self,
        normalized_shape: Union[int, list[int], Size],
        data_format: Literal["channel_first", "channel_last"],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """Initialize class instance.

        Args:
            data_format: Data format of the input tensor. Can be either `channel_first` or
                `channel_last`.
            normalized_shape: Input shape from an expected input of size
                :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`.
                Can be either a single int or a list of ints, where each int is the size of the
                expected input.
            eps: A value added to the denominator for numerical stability. Default: 1e-5.
            elementwise_affine: A boolean value that when set to ``True``, this module has
                learnable per-element affine parameters initialized to ones (for weights) and zeros
                (for biases). Default: ``True``.
            bias: A boolean value that when set to ``True``, this module has learnable bias
                parameters. Default: ``True``.
            device: Device on which the layer is stored. If not specified, this defaults to the
                device of the first parameter. Default: ``None``.
            dtype: Data type of the layer's parameters. If not specified, this defaults to the
                dtype of the first parameter. Default: ``None``.
        """
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if data_format not in ["channel_first", "channel_last"]:
            raise NotImplementedError(
                f"Only 'channel_first' and 'channel_last' data formats are supported, got "
                f"{data_format}."
            )
        self.data_format = data_format

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        # permute to channels_last
        if self.data_format == "channel_first":
            if len(x.shape) == 4:
                x = rearrange(x, "b c h w -> b h w c")
            elif len(x.shape) == 5:
                x = rearrange(x, "b c h w d -> b h w d c")
            else:
                raise ValueError("Only 2D and 3D inputs are supported.")
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # permute back to channels_first
        if self.data_format == "channel_first":
            if len(x.shape) == 4:
                x = rearrange(x, "b h w c -> b c h w")
            elif len(x.shape) == 5:
                x = rearrange(x, "b h w d c -> b c h w d")
            else:
                raise ValueError("Only 2D and 3D inputs are supported.")
        return x
