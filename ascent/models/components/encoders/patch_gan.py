from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import torch
from monai.data import MetaTensor
from torch import Tensor, nn

from ascent.models.components.utils.initialization import get_initialization
from ascent.models.components.utils.layers import ConvLayer


class PatchGAN(nn.Module):
    """A dynamic PatchGAN Discriminator supporting both 2D and 3D inputs."""

    # define some static class attributes
    MAX_NUM_FILTERS_2D = 512
    MAX_NUM_FILTERS_3D = 320

    def __init__(
        self,
        in_channels: int,
        dim: int,
        kernels: Union[
            int, list[Union[int, list[int]]], tuple[Union[int, tuple[int, ...]], ...]
        ] = 4,
        strides: Union[
            int, list[Union[int, list[int]]], tuple[Union[int, tuple[int, ...]], ...]
        ] = 2,
        num_stages: int = 3,
        start_features: int = 64,
        conv_kwargs: Optional[dict] = None,
        norm_layer: Optional[Literal["instance", "batch"]] = "instance",
        norm_kwargs: Optional[dict] = None,
        activation: Optional[
            Literal["relu", "leakyrelu", "gelu", "tanh", "sigmoid"]
        ] = "leakyrelu",
        activation_kwargs: Optional[dict] = None,
        initialization: Literal[
            "kaiming_normal", "xavier_uniform", "trunc_normal"
        ] = "kaiming_normal",
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of input channels.
            num_stages: Number of stages in the encoder.
            dim: Spatial dimension of the input data.
            kernels: Size of the convolving kernel. If the value is an integer, then the same
                value is used for all spatial dimensions and all stages. If the value is a sequence
                of length `num_stages`, then each value is used for each stage. If the value is a
                sequence of length `num_stages` and each element is a sequence of length `dim`,
                then each value is used for each stage and each spatial dimension.
            strides: Stride of the convolution. If the value is an integer, then the same value is
                used for all spatial dimensions and all stages. If the value is a sequence of
                length `num_stages`, then each value is used for each stage. If the value is a
                sequence of length `num_stages` and each element is a sequence of length `dim`,
                then each value is used for each stage and each spatial dimension.
            start_features: Number of features in the first stage.
            conv_bias: Whether to use bias in the convolutional layers.
            conv_kwargs: Keyword arguments for the convolution layers.
            pooling: Pooling to use. Can be either `max`, `avg` or `stride`.
            adaptive_pooling: Whether to use adaptive pooling in case of `max` or `avg` pooling.
            norm_layer: Normalization to use. Can be either `instance` or `batch`.
            norm_kwargs: Keyword arguments for the normalization layers.
            activation: Activation function to use. Can be either `relu`, `leakyrelu`, `gelu`,
                `tanh` or `sigmoid`.
            activation_kwargs: Keyword arguments for the activation functions.
            initialization: Weight initialization technique. Can be either `kaiming_normal` or
                `xavier_uniform`.

        Raises:
            ValueError: When `len(kernels)` is not equal to `num_stages`.
            ValueError: When `len(strides)` is not equal to `num_stages`.
        """
        super().__init__()

        if isinstance(kernels, int):
            kernels = ((kernels,) * dim,) * num_stages

        if isinstance(strides, int):
            strides = ((strides,) * dim,) * num_stages

        if not len(kernels) == num_stages:
            raise ValueError(f"len(kernels) must be equal to num_stages: {num_stages}")

        if not len(strides) == num_stages:
            raise ValueError(f"len(strides) must be equal to num_stages: {num_stages}")

        self.in_channels = in_channels

        # compute the number of filters per stage
        self.filters_per_stage = [
            min(
                start_features * (2**i),
                self.MAX_NUM_FILTERS_3D if dim == 3 else self.MAX_NUM_FILTERS_2D,
            )
            for i in range(num_stages + 1)
        ]

        down_layer = ConvLayer
        conv_bias = norm_layer == "instance"

        # first stage
        all_modules = [
            nn.Sequential(
                *[
                    down_layer(
                        self.in_channels,
                        self.filters_per_stage[0],
                        kernels[0],
                        strides[0],
                        dim,
                        conv_bias,
                        conv_kwargs,
                        None,
                        None,
                        activation,
                        activation_kwargs,
                        False,
                        None,
                    )
                ]
            )
        ]

        # intermediate stages
        for stage in range(1, num_stages):
            stage_modules = [
                down_layer(
                    self.filters_per_stage[stage - 1],
                    self.filters_per_stage[stage],
                    kernels[stage],
                    strides[stage],
                    dim,
                    conv_bias,
                    conv_kwargs,
                    norm_layer,
                    norm_kwargs,
                    activation,
                    activation_kwargs,
                    False,
                    None,
                )
            ]
            all_modules.append(nn.Sequential(*stage_modules))

        # last stage
        stage_modules = [
            down_layer(
                self.filters_per_stage[-2],
                self.filters_per_stage[-1],
                kernels[-1],
                1,
                dim,
                True,
                {"padding": 1},
                norm_layer,
                norm_kwargs,
                activation,
                activation_kwargs,
                False,
                None,
            )
        ]
        all_modules.append(nn.Sequential(*stage_modules))

        # output a single channel prediction map
        stage_modules = [
            down_layer(
                self.filters_per_stage[-1],
                1,
                kernels[-1],
                1,
                dim,
                True,
                {"padding": 1},
                None,
                None,
                None,
                None,
                False,
                None,
            )
        ]
        all_modules.append(nn.Sequential(*stage_modules))

        self.stages = nn.ModuleList(all_modules)
        self.strides = [[i] * dim if isinstance(i, int) else list(i) for i in strides]

        # initialize weights
        init_kwargs = {}
        if activation == "leakyrelu":
            if initialization == "kaiming_normal":
                if activation_kwargs is not None and "negative_slope" in activation_kwargs:
                    init_kwargs["neg_slope"] = activation_kwargs["negative_slope"]
                else:
                    init_kwargs["neg_slope"] = 0.01
        self.apply(get_initialization(initialization, **init_kwargs))

        # store some attributes that a potential decoder needs
        self.dim = dim
        self.kernels = kernels
        self.conv_bias = conv_bias
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.activation = activation
        self.activation_kwargs = activation_kwargs
        self.initialization = initialization
        self.init_kwargs = init_kwargs

    def forward(
        self, input_data: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor, list[Union[Tensor, MetaTensor]]]:  # noqa: D102
        out = input_data
        for stage in self.stages:
            out = stage(out)
        return out

    def get_output_shape(self, input_size: tuple[int, ...]) -> tuple[int, ...]:
        """Get the output shape of the discriminator.

        Args:
            input_size: Size of the input image. (H, W(, D))

        Returns:
            Output shape of the discriminator.

        Raises:
            ValueError: If length of `input_size` is not equal to `dim`.
        """
        input_tensor = torch.rand(
            (1, self.in_channels, *input_size), device=next(self.parameters()).device
        )
        with torch.no_grad():
            out = self(input_tensor)
        return tuple(list(out.shape[2:]))


if __name__ == "__main__":
    from torchinfo import summary

    # Example usage
    patch_size = (256, 256, 20)
    in_channels = 1
    kernels = [[4, 4, 1], [4, 4, 4], [4, 4, 4], [4, 4, 4]]
    strides = [[2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 1]]

    # Create a PatchGAN discriminator
    discriminator = PatchGAN(
        in_channels=in_channels,
        dim=len(kernels[0]),
        kernels=kernels,
        strides=strides,
        num_stages=4,
        conv_kwargs=None,
        norm_layer="instance",
        norm_kwargs=None,
        activation="leakyrelu",
        activation_kwargs={"inplace": True},
    )

    print(discriminator)

    # Test the discriminator with a dummy input
    dummy_input = torch.rand((2, in_channels, *patch_size))
    out = discriminator(dummy_input)
    print(discriminator.get_output_shape(patch_size))
    print(
        summary(
            discriminator,
            input_size=(2, in_channels, *patch_size),
            device="cpu",
            depth=6,
            col_names=("input_size", "output_size", "num_params"),
        )
    )
