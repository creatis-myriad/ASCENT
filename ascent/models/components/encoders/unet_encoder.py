from typing import Literal, Optional, Union

import numpy as np
import torch
from monai.data import MetaTensor
from omegaconf.listconfig import ListConfig
from torch import Tensor, nn

from ascent.models.components.utils.blocks import ConvBlock, ResidBlock
from ascent.models.components.utils.initialization import get_initialization
from ascent.models.components.utils.layers import get_pooling


class UNetEncoder(nn.Module):
    """A generic U-Net encoder that can be instantiated dynamically."""

    # define some static class attributes
    MAX_NUM_FILTERS_2D = 480
    MAX_NUM_FILTERS_3D = 320

    def __init__(
        self,
        in_channels: int,
        num_stages: int,
        dim: int,
        kernels: Union[int, list[Union[int, list[int]]], tuple[Union[int, tuple[int, ...]], ...]],
        strides: Union[int, list[Union[int, list[int]]], tuple[Union[int, tuple[int, ...]], ...]],
        start_features: int = 32,
        num_conv_per_stage: Union[int, list[int], tuple[int, ...]] = 2,
        conv_bias: bool = True,
        conv_kwargs: Optional[dict] = None,
        pooling: Literal["max", "avg", "stride"] = "stride",
        adaptive_pooling: bool = False,
        norm_layer: Optional[Literal["instance", "batch", "group", "layer"]] = "instance",
        norm_kwargs: Optional[dict] = None,
        activation: Optional[
            Literal["relu", "leakyrelu", "gelu", "tanh", "sigmoid"]
        ] = "leakyrelu",
        activation_kwargs: Optional[dict] = None,
        drop_block: bool = False,
        drop_kwargs: Optional[dict] = None,
        residual: bool = False,
        return_skip: bool = True,
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
            num_conv_per_stage: Number of convolutional layers per stage.
            conv_bias: Whether to use bias in the convolutional layers.
            conv_kwargs: Keyword arguments for the convolution layers.
            pooling: Pooling to use. Can be either `max`, `avg` or `stride`.
            adaptive_pooling: Whether to use adaptive pooling in case of `max` or `avg` pooling.
            norm_layer: Normalization to use. Can be either `instance`, `batch`,
                `group` or `layer`.
            norm_kwargs: Keyword arguments for the normalization layers.
            activation: Activation function to use. Can be either `relu`, `leakyrelu`, `gelu`,
                `tanh` or `sigmoid`.
            activation_kwargs: Keyword arguments for the activation functions.
            drop_block: Whether to use drop out layers.
            drop_kwargs: Keyword arguments for the drop out layers: `nn.Dropout2d` or
                `nn.Dropout3d`.
            residual: Whether to use residual block.
            return_skip: Whether to return the skip connections.
            initialization: Weight initialization technique. Can be either `kaiming_normal` or
                `xavier_uniform`.

        Raises:
            ValueError: When `len(kernels)` is not equal to `num_stages`.
            ValueError: When `len(strides)` is not equal to `num_stages`.
            ValueError: When `len(num_conv_per_stage)` is not equal to `num_stages`.
        """
        super().__init__()

        if isinstance(kernels, int):
            kernels = (kernels,) * num_stages

        if isinstance(strides, int):
            strides = (strides,) * num_stages

        if isinstance(num_conv_per_stage, int):
            num_conv_per_stage = (num_conv_per_stage,) * num_stages

        if not len(kernels) == num_stages:
            raise ValueError(f"len(kernels) must be equal to num_stages: {num_stages}")

        if not len(strides) == num_stages:
            raise ValueError(f"len(strides) must be equal to num_stages: {num_stages}")

        if not len(num_conv_per_stage) == num_stages:
            raise ValueError(f"len(num_conv_per_stage) must be equal to num_stages: {num_stages}")

        self.in_channels = in_channels

        # compute the number of filters per stage
        self.filters_per_stage = [
            min(
                start_features * (2**i),
                self.MAX_NUM_FILTERS_3D if dim == 3 else self.MAX_NUM_FILTERS_2D,
            )
            for i in range(num_stages)
        ]

        # determine to use whether residual or convolutional blocks
        down_block = ResidBlock if residual else ConvBlock

        all_modules = []
        for stage in range(num_stages):
            stage_modules = []
            conv_stride = strides[stage]
            if pooling in ["max", "avg"]:
                if (
                    (isinstance(strides[stage], int) and not strides[stage] == 1)
                    or isinstance(strides[stage], (tuple, list, ListConfig))
                    and not all(i == 1 for i in strides[stage])
                ):
                    stage_modules.append(
                        get_pooling(pooling, dim, adaptive_pooling)(
                            kernel_size=strides[stage], stride=strides[stage]
                        )
                    )
                    conv_stride = 1
            stage_modules.append(
                down_block(
                    num_conv_per_stage[stage],
                    in_channels,
                    self.filters_per_stage[stage],
                    kernels[stage],
                    conv_stride,
                    dim,
                    conv_bias,
                    conv_kwargs,
                    norm_layer,
                    norm_kwargs,
                    activation,
                    activation_kwargs,
                    drop_block,
                    drop_kwargs,
                )
            )
            all_modules.append(nn.Sequential(*stage_modules))
            in_channels = self.filters_per_stage[stage]

        self.stages = nn.ModuleList(all_modules)
        self.return_skip = return_skip
        self.strides = [[i] * dim if isinstance(i, int) else list(i) for i in strides]

        # initialize weights
        init_kwargs = {}
        if activation == "leakyrelu":
            if activation_kwargs is not None and "negative_slope" in activation_kwargs:
                init_kwargs["neg_slope"] = activation_kwargs["negative_slope"]
        self.apply(get_initialization(initialization, **init_kwargs))

        # store some attributes that a potential decoder needs
        self.dim = dim
        self.kernels = kernels
        self.conv_bias = conv_bias
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.activation = activation
        self.activation_kwargs = activation_kwargs
        self.drop_block = drop_block
        self.drop_kwargs = drop_kwargs
        self.initialization = initialization
        self.init_kwargs = init_kwargs

    def forward(
        self, input_data: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor, list[Union[Tensor, MetaTensor]]]:  # noqa: D102
        out = input_data
        encoder_outputs = []
        for stage in self.stages:
            out = stage(out)
            encoder_outputs.append(out)
        if self.return_skip:
            return encoder_outputs
        else:
            return out

    def compute_pixels_in_output_feature_map(self, input_size: tuple[int, ...]) -> int:
        """Compute total number of pixels/voxels in the output feature map after convolutions.

        Args:
            input_size: Size of the input image. (H, W(, D))

        Returns:
            Number of pixels/voxels in the output feature map after convolution.

        Raises:
            ValueError: If length of `input_size` is not equal to `dim`.
        """
        if not len(input_size) == len(self.strides[0]):
            raise ValueError(
                "`input_size` should be (H, W(, D)) without channel or batch dimension!"
            )
        output = self.in_channels * np.prod(input_size, dtype=np.int64)
        stride_pooling = True
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if isinstance(
                        sq,
                        (
                            nn.AvgPool2d,
                            nn.AvgPool3d,
                            nn.MaxPool2d,
                            nn.MaxPool3d,
                            nn.AdaptiveAvgPool2d,
                            nn.AdaptiveAvgPool3d,
                            nn.AdaptiveMaxPool2d,
                            nn.AdaptiveMaxPool3d,
                        ),
                    ):
                        stride_pooling = False
                        input_size = [i // j for i, j in zip(input_size, self.strides[s])]
                    if hasattr(sq, "compute_pixels_in_output_feature_map"):
                        output += self.stages[s][-1].compute_pixels_in_output_feature_map(
                            input_size
                        )
            else:
                output += self.stages[s].compute_pixels_in_output_feature_map(input_size)
            if stride_pooling:
                input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


if __name__ == "__main__":
    from torchinfo import summary

    patch_size = (640, 1024)
    in_channels = 1
    kernels = [[3, 1], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    strides = [[1, 1], [1, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

    unet_encoder = UNetEncoder(
        in_channels=in_channels,
        num_stages=len(kernels),
        dim=2,
        kernels=kernels,
        strides=strides,
        start_features=32,
        num_conv_per_stage=2,
        conv_bias=True,
        pooling="stride",
        adaptive_pooling=False,
        conv_kwargs=None,
        norm_layer="instance",
        norm_kwargs=None,
        activation="leakyrelu",
        activation_kwargs={"inplace": True},
        drop_block=False,
        drop_kwargs=None,
        residual=False,
        return_skip=True,
    )

    print(unet_encoder)
    dummy_input = torch.rand((2, in_channels, *patch_size))
    out = unet_encoder(dummy_input)
    print(unet_encoder.compute_pixels_in_output_feature_map(patch_size))
    print(
        summary(
            unet_encoder,
            input_size=(2, in_channels, *patch_size),
            device="cpu",
            depth=6,
            col_names=("input_size", "output_size", "num_params"),
        )
    )
