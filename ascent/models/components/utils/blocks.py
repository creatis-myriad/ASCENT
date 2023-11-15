from typing import Literal, Optional, Union

import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torchvision.ops import StochasticDepth

from ascent.models.components.utils.layers import (
    AttentionLayer,
    ConvLayer,
    get_activation,
    get_conv,
    get_norm,
    get_transp_conv,
)


class ConvBlock(nn.Module):
    """Convolution block that consists of stacked ConvLayer(s)."""

    def __init__(
        self,
        num_conv: int,
        in_channels: int,
        out_channels: Union[int, tuple[int, ...], list[int]],
        kernel_size: Union[int, tuple[int, ...], list[int]],
        stride: Union[int, tuple[int, ...], list[int]],
        dim: int,
        conv_bias: bool = True,
        conv_kwargs: Optional[dict] = None,
        norm_layer: Union[None, Literal["instance", "batch", "group", "layer"]] = None,
        norm_kwargs: Optional[dict] = None,
        activation: Union[None, Literal["relu", "leakyrelu", "gelu", "tanh", "sigmoid"]] = None,
        activation_kwargs: Optional[dict] = None,
        drop_block: bool = False,
        drop_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize class instance.

        Args:
            num_conv: Number of convolution layers in the block.
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution. If a tuple or list is
                provided, it must have the same length as `num_conv`.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            dim: Dimension for convolution. (2 or 3)
            conv_bias: If True, adds a learnable bias to the convolution output.
            conv_kwargs: Keyword arguments for the convolution layer.
            norm_layer: Normalization to use. Can be either `instance`, `batch`,
                `group` or `layer`.
            norm_kwargs: Keyword arguments for the normalization layer.
            activation: Activation function to use. Can be either `relu`, `leakyrelu`, `gelu`,
                `tanh` or `sigmoid`.
            activation_kwargs: Keyword arguments for the activation function.
            drop_block: Whether to use drop out layers.
            drop_kwargs: Keyword arguments for `nn.Dropout2d` or `nn.Dropout3d`.

        Raises:
            ValueError: Error when `num_conv` is smaller than 0.
            ValueError: Error when len(out_channels) != num_conv if out_channels is a tuple or
                list.
        """
        super().__init__()
        if not num_conv > 0:
            raise ValueError("`num_conv` must be strictly greater than 0!")

        if isinstance(out_channels, int):
            out_channels = (out_channels,) * num_conv

        if not len(out_channels) == num_conv:
            raise ValueError(
                f"`out_channels` {out_channels} must be an integer or a tuple/list of length "
                f"{num_conv}!"
            )

        # Store stride, out_channels, and num_conv for computing total number of pixels/voxels in
        # the output feature map
        if isinstance(stride, int):
            self.stride = (stride,) * dim
        else:
            self.stride = stride
        self.out_channels = out_channels[-1]
        self.num_conv = num_conv

        convlayer_kwargs = {
            "dim": dim,
            "conv_bias": conv_bias,
            "conv_kwargs": conv_kwargs,
            "norm_layer": norm_layer,
            "norm_kwargs": norm_kwargs,
            "activation": activation,
            "activation_kwargs": activation_kwargs,
            "drop_block": drop_block,
            "drop_kwargs": drop_kwargs,
        }

        self.convs = nn.Sequential(
            ConvLayer(
                in_channels,
                out_channels[0],
                kernel_size,
                stride,
                **convlayer_kwargs,
            ),
            *[
                ConvLayer(
                    out_channels[i - 1],
                    out_channels[i],
                    kernel_size,
                    1,
                    **convlayer_kwargs,
                )
                for i in range(1, num_conv)
            ],
        )

    def forward(self, input_data: Tensor) -> Tensor:  # noqa: D102
        return self.convs(input_data)

    def compute_pixels_in_output_feature_map(self, input_size: tuple[int, ...]) -> int:
        """Compute total number of pixels/voxels in the output feature map after convolutions.

        Args:
            input_size: Size of the input image.

        Returns:
            Number of pixels/voxels in the output feature map after convolution.
        """
        if not len(input_size) == len(self.stride):
            raise ValueError(
                "`input_size` should be (H, W(, D)) without channel or batch dimension!"
            )

        output = self.convs[0].compute_pixels_in_output_feature_map(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        if self.num_conv > 1:
            for b in self.convs[1:]:
                output += b.compute_pixels_in_output_feature_map(size_after_stride)
        return output


class ResidBlock(nn.Module):
    """Residual convolution block that consists of stacked residual ConvLayer(s)."""

    def __init__(
        self,
        num_conv: int,
        in_channels: int,
        out_channels: Union[int, tuple[int, ...], list[int]],
        kernel_size: Union[int, tuple[int, ...], list[int]],
        stride: Union[int, tuple[int, ...], list[int]],
        dim: int,
        conv_bias: bool = True,
        conv_kwargs: Optional[dict] = None,
        norm_layer: Union[None, Literal["instance", "batch", "group", "layer"]] = None,
        norm_kwargs: Optional[dict] = None,
        activation: Union[None, Literal["relu", "leakyrelu", "gelu", "tanh", "sigmoid"]] = None,
        activation_kwargs: Optional[dict] = None,
        drop_block: bool = False,
        drop_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize class instance.

        Args:
            num_conv: Number of convolution layers in the block.
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution. If a tuple or list is
                provided, it must have the same length as `num_conv`.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            dim: Dimension for convolution. (2 or 3)
            conv_bias: If True, adds a learnable bias to the convolution output.
            conv_kwargs: Keyword arguments for the convolution layer.
            norm_layer: Normalization to use. Can be either `instance`, `batch`,
                `group` or `layer`.
            norm_kwargs: Keyword arguments for the normalization layer.
            activation: Activation function to use. Can be either `relu`, `leakyrelu`, `gelu`,
                `tanh` or `sigmoid`.
            activation_kwargs: Keyword arguments for the activation function.
            drop_block: Whether to use drop out layers.
            drop_kwargs: Keyword arguments for `nn.Dropout2d` or `nn.Dropout3d`.

        Raises:
            ValueError: Error when `num_conv` is smaller than 1.
            ValueError: Error when len(out_channels) != num_conv if out_channels is a tuple or
                list.
        """
        super().__init__()
        if not num_conv > 1:
            raise ValueError("`num_conv` must be greater than 1!")

        if isinstance(out_channels, int):
            out_channels = (out_channels,) * num_conv

        if not len(out_channels) == num_conv:
            raise ValueError(
                f"`out_channels` {out_channels} must be an integer or a tuple/list of length "
                f"{num_conv}!"
            )

        # Store stride, out_channels, and num_convs for computing total number of pixels/voxels in
        # the output feature map
        if isinstance(stride, int):
            self.stride = (stride,) * dim
        else:
            self.stride = stride
        self.out_channels = out_channels[-1]
        self.num_conv = num_conv

        convlayer_kwargs = {
            "dim": dim,
            "conv_bias": conv_bias,
            "conv_kwargs": conv_kwargs,
            "norm_layer": norm_layer,
            "norm_kwargs": norm_kwargs,
            "activation": activation,
            "activation_kwargs": activation_kwargs,
            "drop_block": drop_block,
            "drop_kwargs": drop_kwargs,
        }
        convlayer_kwargs_no_activation = {
            "dim": dim,
            "conv_bias": conv_bias,
            "conv_kwargs": conv_kwargs,
            "norm_layer": norm_layer,
            "norm_kwargs": norm_kwargs,
            "activation": None,
            "activation_kwargs": None,
            "drop_block": drop_block,
            "drop_kwargs": drop_kwargs,
        }
        self.convs = nn.Sequential(
            ConvLayer(
                in_channels,
                out_channels[0],
                kernel_size,
                stride,
                **convlayer_kwargs,
            ),
            *[
                ConvLayer(
                    out_channels[i - 1],
                    out_channels[i],
                    kernel_size,
                    1,
                    **convlayer_kwargs,
                )
                for i in range(1, num_conv - 1)
            ],
            ConvLayer(
                out_channels[-2],
                out_channels[-1],
                kernel_size,
                1,
                **convlayer_kwargs_no_activation,
            ),
        )
        self.activation = get_activation(activation, **activation_kwargs)
        self.resid_conv = None
        if max(self.stride) > 1 or not in_channels == out_channels:
            self.resid_conv = ConvLayer(
                in_channels,
                out_channels[-1],
                kernel_size,
                stride,
                **convlayer_kwargs_no_activation,
            )

    def forward(self, input_data: Tensor) -> Tensor:  # noqa: D102
        residual = input_data
        out = self.convs(input_data)
        if self.resid_conv is not None:
            residual = self.resid_conv(residual)
        out = self.activation(out + residual)
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
        if not len(input_size) == len(self.stride):
            raise ValueError(
                "`input_size` should be (H, W(, D)) without channel or batch dimension!"
            )

        # first conv
        output = self.convs[0].compute_pixels_in_output_feature_map(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        # following convs if applicable
        if self.num_conv > 1:
            for b in self.convs[1:]:
                output += b.compute_pixels_in_output_feature_map(size_after_stride)
        # skip residual conv if applicable
        if self.resid_conv is not None:
            output += self.resid_conv.compute_pixels_in_output_feature_map(input_size)
        return output


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block.

    References:
        Liu et al. "A ConvNet for the 2020s". IEEE/CVF CVPR 2022.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...], list[int]],
        stride: Union[int, tuple[int, ...], list[int]],
        dim: int,
        conv_bias: bool = True,
        expansion_rate: int = 4,
        conv_kwargs: Optional[dict] = None,
        norm_layer: Union[None, Literal["instance", "batch", "group", "layer"]] = "layer",
        norm_kwargs: Optional[dict] = None,
        activation: Union[None, Literal["relu", "leakyrelu", "gelu", "tanh", "sigmoid"]] = "gelu",
        activation_kwargs: Optional[dict] = None,
        drop_block: bool = False,
        drop_kwargs: Optional[dict] = None,
        stochastic_depth_p: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        """Initialize class instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            dim: Dimension for convolution. (2 or 3)
            conv_bias: If True, adds a learnable bias to the convolution output.
            conv_kwargs: Keyword arguments for the convolution layer.
            norm_layer: Normalization to use. Can be either `instance`, `batch`,
                `group` or `layer`.
            norm_kwargs: Keyword arguments for the normalization layer.
            activation: Activation function to use. Can be either `relu`, `leakyrelu`, `gelu`,
                `tanh` or `sigmoid`.
            activation_kwargs: Keyword arguments for the activation function.
            drop_block: Whether to use drop out layers.
            drop_kwargs: Keyword arguments for `nn.Dropout2d` or `nn.Dropout3d`.
            stochastic_depth_p: Stochastic depth rate.
            layer_scale_init_value: Init value for Layer Scale.

        Raises:
            ValueError: Error when `expansion_rate` is smaller than 1.
        """
        super().__init__()
        if conv_kwargs is None:
            conv_kwargs = {}
        if norm_kwargs is None:
            norm_kwargs = {}
        if activation_kwargs is None:
            activation_kwargs = {}
        if "groups" not in conv_kwargs:
            conv_kwargs["groups"] = in_channels

        if not expansion_rate > 0:
            raise ValueError("`expansion_rate` must be strictly greater than 0!")

        # Store stride and in_channels for computing total number of pixels/voxels in
        # the output feature map
        if isinstance(stride, int):
            self.stride = (stride,) * dim
        else:
            self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        convlayer_kwargs_no_norm_no_activation = {
            "dim": dim,
            "conv_bias": conv_bias,
            "conv_kwargs": conv_kwargs,
            "norm_layer": None,
            "norm_kwargs": None,
            "activation": None,
            "activation_kwargs": None,
            "drop_block": drop_block,
            "drop_kwargs": drop_kwargs,
        }

        modules = []

        # depthwise conv
        conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            **convlayer_kwargs_no_norm_no_activation,
        )
        modules.append(conv)

        # normalization
        if norm_layer == "layer" and "data_format" not in norm_kwargs:
            norm_kwargs["data_format"] = "channel_first"
        modules.append(get_norm(norm_layer, in_channels, dim, **norm_kwargs))

        # permute to channels_last
        if dim == 2:
            modules.append(Rearrange("b c h w -> b h w c"))
        else:
            modules.append(Rearrange("b c h w d -> b h w d c"))

        # pointwise/1x1(x1) convs, implemented with linear layers
        pwconv1 = nn.Linear(in_channels, expansion_rate * in_channels)
        modules.append(pwconv1)

        activation = get_activation(activation, **activation_kwargs)
        modules.append(activation)

        # pointwise/1x1(x1) convs, implemented with linear layers
        pwconv2 = nn.Linear(expansion_rate * in_channels, out_channels)
        modules.append(pwconv2)

        # permute back to channels_first
        if dim == 2:
            modules.append(Rearrange("b h w c -> b c h w"))
        else:
            modules.append(Rearrange("b h w d c -> b c h w d"))

        self.block = nn.Sequential(*modules)

        self.layer_scale = (
            nn.Parameter(
                layer_scale_init_value * torch.ones(out_channels, *((1,) * dim)),
            )
            if layer_scale_init_value > 0
            else None
        )

        # stochastic depth
        if stochastic_depth_p > 0:
            self.stochastic_depth = StochasticDepth(stochastic_depth_p, "row")

    def forward(self, input_data: Tensor) -> Tensor:  # noqa: D102
        out = self.block(input_data)
        if self.layer_scale is not None:
            out = self.layer_scale * out
        if hasattr(self, "stochastic_depth"):
            out = self.stochastic_depth(out)
        out += input_data
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
        if not len(input_size) == len(self.stride):
            raise ValueError(
                "`input_size` should be (H, W(, D)) without channel or batch dimension!"
            )

        # depthwise conv
        output = self.block[0].compute_pixels_in_output_feature_map(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]

        # pointwise conv 1
        output += np.prod(size_after_stride) * self.in_channels * 4

        # pointwise conv 2
        output += np.prod(size_after_stride) * self.out_channels
        return output


class UpsampleBlock(nn.Module):
    """Upsample block that consists of a transposed convolution layer, an optional attention layer,
    and a double ConvLayer."""

    def __init__(
        self,
        num_conv: int,
        in_channels: int,
        out_channels: Union[int, tuple[int, ...], list[int]],
        kernel_size: Union[int, tuple[int, ...]],
        stride: Union[int, tuple[int, ...]],
        dim: int,
        attention: bool = False,
        conv_bias: bool = True,
        conv_kwargs: Optional[dict] = None,
        norm_layer: Optional[Literal["instance", "batch", "group", "layer"]] = None,
        norm_kwargs: Optional[dict] = None,
        activation: Optional[Literal["relu", "leakyrelu", "gelu", "tanh", "sigmoid"]] = None,
        activation_kwargs: Optional[dict] = None,
        drop_block: bool = False,
        drop_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize class instance.

        Args:
            num_conv: Number of convolution layers in the block.
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution. If a tuple or list is
                provided, it must have the same length as `num_conv`.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            dim: Dimension for convolution. (2 or 3)
            attention: Whether to use attention layers.
            conv_bias: If True, adds a learnable bias to the convolution output.
            conv_kwargs: Keyword arguments for the convolution layer.
            norm_layer: Normalization to use. Can be either `instance`, `batch`,
                `group` or `layer`.
            norm_kwargs: Keyword arguments for the normalization layer.
            activation: Activation function to use. Can be either `relu`, `leakyrelu`, `gelu`,
                `tanh` or `sigmoid`.
            activation_kwargs: Keyword arguments for the activation function.
            drop_block: Whether to use drop out layers.
            drop_kwargs: Keyword arguments for `nn.Dropout2d` or `nn.Dropout3d`.

        Raises:
            ValueError: Error when len(kernel_size) != dim if kernel_size is a tuple or list.
        """
        super().__init__()
        if not num_conv > 0:
            raise ValueError("`num_conv` must be strictly greater than 0!")

        if isinstance(out_channels, int):
            out_channels = (out_channels,) * num_conv

        if not len(out_channels) == num_conv:
            raise ValueError(
                f"`out_channels` {out_channels} must be an integer or a tuple/list of length "
                f"{num_conv}!"
            )

        # Store stride, in_channels, out_channels, and num_conv for computing total number of
        # pixels/voxels in the output feature map
        if isinstance(stride, int):
            self.stride = (stride,) * dim
        else:
            self.stride = stride
        self.trans_conv_out_channels = out_channels[0]
        self.out_channels = out_channels[-1]
        self.num_conv = num_conv

        if conv_kwargs is None:
            conv_kwargs = {}
        if norm_kwargs is None:
            norm_kwargs = {}
        if activation_kwargs is None:
            activation_kwargs = {}
        if drop_kwargs is None:
            drop_kwargs = {}

        convlayer_kwargs = {
            "dim": dim,
            "conv_bias": conv_bias,
            "conv_kwargs": conv_kwargs,
            "norm_layer": norm_layer,
            "norm_kwargs": norm_kwargs,
            "activation": activation,
            "activation_kwargs": activation_kwargs,
            "drop_block": drop_block,
            "drop_kwargs": drop_kwargs,
        }

        self.transp_conv = get_transp_conv(
            in_channels, out_channels[0], stride, stride, dim, conv_bias, **conv_kwargs
        )
        self.conv_block = ConvBlock(
            num_conv, 2 * out_channels[0], out_channels, kernel_size, 1, **convlayer_kwargs
        )
        self.attention = attention
        if self.attention:
            att_out = out_channels[0] // 2
            self.conv_o = AttentionLayer(
                out_channels[0], att_out, dim, conv_bias, conv_kwargs, norm_layer, norm_kwargs
            )
            self.conv_s = AttentionLayer(
                out_channels[0], att_out, dim, conv_bias, conv_kwargs, norm_layer, norm_kwargs
            )
            self.psi = AttentionLayer(
                att_out, 1, dim, conv_bias, conv_kwargs, norm_layer, norm_kwargs
            )
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data: Tensor, skip_data: Tensor) -> Tensor:  # noqa: D102
        out = self.transp_conv(input_data)
        if self.attention:
            out_a = self.conv_o(out)
            skip_a = self.conv_s(skip_data)
            psi_a = self.psi(self.relu(out_a + skip_a))
            attention = self.sigmoid(psi_a)
            skip_data = skip_data * attention
        out = torch.cat((out, skip_data), dim=1)
        out = self.conv_block(out)
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
        if not len(input_size) == len(self.stride):
            raise ValueError(
                "`input_size` should be (H, W(, D)) without channel or batch dimension!"
            )
        skip_sizes = tuple([int(i * j) for i, j in zip(input_size, self.stride)])

        # transposed conv
        output = np.prod([self.trans_conv_out_channels, *skip_sizes], dtype=np.int64)

        # conv block
        output += self.conv_block.compute_pixels_in_output_feature_map(skip_sizes)

        # attention layer
        if self.attention:
            output += self.conv_o.compute_pixels_in_output_feature_map(skip_sizes)
            output += self.conv_s.compute_pixels_in_output_feature_map(skip_sizes)
            output += self.psi.compute_pixels_in_output_feature_map(skip_sizes)
        return output


class OutputBlock(nn.Module):
    """Output convolution block that includes only a single convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        conv_bias: bool = True,
        conv_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dim: Dimension for convolution. (2 or 3)
            conv_bias: If True, adds a learnable bias to the convolution output.
            conv_kwargs: Keyword arguments for the convolution layer.
        """
        super().__init__()
        # store stride and out_channels for computing total number of pixels/voxels in the output
        # feature map
        self.out_channels = out_channels
        self.stride = (1,) * dim

        if conv_kwargs is None:
            conv_kwargs = {}
        self.conv = get_conv(in_channels, out_channels, 1, 1, dim, conv_bias, **conv_kwargs)
        if conv_bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, input_data: Tensor) -> Tensor:  # noqa: D102
        return self.conv(input_data)

    def compute_pixels_in_output_feature_map(self, input_size: tuple[int, ...]) -> int:
        """Compute total number of pixels/voxels in the output feature map after convolutions.

        Args:
            input_size: Size of the input image. (H, W(, D))

        Returns:
            Number of pixels/voxels in the output feature map after convolution.

        Raises:
            ValueError: If length of `input_size` is not equal to `dim`.
        """
        if not len(input_size) == len(self.stride):
            raise ValueError(
                "`input_size` should be (H, W(, D)) without channel or batch dimension!"
            )

        return np.prod([self.out_channels, *input_size], dtype=np.int64)
