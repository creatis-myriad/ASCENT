from typing import Literal, Union

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn

from ascent.models.components.unet_related.drop import DropPath
from ascent.models.components.unet_related.normalization import LayerNorm

normalizations = {
    "instancenorm3d": nn.InstanceNorm3d,
    "instancenorm2d": nn.InstanceNorm2d,
    "batchnorm3d": nn.BatchNorm3d,
    "batchnorm2d": nn.BatchNorm2d,
}

convolutions = {
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "ConvTranspose2d": nn.ConvTranspose2d,
    "ConvTranspose3d": nn.ConvTranspose3d,
}

dropouts = {
    "Dropout2d": nn.Dropout2d,
    "Dropout3d": nn.Dropout3d,
}


def get_norm(
    name: Literal[
        "batchnorm2d", "batchnorm3d", "instancenorm2d", "instancenorm3d", "groupnorm", "layernorm"
    ],
    num_channels: int,
    **kwargs,
):
    """Get normalization layer.

    Args:
        name: Name of normalization layer to use.
        num_channels: Number of channels expected in input.
        **kwargs: Keyword arguments to be passed to the normalization layer.

    Returns:
        PyTorch normalization layer with learnable parameters.
    """
    if "groupnorm" in name:
        return nn.GroupNorm(32, num_channels, affine=True, **kwargs)
    elif "layernorm" in name:
        return LayerNorm(normalized_shape=num_channels, eps=1e-6, **kwargs)
    return normalizations[name](num_channels, affine=True, **kwargs)


def get_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, tuple[int, ...]],
    stride: Union[int, tuple[int, ...]],
    dim: Literal[2, 3],
    bias: bool = True,
    **kwargs,
):
    """Get 2D or 3D convolution layer.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dim: Dimension of convolution.
        bias: Set True to add a learnable bias to the output.
        **kwargs: Keyword arguments to be passed to either Conv2d or Conv3d.

    Returns:
        PyTorch convolution layer.
    """
    conv = convolutions[f"Conv{dim}d"]
    padding = get_padding(kernel_size, stride)
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs)


def get_transp_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, tuple[int, ...]],
    stride: Union[int, tuple[int, ...]],
    dim: Literal[2, 3],
    bias: bool = True,
):
    """Get 2D or 3D transposed convolution layer.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dim: Dimension of convolution.
        bias: Set True to add a learnable bias to the output.

    Returns:
        PyTorch transposed convolution layer.
    """
    conv = convolutions[f"ConvTranspose{dim}d"]
    padding = get_padding(kernel_size, stride)
    output_padding = get_output_padding(kernel_size, stride, padding)
    return conv(
        in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False
    )


def get_padding(
    kernel_size: Union[int, tuple[int, ...]], stride: Union[int, tuple[int, ...]]
) -> Union[int, tuple[int, ...]]:
    """Compute padding based on kernel size and stride.

    Args:
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.

    Returns:
        Padding, either int or tuple.
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[int, tuple[int, ...]],
    stride: Union[int, tuple[int, ...]],
    padding: Union[int, tuple[int, ...]],
) -> Union[int, tuple[int, ...]]:
    """Compute output padding based on kernel size, stride, and input padding.

    Args:
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to input.

    Returns:
        Output padding, either int or tuple.
    """
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


def get_drop_block(dim: Literal[2, 3], p: float = 0.5, inplace: bool = True):
    """Get 2D or 3D drop out layer.

    Args:
        dim: Dimension of drop out layer.

    Returns:
        PyTorch drop out layer.
    """
    drop = dropouts[f"Dropout{dim}d"]
    return drop(p=p, inplace=inplace)


class ConvLayer(nn.Module):
    """Convolution layer that consists of convolution, normalization, leakyRELU, and optional drop
    out layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        stride: Union[int, tuple[int, ...]],
        **kwargs,
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            **dim (int): Dimension for convolution. (2 or 3)
            **norm (str): Name of normalization layer.
            **negative_slope (float): Negative slope to be used in leakyRELU.
            **drop_block (bool): Set true to add a drop out layer.
        """
        super().__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size, stride, kwargs["dim"])
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=kwargs["negative_slope"], inplace=True)
        self.use_drop_block = kwargs["drop_block"]
        if self.use_drop_block:
            self.drop_block = get_drop_block(kwargs["dim"])

    def forward(self, data: Tensor):  # noqa: D102
        out = self.conv(data)
        if self.use_drop_block:
            out = self.drop_block(out)
        out = self.norm(out)
        out = self.lrelu(out)
        return out


class ConvBlock(nn.Module):
    """Convolution block that consists of double ConvLayer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        stride: Union[int, tuple[int, ...]],
        **kwargs,
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            **dim (int): Dimension for convolution. (2 or 3)
            **norm (str): Name of normalization layer.
            **negative_slope (float): Negative slope to be used in leakyRELU.
            **drop_block (bool): Set true to add a drop out layer.
        """
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, 1, **kwargs)

    def forward(self, input_data: Tensor):  # noqa: D102
        out = self.conv1(input_data)
        out = self.conv2(out)
        return out


class ResidBlock(nn.Module):
    """Residual convolution block that consists of double ConvLayer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        stride: Union[int, tuple[int, ...]],
        **kwargs,
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            **dim (int): Dimension for convolution. (2 or 3)
            **norm (str): Name of normalization layer.
            **negative_slope (float): Negative slope to be used in leakyRELU.
            **drop_block (bool): Set true to add a drop out layer.
        """
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.conv2 = get_conv(out_channels, out_channels, kernel_size, 1, kwargs["dim"])
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=kwargs["negative_slope"], inplace=True)
        self.use_drop_block = kwargs["drop_block"]
        if self.use_drop_block:
            self.drop_block = get_drop_block(kwargs["dim"])
            self.skip_drop_block = get_drop_block(kwargs["dim"])
        self.downsample = None
        if max(stride) > 1 or in_channels != out_channels:
            self.downsample = get_conv(
                in_channels, out_channels, kernel_size, stride, kwargs["dim"]
            )
            self.norm_res = get_norm(kwargs["norm"], out_channels)

    def forward(self, input_data):  # noqa: D102
        residual = input_data
        out = self.conv1(input_data)
        out = self.conv2(out)
        if self.use_drop_block:
            out = self.drop_block(out)
        out = self.norm(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
            if self.use_drop_block:
                residual = self.skip_drop_block(residual)
            residual = self.norm_res(residual)
        out = self.lrelu(out + residual)
        return out


class AttentionLayer(nn.Module):
    """Attention layer that consists of a convolution and normalization layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: Literal[
            "batchnorm2d", "batchnorm3d", "instancenorm2d", "instancenorm3d", "groupnorm"
        ],
        dim: Literal[2, 3],
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dim: Dimension for convolution. (2 or 3)
            norm: Name of normalization layer.
        """
        super().__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size=3, stride=1, dim=dim)
        self.norm = get_norm(norm, out_channels)

    def forward(self, inputs: Tensor):  # noqa: D102
        out = self.conv(inputs)
        out = self.norm(out)
        return out


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block.

    There are two equivalent implementations: (1) Conv -> LayerNorm (channels_first) -> 1x1 Conv ->
    GELU -> 1x1 Conv; all in (N, C, H, W, D) (2) Conv -> Permute to (N, H, W, D, C); LayerNorm
    (channels_last) -> Linear -> GELU -> Linear; Permute back We use (2) as we find it slightly
    faster in PyTorch
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        stride: Union[int, tuple[int, ...]],
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        **kwargs,
    ):
        """Initialize class instance.

        Args:
            in_channels: Number of input channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            drop_path: Stochastic depth rate.
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
            **dim (int): Dimension for convolution. (2 or 3)
            **norm (str): Name of normalization layer.
        """
        super().__init__()
        self.conv = get_conv(
            in_channels, in_channels, kernel_size, stride, kwargs["dim"], groups=in_channels
        )  # depthwise conv
        self.norm = get_norm(kwargs["norm"], in_channels)
        self.pwconv1 = nn.Linear(
            in_channels, 4 * in_channels
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(in_channels), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv(x)
        x = rearrange(x, "b c h w d -> b h w d c")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, "b h w d c -> b c h w d")

        x = input + self.drop_path(x)
        return x


class UpsampleBlock(nn.Module):
    """Upsample block that consists of a transposed convolution layer, an optional attention layer,
    and a double ConvLayer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        stride: Union[int, tuple[int, ...]],
        bias: bool = False,
        **kwargs,
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            bias: Set True to add a learnable bias to the output.
            **dim (int): Dimension for convolution. (2 or 3)
            **norm (str): Name of normalization layer.
            **attention (bool): Set true to add an attention layer.
        """
        super().__init__()
        self.transp_conv = get_transp_conv(
            in_channels, out_channels, stride, stride, kwargs["dim"], bias=bias
        )
        self.conv_block = ConvBlock(2 * out_channels, out_channels, kernel_size, 1, **kwargs)
        self.attention = kwargs["attention"]
        if self.attention:
            att_out, norm, dim = out_channels // 2, kwargs["norm"], kwargs["dim"]
            self.conv_o = AttentionLayer(out_channels, att_out, norm, dim)
            self.conv_s = AttentionLayer(out_channels, att_out, norm, dim)
            self.psi = AttentionLayer(att_out, 1, norm, dim)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data: Tensor, skip_data: Tensor):  # noqa: D102
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


class OutputBlock(nn.Module):
    """Output convolution block that includes only a single convolution layer."""

    def __init__(
        self, in_channels: int, out_channels: int, dim: Literal[2, 3], bias: bool = False
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            dim: Dimension for convolution. (2 or 3)
        """
        super().__init__()
        self.conv = get_conv(
            in_channels, out_channels, kernel_size=1, stride=1, dim=dim, bias=bias
        )
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, input_data):  # noqa: D102
        return self.conv(input_data)
