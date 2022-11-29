from typing import Literal, Union

import numpy as np
import torch
from torch import Tensor, nn

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
    name: Literal["batchnorm2d", "batchnorm3d", "instancenorm2d", "instancenorm3d", "groupnorm"],
    num_channels: int,
):
    """Get normalization layer.

    Args:
        name: Name of normalization layer to use.
        num_channels: Number of channels expected in input.

    Returns:
        PyTorch normalization layer with learnable parameters.
    """
    if "groupnorm" in name:
        return nn.GroupNorm(32, num_channels, affine=True)
    return normalizations[name](num_channels, affine=True)


def get_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, tuple[int, ...]],
    stride: Union[int, tuple[int, ...]],
    dim: Literal[2, 3],
    bias: bool = True,
):
    """Get 2D or 3D convolution layer.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dim: Dimension of convolution.
        bias: Set True to add a learnable bias to the output.

    Returns:
        PyTorch convolution layer.
    """
    conv = convolutions[f"Conv{dim}d"]
    padding = get_padding(kernel_size, stride)
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)


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
            self.drop_block = get_drop_block()

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
            self.drop_block = get_drop_block()
            self.skip_drop_block = get_drop_block()
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
