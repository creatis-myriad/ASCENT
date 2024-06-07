from typing import Literal, Optional, Union

import numpy as np
from monai.utils import has_option
from torch import Tensor, nn

from ascent.models.components.utils.normalization import LayerNorm

convolutions = {
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "ConvTranspose2d": nn.ConvTranspose2d,
    "ConvTranspose3d": nn.ConvTranspose3d,
}

normalizations = {
    "instancenorm1d": nn.InstanceNorm1d,
    "instancenorm2d": nn.InstanceNorm2d,
    "instancenorm3d": nn.InstanceNorm3d,
    "batchnorm1d": nn.BatchNorm1d,
    "batchnorm2d": nn.BatchNorm2d,
    "batchnorm3d": nn.BatchNorm3d,
    "groupnorm": nn.GroupNorm,
    "layernorm": LayerNorm,
}

activations = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

dropouts = {
    "Dropout1d": nn.Dropout1d,
    "Dropout2d": nn.Dropout2d,
    "Dropout3d": nn.Dropout3d,
}

poolings = {
    "AvgPool1d": nn.AvgPool1d,
    "AvgPool2d": nn.AvgPool2d,
    "AvgPool3d": nn.AvgPool3d,
    "MaxPool1d": nn.MaxPool1d,
    "MaxPool2d": nn.MaxPool2d,
    "MaxPool3d": nn.MaxPool3d,
    "AdaptiveAvgPool1d": nn.AdaptiveAvgPool1d,
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
    "AdaptiveAvgPool3d": nn.AdaptiveAvgPool3d,
    "AdaptiveMaxPool1d": nn.AdaptiveMaxPool1d,
    "AdaptiveMaxPool2d": nn.AdaptiveMaxPool2d,
    "AdaptiveMaxPool3d": nn.AdaptiveMaxPool3d,
}


def get_norm(
    name: Literal["batch", "instance", "group", "layer"],
    num_channels: Optional[int] = None,
    dim: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Get normalization layer.

    Args:
        name: Name of normalization layer to use.
        num_channels: Number of channels expected in input.
        dim: Dimension for normalization layer. (1, 2 or 3). Required for `batch` and `instance`.
        **kwargs: Keyword arguments to be passed to the normalization layer.

    Returns:
        PyTorch normalization layer with learnable parameters.

    Raises:
        NotImplementedError: If dimension of normalization is not 1, 2 or 3.
        NotImplementedError: If normalization layer is not one of `batch`, `instance`, `group`, or
            `layer`.
    """
    if dim is not None:
        if dim not in [1, 2, 3]:
            raise NotImplementedError(f"{dim}D normalization is not supported right now!")
    if name in ["instance", "batch"]:
        norm_name = f"{name}norm{dim}d"
    elif name in ["group", "layer"]:
        norm_name = name + "norm"
    else:
        raise NotImplementedError(f"Normalization layer {name} is not supported!")

    norm_type = normalizations[norm_name]

    if has_option(norm_type, "num_features") and "num_features" not in kwargs:
        kwargs["num_features"] = num_channels
    if has_option(norm_type, "num_channels") and "num_channels" not in kwargs:
        kwargs["num_channels"] = num_channels
    if has_option(norm_type, "num_groups") and "num_groups" not in kwargs:
        kwargs["num_groups"] = num_channels
    if has_option(norm_type, "normalized_shape") and "normalized_shape" not in kwargs:
        kwargs["normalized_shape"] = num_channels

    return norm_type(**kwargs)


def get_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, tuple[int, ...], list[int]],
    stride: Union[int, tuple[int, ...], list[int]],
    dim: int,
    conv_bias: bool = True,
    padding: Optional[Union[int, tuple[int, ...], list[int]]] = None,
    **kwargs,
) -> nn.Module:
    """Get 2D or 3D convolution layer.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dim: Dimension of convolution.
        conv_bias: If True, adds a learnable bias to the convolution output
        padding: Padding added to input. If None, padding is computed based on kernel size and
            stride.
        **kwargs: Keyword arguments to be passed to either `nn.Conv2d` or `nn.Conv3d`.

    Returns:
        PyTorch convolution layer.

    Raises:
        NotImplementedError: If dimension of convolution is not 2 or 3.
    """
    if dim not in [2, 3]:
        raise NotImplementedError(f"{dim}D convolution is not supported right now!")
    conv = convolutions[f"Conv{dim}d"]
    pad = get_padding(kernel_size, stride) if padding is None else padding
    return conv(in_channels, out_channels, kernel_size, stride, pad, bias=conv_bias, **kwargs)


def get_transp_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, tuple[int, ...], list[int]],
    stride: Union[int, tuple[int, ...], list[int]],
    dim: int,
    conv_bias: bool = True,
    **kwargs,
) -> nn.Module:
    """Get 2D or 3D transposed convolution layer.

    Args:
        in_channels: Number of channels in the input image.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dim: Dimension of convolution.
        conv_bias: If True, adds a learnable bias to the convolution output
        **kwargs: Keyword arguments to be passed to either `nn.ConvTranspose2d` or
            `nn.ConvTranspose3d`.

    Returns:
        PyTorch transposed convolution layer.

    Raises:
        NotImplementedError: If dimension of transposed convolution is not 2 or 3.
    """
    if dim not in [2, 3]:
        raise NotImplementedError(f"{dim}D transposed convolution is not supported right now!")
    conv = convolutions[f"ConvTranspose{dim}d"]
    padding = get_padding(kernel_size, stride)
    output_padding = get_output_padding(kernel_size, stride, padding)
    return conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias=conv_bias,
        **kwargs,
    )


def get_padding(
    kernel_size: Union[int, tuple[int, ...], list[int]],
    stride: Union[int, tuple[int, ...], list[int]],
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
    kernel_size: Union[int, tuple[int, ...], list[int]],
    stride: Union[int, tuple[int, ...], list[int]],
    padding: Union[int, tuple[int, ...], list[int]],
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


def get_drop_block(dim: int, p: float = 0.5, inplace: bool = True, **kwargs) -> nn.Module:
    """Get 1D/2D/3D drop out layer.

    Args:
        dim: Dimension of drop out layer.
        p: Probability of an element to be zeroed.
        inplace: If set to True, will do this operation in-place.
        **kwargs: Keyword arguments to be passed to either `nn.Dropout1d` or `nn.Dropout2d` or
            `nn.Dropout3d`.

    Returns:
        PyTorch drop out layer.

    Raises:
        NotImplementedError: If dimension of convolution is not 2 or 3.
    """
    if dim not in [1, 2, 3]:
        raise NotImplementedError(f"{dim}D dropout is not supported right now!")
    drop = dropouts[f"Dropout{dim}d"]
    return drop(p=p, inplace=inplace, **kwargs)


def get_activation(
    name: Literal["relu", "leakyrelu", "gelu", "tanh", "sigmoid"], **kwargs
) -> nn.Module:
    """Get activation function.

    Args:
        name: Name of activation function to use.
        **kwargs: Keyword arguments to be passed to the activation function.

    Returns:
        PyTorch activation function.
    """
    return activations[name](**kwargs)


def get_pooling(
    name: Literal["max", "avg"],
    dim: int,
    adaptive: bool = False,
) -> nn.Module:
    """Get pooling layer.

    Args:
        name: Name of pooling layer to use.
        dim: Dimension of pooling layer.
        adaptive: Whether to use adaptive pooling.

    Returns:
        PyTorch pooling layer.
    """
    if adaptive:
        name = "Adaptive" + name.capitalize()
    return poolings[f"{name.capitalize()}Pool{dim}d"]


class ConvLayer(nn.Module):
    """Convolution layer that consists of convolution, normalization, activation, and optional drop
    out layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...], list[int]],
        stride: Union[int, tuple[int, ...], list[int]],
        dim: int,
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

        Raises:
            NotImplementedError: If dimension of convolution is not 2 or 3.
            ValueError: If length of `kernel_size` or `stride` is not equal to `dim` if they are
                tuples or lists.
        """
        super().__init__()
        if dim not in [2, 3]:
            raise NotImplementedError(f"{dim}D convolution is not supported right now!")
        if conv_kwargs is None:
            conv_kwargs = {}
        if norm_kwargs is None:
            norm_kwargs = {}
        if activation_kwargs is None:
            activation_kwargs = {}
        if drop_kwargs is None:
            drop_kwargs = {}

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * dim

        if not len(kernel_size) == dim:
            raise ValueError(
                f"`kernel_size` {kernel_size} must be an integer or a tuple/list of length "
                f"{dim}!"
            )

        if isinstance(stride, int):
            stride = (stride,) * dim

        if not len(stride) == dim:
            raise ValueError(
                f"`stride` {stride} must be an integer or a tuple/list of length " f"{dim}!"
            )

        # Store stride and out_channels for computing total number of pixels/voxels in the output
        # feature map
        self.stride = stride
        self.out_channels = out_channels

        modules = []
        conv = get_conv(
            in_channels, out_channels, kernel_size, stride, dim, conv_bias, **conv_kwargs
        )
        modules.append(conv)

        if drop_block:
            drop_block = get_drop_block(dim, **drop_kwargs)
            modules.append(drop_block)

        if norm_layer is not None:
            norm = get_norm(norm_layer, out_channels, dim, **norm_kwargs)
            modules.append(norm)

        if activation is not None:
            activation = get_activation(activation, **activation_kwargs)
            modules.append(activation)

        self.all_modules = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:  # noqa: D102
        return self.all_modules(inputs)

    def compute_pixels_in_output_feature_map(self, input_size: tuple[int, ...]) -> int:
        """Compute total number of pixels/voxels in the output feature map after convolution.

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
        output_size = [
            i // j for i, j in zip(input_size, self.stride)
        ]  # we always do same padding
        return np.prod([self.out_channels, *output_size], dtype=np.int64)


class AttentionLayer(nn.Module):
    """Attention layer that consists of a convolution and normalization layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        conv_bias: bool = True,
        conv_kwargs: Optional[dict] = None,
        norm_layer: Optional[Literal["instance", "batch", "group", "layer"]] = None,
        norm_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dim: Dimension for convolution. (2 or 3)
            conv_bias: If True, adds a learnable bias to the convolution output.
            conv_kwargs: Keyword arguments for the convolution layer.
            norm_layer: Name of normalization layer.
            norm_kwargs: Keyword arguments for the normalization layer.

        Raises:
            NotImplementedError: If dimension of convolution is not 2 or 3.
        """
        super().__init__()
        if dim not in [2, 3]:
            raise NotImplementedError(f"{dim}D convolution is not supported right now!")
        if conv_kwargs is None:
            conv_kwargs = {}
        if norm_kwargs is None:
            norm_kwargs = {}

        # Store stride and out_channels for computing total number of pixels/voxels in the output
        # feature map
        self.stride = (1,) * dim
        self.out_channels = out_channels

        modules = []
        self.conv = get_conv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dim=dim,
            conv_bias=conv_bias,
            **conv_kwargs,
        )
        modules.append(self.conv)
        if norm_layer is not None:
            self.norm = get_norm(norm_layer, out_channels, dim, **norm_kwargs)
            modules.append(self.norm)

        self.all_modules = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:  # noqa: D102
        return self.all_modules(inputs)

    def compute_pixels_in_output_feature_map(self, input_size: tuple[int, ...]) -> int:
        """Compute total number of pixels/voxels in the output feature map after convolution.

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
        output_size = [
            i // j for i, j in zip(input_size, self.stride)
        ]  # we always do same padding
        return np.prod([self.out_channels, *output_size], dtype=np.int64)
