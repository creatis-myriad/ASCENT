from typing import Union

import torch
from torch import Tensor, nn
from torch.nn import Conv2d

from ascent.models.components.decoders.unet_decoder import UNetDecoder
from ascent.models.components.encoders.unet_encoder import UNetEncoder
from ascent.models.components.unet import UNet


class PDNet(nn.Module):
    """Modified primal-dual based reconstruction network to perform color Doppler dealiasing.

    PDNet uses the same feature maps per iteration.

    References:
        - Paper that introduced the Learned Primal-dual Reconstruction: https://arxiv.org/pdf/1707.06474.pdf
    """

    def __init__(
        self,
        iterations: int,
        n_primal: int,
        n_dual: int,
        in_channels: int,
        num_classes: int,
        patch_size: Union[list[int], tuple[int, ...]],
        negative_slope: float = 1e-2,
        out_conv: bool = True,
    ) -> None:
        """Initialize class instance.

        Args:
            iterations: Number of iterations.
            n_primal: Number of data that persists between primal iterates.
            n_dual: Number of data that persists between dual iterates.
            in_channels: Number of input channels.
            num_classes: Number of segmentation classes.
            patch_size: Input patch size.
            negative_slope: Negative slope used in LeakyRELU.
            out_conv: Whether to apply convolution to the output.

        Raises:
            NotImplementedError: Error when input patch size is neither 2D nor 3D.
        """
        super().__init__()
        if not len(patch_size) in [2, 3]:
            raise NotImplementedError("Only 2D and 3D patches are supported right now!")

        self.iterations = iterations
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.negative_slope = negative_slope
        self.out_conv = out_conv

        # to keep the compatibility with nnUNetLitModule
        self.deep_supervision = False

        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.bias = True

        self.conv_dual_1 = Conv2d(
            n_dual + self.in_channels + 1,
            32,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=self.bias,
        )

        self.conv_primal_1 = Conv2d(
            n_primal + 1, 32, self.kernel_size, self.stride, self.padding, bias=self.bias
        )

        self.conv_dual_2 = Conv2d(
            32, 32, self.kernel_size, self.stride, self.padding, bias=self.bias
        )

        self.conv_dual_3 = Conv2d(
            32, n_dual, self.kernel_size, self.stride, self.padding, bias=self.bias
        )

        self.conv_primal_2 = Conv2d(
            32, 32, self.kernel_size, self.stride, self.padding, bias=self.bias
        )

        self.conv_primal_3 = Conv2d(
            32, n_primal, self.kernel_size, self.stride, self.padding, bias=self.bias
        )

        self.lrelu = nn.LeakyReLU(self.negative_slope, inplace=True)

        if self.out_conv:
            self.output_conv = Conv2d(
                1,
                self.num_classes,
                self.kernel_size,
                self.stride,
                self.padding,
                bias=self.bias,
            )

        self.apply(self.initialize_weights)

    @staticmethod
    def wrap(x: Tensor, wrap_param: float = 1.0, normalize: bool = False) -> Tensor:
        """Wrap any element with its absolute value surpassing the wrapping parameter.

        Args:
            x: Input tensor.
            wrap_param: Wrapping parameter.
            normalize: Whether to normalize the wrapped tensor between -1 and 1.

        Returns:
            Wrapped tensor.
        """
        x = (x + wrap_param) % (2 * wrap_param) - wrap_param
        if normalize:
            return x / wrap_param
        else:
            return x

    def forward(self, input_data: Tensor) -> Tensor:  # noqa: D102
        primal = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_primal,
            dim=1,
        )
        dual = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_dual,
            dim=1,
        )

        for i in range(self.iterations):
            # dual iterates
            eval_pt = primal[:, 1:2, ...]

            eval_op = self.wrap(eval_pt)

            update = torch.concat([dual, eval_op, input_data], dim=1)

            update = self.conv_dual_1(update)
            update = self.lrelu(update)
            update = self.conv_dual_2(update)
            update = self.lrelu(update)
            update = self.conv_dual_3(update)

            # residual connection
            dual = dual + update

            # primal iterates
            eval_op = dual[:, 0:1, ...]
            update = torch.concat([primal, eval_op], dim=1)

            update = self.conv_primal_1(update)
            update = self.lrelu(update)
            update = self.conv_primal_2(update)
            update = self.lrelu(update)
            update = self.conv_primal_3(update)

            # residual connection
            primal = primal + update

        out = primal[:, 0:1, ...]
        if self.out_conv:
            out = self.output_conv(out)

        return out

    def debug(self, input_data: Tensor) -> Tensor:  # noqa: D102
        primal = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_primal,
            dim=1,
        )
        dual = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_dual,
            dim=1,
        )

        results = []

        for i in range(self.iterations):
            # dual iterates
            eval_pt = primal[:, 1:2, ...]

            eval_op = self.wrap(eval_pt)

            update = torch.concat([dual, eval_op, input_data], dim=1)
            update = self.conv_dual_1(update)
            update = self.lrelu(update)
            update = self.conv_dual_2(update)
            update = self.lrelu(update)
            update = self.conv_dual_3(update)

            # residual connection
            dual = dual + update

            # primal iterates
            eval_op = dual[:, 0:1, ...]
            update = torch.concat([primal, eval_op], dim=1)

            update = self.conv_primal_1(update)
            update = self.lrelu(update)
            update = self.conv_primal_2(update)
            update = self.lrelu(update)
            update = self.conv_primal_3(update)

            # residual connection
            primal = primal + update

            results.append([primal[:, 1:2, ...], primal[:, 0:1, ...], dual[:, 0:1, ...]])

        out = primal[:, 0:1, ...]
        if self.out_conv:
            out = self.output_conv(out)

        return out, results

    def initialize_weights(self, module: nn.Module) -> None:
        """Initialize the weights of all nn Modules using Kaimimg normal initialization."""
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.negative_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class OriPDNet(nn.Module):
    """Original primal-dual based reconstruction network to perform color Doppler dealiasing.

    OriPDNet uses different feature maps per iteration.

    References:
        - Paper that introduced the Learned Primal-dual Reconstruction: https://arxiv.org/pdf/1707.06474.pdf
    """

    def __init__(
        self,
        iterations: int,
        n_primal: int,
        n_dual: int,
        in_channels: int,
        num_classes: int,
        patch_size: Union[list[int], tuple[int, ...]],
        negative_slope: float = 1e-2,
        out_conv: bool = True,
    ) -> None:
        """Initialize class instance.

        Args:
            iterations: Number of iterations.
            n_primal: Number of data that persists between primal iterates.
            n_dual: Number of data that persists between dual iterates.
            in_channels: Number of input channels.
            num_classes: Number of segmentation classes.
            patch_size: Input patch size.
            negative_slope: Negative slope used in LeakyRELU.
            out_conv: Whether to apply convolution to the output.

        Raises:
            NotImplementedError: Error when input patch size is neither 2D nor 3D.
        """
        super().__init__()
        if not len(patch_size) in [2, 3]:
            raise NotImplementedError("Only 2D and 3D patches are supported right now!")

        self.iterations = iterations
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.negative_slope = negative_slope
        self.out_conv = out_conv

        # to keep the compatibility with nnUNetLitModule
        self.deep_supervision = False

        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.bias = True

        self.conv_dual_1 = nn.ModuleList(
            [
                Conv2d(
                    n_dual + self.in_channels + 1,
                    32,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    bias=self.bias,
                )
                for i in range(self.iterations)
            ]
        )

        self.conv_primal_1 = nn.ModuleList(
            [
                Conv2d(
                    n_primal + 1,
                    32,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    bias=self.bias,
                )
                for i in range(self.iterations)
            ]
        )

        self.conv_dual_2 = nn.ModuleList(
            [
                Conv2d(32, 32, self.kernel_size, self.stride, self.padding, bias=self.bias)
                for i in range(self.iterations)
            ]
        )
        self.conv_dual_3 = nn.ModuleList(
            [
                Conv2d(32, n_dual, self.kernel_size, self.stride, self.padding, bias=self.bias)
                for i in range(self.iterations)
            ]
        )

        self.conv_primal_2 = nn.ModuleList(
            [
                Conv2d(32, 32, self.kernel_size, self.stride, self.padding, bias=self.bias)
                for i in range(self.iterations)
            ]
        )
        self.conv_primal_3 = nn.ModuleList(
            [
                Conv2d(32, n_primal, self.kernel_size, self.stride, self.padding, bias=self.bias)
                for i in range(self.iterations)
            ]
        )

        self.lrelu = nn.ModuleList(
            [nn.LeakyReLU(self.negative_slope, inplace=True) for i in range(self.iterations)]
        )

        if self.out_conv:
            self.output_conv = Conv2d(
                1,
                self.num_classes,
                self.kernel_size,
                self.stride,
                self.padding,
                bias=self.bias,
            )

        self.apply(self.initialize_weights)

    @staticmethod
    def wrap(x: Tensor, wrap_param: float = 1.0, normalize: bool = False) -> Tensor:
        """Wrap any element with its absolute value surpassing the wrapping parameter.

        Args:
            x: Input tensor.
            wrap_param: Wrapping parameter.
            normalize: Whether to normalize the wrapped tensor between -1 and 1.

        Returns:
            Wrapped tensor.
        """
        x = (x + wrap_param) % (2 * wrap_param) - wrap_param
        if normalize:
            return x / wrap_param
        else:
            return x

    def forward(self, input_data: Tensor) -> Tensor:  # noqa: D102
        primal = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_primal,
            dim=1,
        )
        dual = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_dual,
            dim=1,
        )

        for i in range(self.iterations):
            # dual iterates
            eval_pt = primal[:, 1:2, ...]

            eval_op = self.wrap(eval_pt)

            update = torch.concat([dual, eval_op, input_data], dim=1)
            update = self.conv_dual_1[i](update)
            update = self.lrelu[i](update)
            update = self.conv_dual_2[i](update)
            update = self.lrelu[i](update)
            update = self.conv_dual_3[i](update)

            # residual connection
            dual = dual + update

            # primal iterates
            eval_op = dual[:, 0:1, ...]
            update = torch.concat([primal, eval_op], dim=1)

            update = self.conv_primal_1[i](update)
            update = self.lrelu[i](update)
            update = self.conv_primal_2[i](update)
            update = self.lrelu[i](update)
            update = self.conv_primal_3[i](update)

            # residual connection
            primal = primal + update

        out = primal[:, 0:1, ...]
        if self.out_conv:
            out = self.output_conv(out)

        return out

    def debug(self, input_data: Tensor) -> Tensor:  # noqa: D102
        primal = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_primal,
            dim=1,
        )
        dual = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_dual,
            dim=1,
        )

        results = []

        for i in range(self.iterations):
            # dual iterates
            eval_pt = primal[:, 1:2, ...]

            eval_op = self.wrap(eval_pt)

            update = torch.concat([dual, eval_op, input_data], dim=1)
            update = self.conv_dual_1[i](update)
            update = self.lrelu[i](update)
            update = self.conv_dual_2[i](update)
            update = self.lrelu[i](update)
            update = self.conv_dual_3[i](update)

            # residual connection
            dual = dual + update

            # primal iterates
            eval_op = dual[:, 0:1, ...]
            update = torch.concat([primal, eval_op], dim=1)

            update = self.conv_primal_1[i](update)
            update = self.lrelu[i](update)
            update = self.conv_primal_2[i](update)
            update = self.lrelu[i](update)
            update = self.conv_primal_3[i](update)

            # residual connection
            primal = primal + update

            results.append([primal[:, 1:2, ...], primal[:, 0:1, ...], dual[:, 0:1, ...]])

        out = primal[:, 0:1, ...]
        if self.out_conv:
            out = self.output_conv(out)

        return out, results

    def initialize_weights(self, module: nn.Module) -> None:
        """Initialize the weights of all nn Modules using Kaimimg normal initialization."""
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.negative_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class PDUNet(nn.Module):
    """Modified primal-dual based reconstruction network to perform color Doppler dealiasing.

    Instead of using just few convolution layers to learn the primal and dual's proximal operators,
    U-Nets are used.

    References:
        - Paper that introduced the Learned Primal-dual Reconstruction: https://arxiv.org/pdf/1707.06474.pdf
    """

    def __init__(
        self,
        iterations: int,
        n_primal: int,
        n_dual: int,
        in_channels: int,
        num_classes: int,
        patch_size: Union[list[int], tuple[int, ...]],
        kernels: Union[int, list[Union[int, list[int]]], tuple[Union[int, tuple[int, ...]], ...]],
        strides: Union[int, list[Union[int, list[int]]], tuple[Union[int, tuple[int, ...]], ...]],
        out_conv: bool = True,
    ) -> None:
        """Initialize class instance.

        Args:
            iterations: Number of iterations.
            n_primal: Number of data that persists between primal iterates.
            n_dual: Number of data that persists between dual iterates.
            in_channels: Number of input channels.
            num_classes: Number of segmentation classes.
            patch_size: Input patch size.
            kernels: Size of the convolving kernel. If the value is an integer, then the same
                value is used for all spatial dimensions and all stages. If the value is a sequence
                , then each value is used for each stage. If the value is a sequence of sequence,
                then each value is used for each stage and each spatial dimension.
            strides: Stride of the convolution. If the value is an integer, then the same value is
                used for all spatial dimensions and all stages. If the value is a sequence
                , then each value is used for each stage. If the value is a sequence of sequence,
                then each value is used for each stage and each spatial dimension.
            out_conv: Whether to apply convolution to the output.

        Raises:
            NotImplementedError: Error when input patch size is neither 2D nor 3D.
        """
        super().__init__()
        if not len(patch_size) in [2, 3]:
            raise NotImplementedError("Only 2D and 3D patches are supported right now!")

        self.iterations = iterations
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.negative_slope = 1e-2
        self.out_conv = out_conv

        # to keep the compatibility with nnUNetLitModule
        self.deep_supervision = False

        primal_channels = n_primal + 1
        primal_encoder_kwargs = {
            "in_channels": primal_channels,
            "num_stages": len(kernels),
            "dim": len(patch_size),
            "kernels": kernels,
            "strides": strides,
            "activation_kwargs": {"inplace": True},
        }
        primal_encoder = UNetEncoder(**primal_encoder_kwargs)
        primal_decoder = UNetDecoder(primal_encoder, num_classes=n_primal, deep_supervision=False)
        self.unet_primal = UNet(patch_size, primal_encoder, primal_decoder)

        dual_channels = n_dual + self.in_channels + 1
        dual_encoder_kwargs = {
            "in_channels": dual_channels,
            "num_stages": len(kernels),
            "dim": len(patch_size),
            "kernels": kernels,
            "strides": strides,
            "activation_kwargs": {"inplace": True},
        }
        dual_encoder = UNetEncoder(**dual_encoder_kwargs)
        dual_decoder = UNetDecoder(dual_encoder, num_classes=n_dual, deep_supervision=False)

        self.unet_dual = UNet(patch_size, dual_encoder, dual_decoder)

        if self.out_conv:
            self.output_conv = Conv2d(1, self.num_classes, 3, 1, 1, bias=False)

        self.apply(self.initialize_weights)

    @staticmethod
    def wrap(x: Tensor, wrap_param: float = 1.0, normalize: bool = False) -> Tensor:
        """Wrap any element with its absolute value surpassing the wrapping parameter.

        Args:
            x: Input tensor.
            wrap_param: Wrapping parameter.
            normalize: Whether to normalize the wrapped tensor between -1 and 1.

        Returns:
            Wrapped tensor.
        """
        x = (x + wrap_param) % (2 * wrap_param) - wrap_param
        if normalize:
            return x / wrap_param
        else:
            return x

    def forward(self, input_data: Tensor) -> Tensor:  # noqa: D102
        primal = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_primal,
            dim=1,
        )
        dual = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_dual,
            dim=1,
        )

        for i in range(self.iterations):
            # dual iterates
            eval_pt = primal[:, 1:2, ...]
            eval_op = self.wrap(eval_pt)
            update = torch.concat([dual, eval_op, input_data], dim=1)
            update = self.unet_dual(update)

            # residual connection
            dual = dual + update

            # primal iterates
            eval_op = dual[:, 0:1, ...]
            update = torch.concat([primal, eval_op], dim=1)

            update = self.unet_primal(update)

            # residual connection
            primal = primal + update

        out = primal[:, 0:1, ...]
        # convolution to get output having same number of channels as the segmentation class
        if self.out_conv:
            out = self.output_conv(out)

        return out

    def debug(self, input_data: Tensor) -> Tensor:  # noqa: D102
        primal = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_primal,
            dim=1,
        )
        dual = torch.concat(
            [torch.zeros(input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device)]
            * self.n_dual,
            dim=1,
        )

        results = []

        for i in range(self.iterations):
            # dual iterates
            eval_pt = primal[:, 1:2, ...]
            eval_op = self.wrap(eval_pt)
            update = torch.concat([dual, eval_op, input_data], dim=1)
            update = self.unet_dual(update)

            # residual connection
            dual = dual + update

            # primal iterates
            eval_op = dual[:, 0:1, ...]
            update = torch.concat([primal, eval_op], dim=1)

            update = self.unet_primal(update)

            # residual connection
            primal = primal + update

            results.append([primal[:, 1:2, ...], primal[:, 0:1, ...], dual[:, 0:1, ...]])

        out = primal[:, 0:1, ...]
        # convolution to get output having same number of channels as the segmentation class
        if self.out_conv:
            out = self.output_conv(out)

        return out, results

    def initialize_weights(self, module: nn.Module) -> None:
        """Initialize the weights of all nn Modules using Kaimimg normal initialization."""
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.negative_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    from torchinfo import summary

    iterations = 10
    n_primal = 5
    n_dual = 5
    in_channels = 1
    num_classes = 3
    patch_size = [40, 192]
    negative_slope = 1e-2

    pdnet = PDNet(
        iterations, n_primal, n_dual, in_channels, num_classes, patch_size, negative_slope
    )
    dummy_input = torch.rand((2, 1, *patch_size))
    print(summary(pdnet, input_size=dummy_input.shape))
    out = pdnet(dummy_input)
