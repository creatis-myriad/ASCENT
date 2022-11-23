from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import Conv2d

from ascent.models.components.unet import UNet


class PDNet(nn.Module):
    def __init__(
        self,
        iterations: int,
        n_primal: int,
        n_dual: int,
        in_channels: int,
        num_classes: int,
        patch_size: list[
            int,
        ],
        negative_slope: float = 1e-2,
        variant: int = 0,
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
            variant: PDNet variants.
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
        self.variant = variant

        self.deep_supervision = False
        self.dim = len(patch_size)
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

        if self.variant <= 3:
            if self.variant in [2, 3]:
                self.conv_primal_1 = Conv2d(
                    n_primal + 2, 32, self.kernel_size, self.stride, self.padding, bias=self.bias
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

        self.output_conv = Conv2d(
            1, self.num_classes, self.kernel_size, self.stride, self.padding, bias=self.bias
        )

        if self.variant == 3:
            self.conv1 = Conv2d(
                n_primal, 1, self.kernel_size, self.stride, self.padding, bias=self.bias
            )
            self.conv2 = Conv2d(
                n_primal, 1, self.kernel_size, self.stride, self.padding, bias=self.bias
            )
            self.conv3 = Conv2d(
                n_dual, 1, self.kernel_size, self.stride, self.padding, bias=self.bias
            )
            self.output_conv = Conv2d(
                n_primal,
                self.num_classes,
                self.kernel_size,
                self.stride,
                self.padding,
                bias=self.bias,
            )

        self.lrelu = nn.LeakyReLU(self.negative_slope, inplace=True)

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
        for i in range(self.iterations):
            primal = torch.concat(
                [
                    torch.zeros(
                        input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device
                    )
                ]
                * self.n_primal,
                dim=1,
            )
            dual = torch.concat(
                [
                    torch.zeros(
                        input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device
                    )
                ]
                * self.n_dual,
                dim=1,
            )

            # dual iterates
            if self.variant <= 2:
                eval_pt = primal[:, 1:2, ...]
            if self.variant == 3:
                eval_pt = self.conv1(primal)
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

            if self.variant <= 1:
                eval_op = dual[:, 0:1, ...]
                update = torch.concat([primal, eval_op], dim=1)
            elif self.variant == 2:
                eval_op = dual[:, 0:1, ...]
                eval_pt = primal[:, 0:1, ...]
                update = torch.concat([primal, eval_pt, eval_op], dim=1)
            elif self.variant == 3:
                eval_pt = self.conv2(primal)
                eval_op = self.conv3(dual)
                update = torch.concat([primal, eval_pt, eval_op], dim=1)

            update = self.conv_primal_1(update)
            update = self.lrelu(update)
            update = self.conv_primal_2(update)
            update = self.lrelu(update)
            update = self.conv_primal_3(update)

            # residual connection
            primal = primal + update

            if self.variant == 1:
                # convolution to get output having same number of channels as the segmentation class
                out = self.output_conv(primal[:, 0:1, ...])

            if self.variant == 3:
                # convolution to get output having same number of channels as the segmentation class
                out = self.output_conv(primal)

        if self.variant == 0 or self.variant == 2:
            # convolution to get output having same number of channels as the segmentation class
            out = self.output_conv(primal[:, 0:1, ...])

        return out


class PDNetV2(nn.Module):
    def __init__(
        self,
        iterations: int,
        n_primal: int,
        n_dual: int,
        in_channels: int,
        num_classes: int,
        patch_size: list[
            int,
        ],
        kernels: list[Sequence[tuple[int]]],
        strides: list[Sequence[tuple[int]]],
    ) -> None:
        """Initialize class instance.

        Args:
            iterations: Number of iterations.
            n_primal: Number of data that persists between primal iterates.
            n_dual: Number of data that persists between dual iterates.
            in_channels: Number of input channels.
            num_classes: Number of segmentation classes.
            patch_size: Input patch size.
            kernels: List of list containing convolution kernel size of the first convolution layer
                of each double convolutions block.
            strides: List of list containing convolution strides of the first convolution layer
                of each double convolutions block.
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

        # compatibility with nnunet module
        self.deep_supervision = False

        primal_channels = n_primal + 1
        self.unet_primal = UNet(
            primal_channels, n_primal, patch_size, kernels, strides, deep_supervision=False
        )

        dual_channels = n_dual + self.in_channels + 1
        self.unet_dual = UNet(
            dual_channels, n_dual, patch_size, kernels, strides, deep_supervision=False
        )

        self.output_conv = Conv2d(1, self.num_classes, 3, 1, 1, bias=False)

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
        for i in range(self.iterations):
            primal = torch.concat(
                [
                    torch.zeros(
                        input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device
                    )
                ]
                * self.n_primal,
                dim=1,
            )
            dual = torch.concat(
                [
                    torch.zeros(
                        input_data.shape[0], 1, *input_data.shape[2:], device=input_data.device
                    )
                ]
                * self.n_dual,
                dim=1,
            )

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

        # convolution to get output having same number of channels as the segmentation class
        out = self.output_conv(primal[:, 0:1, ...])

        return out
