from typing import Sequence, Union

import numpy as np
import torch
from monai.data import MetaTensor
from torch import Tensor, nn

from ascent.models.components.unet_related.layers import (
    ConvBlock,
    OutputBlock,
    ResidBlock,
    UpsampleBlock,
)


class UNet(nn.Module):
    """A generic U-Net that can be instantiated dynamically."""

    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 32
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        patch_size: list,
        kernels: list[Sequence[tuple[int, ...]]],
        strides: list[Sequence[tuple[int, ...]]],
        normalization_layer: str = "instance",
        negative_slope: float = 1e-2,
        deep_supervision: bool = True,
        attention: bool = False,
        drop_block: bool = False,
        residual: bool = False,
        out_seg_bias: bool = False,
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of input channels.
            num_classes: Total number of classes in the dataset including background.
            patch_size: Input patch size according to which the data will be cropped. Can be 2D or
                3D.
            kernels: List of list containing convolution kernel size of the first convolution layer
                of each double convolutions block.
            strides: List of list containing convolution strides of the first convolution layer
                of each double convolutions block.
            normalization_layer: Normalization to use. Can be either 'batch' or 'instance'.
            negative_slope: Negative slope used in LeakyRELU.
            deep_supervision: Whether to use deep supervision.
            attention: Whether to use attention module.
            drop_block: Whether to use drop out layers.
            residual: Whether to use residual block.
            out_seg_bias: Whether to include trainable bias in the output segmentation layers.

        Raises:
            NotImplementedError: Error when input patch size is neither 2D nor 3D.
        """
        super().__init__()
        if not len(patch_size) in [2, 3]:
            raise NotImplementedError("Only 2D and 3D patches are supported right now!")

        self.patch_size = patch_size
        self.dim = len(patch_size)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.attention = attention
        self.residual = residual
        self.out_seg_bias = out_seg_bias
        self.negative_slope = negative_slope
        self.deep_supervision = deep_supervision
        self.norm = normalization_layer + f"norm{self.dim}d"
        self.filters = [
            min(2 ** (5 + i), 320 if self.dim == 3 else 480) for i in range(len(strides))
        ]

        down_block = ResidBlock if self.residual else ConvBlock
        self.input_block = self.get_conv_block(
            conv_block=down_block,
            in_channels=in_channels,
            out_channels=self.filters[0],
            kernel_size=kernels[0],
            stride=strides[0],
        )
        self.downsamples = self.get_module_list(
            conv_block=down_block,
            in_channels=self.filters[:-1],
            out_channels=self.filters[1:],
            kernels=kernels[1:-1],
            strides=strides[1:-1],
            drop_block=drop_block,
        )
        self.bottleneck = self.get_conv_block(
            conv_block=down_block,
            in_channels=self.filters[-2],
            out_channels=self.filters[-1],
            kernel_size=kernels[-1],
            stride=strides[-1],
            drop_block=drop_block,
        )
        self.upsamples = self.get_module_list(
            conv_block=UpsampleBlock,
            in_channels=self.filters[1:][::-1],
            out_channels=self.filters[:-1][::-1],
            kernels=kernels[1:][::-1],
            strides=strides[1:][::-1],
        )
        self.output_block = self.get_output_block(decoder_level=0)
        self.deep_supervision_heads = self.get_deep_supervision_heads()
        self.apply(self.initialize_weights)

    def forward(
        self, input_data: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor]:  # noqa: D102
        out = self.input_block(input_data)
        encoder_outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out)
            encoder_outputs.append(out)
        out = self.bottleneck(out)
        decoder_outputs = []
        for upsample, skip in zip(self.upsamples, reversed(encoder_outputs)):
            out = upsample(out, skip)
            decoder_outputs.append(out)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            out = [out]
            for i, decoder_out in enumerate(decoder_outputs[2:-1][::-1]):
                out.append(self.deep_supervision_heads[i](decoder_out))
        return out

    def get_conv_block(
        self,
        conv_block: nn.Module,
        in_channels: int,
        out_channels: int,
        kernel_size: list,
        stride: list,
        drop_block: bool = False,
    ) -> nn.Module:
        """Build a double convolutions block.

        Args:
            conv_block: Convolution block. Can be usual double convolutions block or with residual
                connections or with attention modules.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: List containing convolution kernel size of the first convolution layer
                of double convolutions block.
            strides: List containing convolution strides of the first convolution layer
                of double convolutions block.
            drop_block: Whether to use drop out layers.

        Returns:
            Double convolutions block.
        """
        return conv_block(
            dim=self.dim,
            stride=stride,
            norm=self.norm,
            drop_block=drop_block,
            kernel_size=kernel_size,
            in_channels=in_channels,
            attention=self.attention,
            out_channels=out_channels,
            negative_slope=self.negative_slope,
        )

    def get_output_block(self, decoder_level: int) -> nn.Module:
        """Build the output convolution layer of the specified decoder layer.

        Args:
            decoder_level: Level of decoder.

        Returns:
            Output convolution layer.
        """
        return OutputBlock(
            in_channels=self.filters[decoder_level],
            out_channels=self.num_classes,
            dim=self.dim,
            bias=self.out_seg_bias,
        )

    def get_deep_supervision_heads(self) -> nn.ModuleList:
        """Build the deep supervision heads of all decoder levels except the two lowest
        resolutions.

        Returns:
            ModuleList of all deep supervision heads.
        """
        return nn.ModuleList(
            [self.get_output_block(i + 1) for i in range(len(self.upsamples) - 1)]
        )

    def get_module_list(
        self,
        in_channels: int,
        out_channels: int,
        kernels: list[Sequence[tuple[int, ...]]],
        strides: list[Sequence[tuple[int, ...]]],
        conv_block: nn.Module,
        drop_block: bool = False,
    ) -> nn.ModuleList:
        """Combine multiple convolution blocks to form a ModuleList.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernels: List of tuples containing convolution kernel size of the first convolution layer
                of each double convolutions block.
            strides: List of tuples containing convolution strides of the first convolution layer
                of each double convolutions block.
            conv_block: Convolution block to use.
            drop_block: Whether to use drop out layers.

        Returns:
            ModuleList of chained convolution blocks.
        """
        layers = []
        for i, (in_channel, out_channel, kernel, stride) in enumerate(
            zip(in_channels, out_channels, kernels, strides)
        ):
            use_drop_block = drop_block and len(in_channels) - i <= 2
            conv_layer = self.get_conv_block(
                conv_block, in_channel, out_channel, kernel, stride, use_drop_block
            )
            layers.append(conv_layer)
        return nn.ModuleList(layers)

    def initialize_weights(self, module: nn.Module) -> None:
        """Initialize the weights of all nn Modules using Kaimimg normal initialization."""
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.negative_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    @staticmethod
    def compute_approx_vram_consumption(
        patch_size: tuple[int, ...],
        num_pool_per_axis: list[int, ...],
        base_num_features: int,
        max_num_features: int,
        num_modalities: int,
        num_classes: int,
        pool_op_kernel_sizes: list[Sequence[tuple[int, ...]]],
        deep_supervision: bool = False,
        conv_per_stage: int = 2,
    ) -> int:
        """Calculate the approximate GPU memory required for the U-Net model.

        This only applies for num_conv_per_stage=2 and convolutional_upsampling=True not real vram
        consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage).

        Args:
            patch_size: Input patch size.
            num_pool_per_axis: List containing the number of poolings for each axe.
            base_num_features: Starting features of U-Net.
            max_num_features: Maximum features allowed in U-Net.
            num_modalities: Number of modalities/ Number of input channels.
            num_classes: Total number of classes.
            pool_op_kernel_sizes: List of list containing convolution strides of the first convolution
                layer of each double convolutions block.
            deep_supervision: Whether to use deep supervision.
            conv_per_stage: Number of convolution layers per level.

        Returns:
            Approximate total number of features in pixels of U-Net.

        Ref:
            https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64(
            (conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features
            + num_modalities * np.prod(map_size, dtype=np.int64)
            + num_classes * np.prod(map_size, dtype=np.int64)
        )

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (
                (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage
            )  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp


if __name__ == "__main__":
    kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
    unet = UNet(1, 3, [640, 512], kernels, strides)
    print(unet)
    dummy_input = torch.rand((2, 1, 640, 512))
    out = unet(dummy_input)
