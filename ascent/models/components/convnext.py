from functools import partial
from typing import Sequence, Union

import torch
from monai.data import MetaTensor
from torch import Tensor, nn

from ascent.models.components.unet_related.layers import (
    ConvNeXtBlock,
    OutputBlock,
    UpsampleBlock,
    get_conv,
)
from ascent.models.components.unet_related.normalization import LayerNorm


class ConvNeXt(nn.Module):
    """ConvNeXt backbone paired with U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        patch_size: list,
        convnext_kernels: int,
        decoder_kernels: list[Sequence[tuple[int, ...]]],
        strides: list[Sequence[tuple[int, ...]]],
        depths: Sequence[int] = [2, 3, 3, 9, 3],
        filters: Sequence[int] = [32, 96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        encoder_normalization_layer: str = "layer",
        decoder_normalization_layer: str = "instance",
        negative_slope: float = 1e-2,
        deep_supervision: bool = True,
        out_seg_bias: bool = False,
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of input channels.
            num_classes: Total number of classes in the dataset including background.
            patch_size: Input patch size according to which the data will be cropped. Can be 2D or
                3D.
            convnext_kernels: Kernel size used in the convolutional layers in the ConvNeXt backbone.
            decoder_kernels: List of list containing convolution kernel size of the first convolution
                layer of each double convolutions block in the U-Net decoder.
            strides: List of list containing convolution strides of the downsampling layers.
            depths: Number of ConvNeXt blocks at each encoder stage.
            filters: Feature dimensions at each encoder stage.
            drop_path_rate: Stochastic depth rate.
            layer_scale_init_value: Init value for Layer Scale.
            encoder_normalization_layer: Normalization to use in the encoder.
            decoder_normalization_layer: Normalization to use in the decoder.
            negative_slope: Negative slope used in LeakyRELU.
            deep_supervision: Whether to use deep supervision.
            out_seg_bias: Whether to include trainable bias in the output segmentation layers.

        Raises:
            NotImplementedError: Error when input patch size is neither 2D nor 3D.
        """
        super().__init__()
        if not len(patch_size) in [2, 3]:
            raise NotImplementedError("Only 2D and 3D patches are supported right now!")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.dim = len(patch_size)
        self.num_classes = num_classes
        self.depths = depths
        self.filters = filters
        self.decoder_kernels = decoder_kernels
        self.negative_slope = negative_slope
        self.attention = False
        self.encoder_norm = encoder_normalization_layer + f"norm{self.dim}d"
        self.decoder_norm = decoder_normalization_layer + f"norm{self.dim}d"
        self.out_seg_bias = out_seg_bias
        self.deep_supervision = deep_supervision

        # Drop out rates
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Input blocks = Conv + ConvNeXtBlock + LayerNorm
        input_blocks = []
        input_blocks.append(get_conv(in_channels, self.filters[0], convnext_kernels, 1, self.dim))
        cur = 0
        for j in range(self.depths[0]):
            input_blocks.append(
                ConvNeXtBlock(
                    in_channels=self.filters[0],
                    kernel_size=convnext_kernels,
                    stride=1,
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    dim=self.dim,
                    norm=self.encoder_norm,
                )
            )
        input_blocks.append(LayerNorm(filters[0], eps=1e-6, data_format="channels_first"))
        cur += depths[0]
        self.input_block = nn.Sequential(*input_blocks)

        # Downsampling blocks
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            get_conv(in_channels, filters[1], strides[0], strides[0], dim=self.dim),
            LayerNorm(filters[1], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(1, len(filters) - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(filters[i], eps=1e-6, data_format="channels_first"),
                get_conv(filters[i], filters[i + 1], strides[i], strides[i], self.dim),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()

        for i in range(1, len(filters)):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        in_channels=filters[i],
                        kernel_size=convnext_kernels,
                        stride=1,
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        dim=self.dim,
                        norm=self.encoder_norm,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i - 1]

        # Normalization layers
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        self.norm_layers = nn.ModuleList()
        for i_layer in range(1, len(filters)):
            layer = norm_layer(filters[i_layer])
            self.norm_layers.append(layer)

        # ConvNeXt uses trunc_normal initialization instead of kaiming_normal
        self.apply(self.initialize_ConvNeXt_weights)

        # Upsampling blocks
        self.upsamples = self.get_module_list(
            conv_block=UpsampleBlock,
            in_channels=self.filters[1:][::-1],
            out_channels=self.filters[:-1][::-1],
            kernels=decoder_kernels[::-1],
            strides=strides[::-1],
        )

        # Output block
        self.output_block = self.get_output_block(decoder_level=0)

        # Deep supervision heads
        self.deep_supervision_heads = self.get_deep_supervision_heads()

        # Stick to kaiming_normal initialization for the decoder
        self.upsamples.apply(self.initialize_decoder_weights)
        self.output_block.apply(self.initialize_decoder_weights)
        self.deep_supervision_heads.apply(self.initialize_decoder_weights)

    def forward(
        self, input_data: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor]:  # noqa: D102
        encoder_outputs = [self.input_block(input_data)]
        out = input_data
        for i, (downsample, stage, norm) in enumerate(
            zip(self.downsample_layers, self.stages, self.norm_layers)
        ):
            out = downsample(out)
            out = stage(out)
            out = norm(out)
            if i < len(self.downsample_layers) - 1:
                encoder_outputs.append(out)

        decoder_outputs = []
        for upsample, skip in zip(self.upsamples, reversed(encoder_outputs)):
            out = upsample(out, skip)
            decoder_outputs.append(out)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            out = [out]
            for i, decoder_out in enumerate(decoder_outputs[:-1][::-1]):
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
            norm=self.decoder_norm,
            drop_block=drop_block,
            kernel_size=kernel_size,
            in_channels=in_channels,
            attention=False,
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
        """Build the deep supervision heads of all decoder levels.

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

    def initialize_ConvNeXt_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            module.weight = nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    def initialize_decoder_weights(self, module: nn.Module) -> None:
        """Initialize the weights of all nn Modules using Kaimimg normal initialization."""
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.negative_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    convnext_kernels = 7
    decoder_kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[4, 4, 1], [2, 1, 1], [2, 2, 2], [2, 2, 2]]
    convnext = ConvNeXt(1, 5, [96, 80, 20], convnext_kernels, decoder_kernels, strides)
    print(convnext)
    dummy_input = torch.rand((2, 1, 96, 80, 20))
    out = convnext(dummy_input)
