import torch.nn as nn

from covid.models.components.unet_related.layers import (
    ConvBlock,
    OutputBlock,
    ResidBlock,
    UpsampleBlock,
)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        patch_size,
        kernels,
        strides,
        normalization_layer="instance",
        negative_slope=1e-2,
        deep_supervision=True,
        attention=False,
        drop_block=False,
        residual=False,
    ):
        super().__init__()
        assert len(patch_size) in [2, 3], "Only 2D and 3D patches are supported right now!"

        self.patch_size = patch_size
        self.dim = str(len(patch_size))
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.attention = attention
        self.residual = residual
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

    def forward(self, input_data):
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
        self, conv_block, in_channels, out_channels, kernel_size, stride, drop_block=False
    ):
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

    def get_output_block(self, decoder_level):
        return OutputBlock(
            in_channels=self.filters[decoder_level],
            out_channels=self.num_classes,
            dim=self.dim,
            bias=False,
        )

    def get_deep_supervision_heads(self):
        return nn.ModuleList(
            [self.get_output_block(i + 1) for i in range(len(self.upsamples) - 1)]
        )

    def get_module_list(
        self, in_channels, out_channels, kernels, strides, conv_block, drop_block=False
    ):
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

    # @staticmethod
    def initialize_weights(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.negative_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    # def initialize_weights(self, module):
    #     name = module.__class__.__name__.lower()
    #     if name in ["conv2d", "conv3d"]:
    #         nn.init.kaiming_normal_(module.weight, a=self.negative_slope)


if __name__ == "__main__":
    import torch

    kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
    unet = UNet(1, 3, [640, 512], kernels, strides)
    print(unet)
    dummy_input = torch.rand((2, 1, 640, 512))
    out = unet(dummy_input)
