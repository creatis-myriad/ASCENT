from typing import Literal, Optional, Type, Union

import numpy as np
import torch
from monai.data import MetaTensor
from torch import Tensor, nn

from ascent.models.components.encoders.convnext import ConvNeXt
from ascent.models.components.encoders.unet_encoder import UNetEncoder
from ascent.models.components.utils.blocks import OutputBlock, UpsampleBlock
from ascent.models.components.utils.initialization import get_initialization


class UNetDecoder(nn.Module):
    """A generic U-Net decoder that can be instantiated dynamically and paired with any U-Net-kind
    encoder."""

    def __init__(
        self,
        encoder: Literal[UNetEncoder, ConvNeXt],
        num_classes: int,
        num_conv_per_stage: Union[int, list[int], tuple[int, ...]] = 2,
        output_conv_bias: bool = True,
        deep_supervision: bool = True,
        attention: bool = False,
        initialization: Optional[
            Literal["kaiming_normal", "xavier_uniform", "trunc_normal"]
        ] = None,
    ) -> None:
        """Initialize class instance.

        Args:
            encoder: A U-Net encoder.
            num_classes: Number of classes to predict.
            num_conv_per_stage: Number of convolutional layers per decoder stage.
            output_conv_bias: If True, add bias to the output convolutional layer.
            deep_supervision: If True, add deep supervision heads.
            attention: If True, add attention module in each decoder stage.
            initialization: Weight initialization technique. Can be either `kaiming_normal` or
                `xavier_uniform` or None. If None, the initialization of the decoder follows the
                initialization of the encoder.
        """
        super().__init__()
        if 2 in encoder.strides[0]:
            raise NotImplementedError(
                f"`{encoder.__class__.__name__}` performs pooling in the stem, which is not "
                f"supported by `{self.__class__.__name__}`!"
            )

        self.deep_supervision = deep_supervision
        self.encoder_strides = encoder.strides
        self.num_classes = num_classes
        num_stages_encoder = len(encoder.filters_per_stage)

        if isinstance(num_conv_per_stage, int):
            num_conv_per_stage = (num_conv_per_stage,) * (num_stages_encoder - 1)

        if not len(num_conv_per_stage) == num_stages_encoder - 1:
            raise ValueError(
                f"len(n_conv_per_stage) must be equal to num_stages_encoder: {num_stages_encoder}"
            )

        # start with the bottleneck and work way up
        stages = []
        for stage in range(1, num_stages_encoder):
            input_features_below = encoder.filters_per_stage[-stage]
            input_features_skip = encoder.filters_per_stage[-(stage + 1)]
            stride_transpconv = encoder.strides[-stage]

            # upsampling + stacked convolutions block
            stages.append(
                UpsampleBlock(
                    num_conv_per_stage[stage - 1],
                    input_features_below,
                    input_features_skip,
                    encoder.kernels[-(stage + 1)],
                    stride_transpconv,
                    encoder.dim,
                    attention,
                    encoder.conv_bias,
                    None,
                    encoder.norm_layer,
                    encoder.norm_kwargs,
                    encoder.activation,
                    encoder.activation_kwargs,
                    encoder.drop_block,
                    encoder.drop_kwargs,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.output_block = OutputBlock(
            stages[-1].out_channels,
            num_classes,
            encoder.dim,
            output_conv_bias,
        )
        self.deep_supervision_heads = nn.ModuleList(
            [
                OutputBlock(
                    stages[i].out_channels,
                    num_classes,
                    encoder.dim,
                    output_conv_bias,
                )
                for i in reversed(range(2, len(stages) - 1))
            ]
        )

        # initialize weights
        if initialization is None:
            self.apply(get_initialization(encoder.initialization, **encoder.init_kwargs))
        else:
            self.apply(get_initialization(initialization))

    def forward(
        self, skips: list[Union[Tensor, MetaTensor]]
    ) -> Union[Tensor, MetaTensor, list[Union[Tensor, MetaTensor]]]:  # noqa: D102
        out = skips[-1]
        decoder_outputs = []
        for stage, skip in zip(self.stages, skips[::-1][1:]):
            out = stage(out, skip)
            decoder_outputs.append(out)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            out = [out]
            for i, decoder_out in enumerate(decoder_outputs[2:-1][::-1]):
                out.append(self.deep_supervision_heads[i](decoder_out))
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
        if not len(input_size) == len(self.encoder_strides[0]):
            raise ValueError(
                "`input_size` should be (H, W(, D)) without channel or batch dimension!"
            )
        # compute the skip connection sizes for each stage
        skip_sizes = []
        for s in range(len(self.encoder_strides)):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder_strides[s])])
            input_size = skip_sizes[-1]

        output = np.int64(0)
        for s in range(len(self.stages)):
            # transposed conv + stacked conv blocks
            output += self.stages[s].compute_pixels_in_output_feature_map(skip_sizes[-(s + 1)])
        # output block
        output += self.output_block.compute_pixels_in_output_feature_map(tuple(skip_sizes[0]))
        # deep supervision heads
        if self.deep_supervision:
            for s in range(len(self.deep_supervision_heads)):
                output += self.deep_supervision_heads[s].compute_pixels_in_output_feature_map(
                    tuple(skip_sizes[s + 1])
                )
        return output


if __name__ == "__main__":
    from torchinfo import summary

    from ascent.models.components.encoders.unet_encoder import UNetEncoder

    patch_size = (640, 1024)
    in_channels = 1
    kernels = [[3, 1], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    strides = [[1, 1], [1, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

    encoder = UNetEncoder(
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

    unet_decoder = UNetDecoder(
        encoder=encoder,
        num_classes=4,
        num_conv_per_stage=2,
        deep_supervision=True,
        attention=False,
    )

    print(unet_decoder)
    print(
        encoder.compute_pixels_in_output_feature_map(patch_size)
        + unet_decoder.compute_pixels_in_output_feature_map(patch_size)
    )
    dummy_input = torch.rand((2, in_channels, *patch_size))
    encoder_out = encoder(dummy_input)
    out = unet_decoder(encoder_out)
    print(
        summary(
            unet_decoder,
            input_data={"skips": encoder_out},
            device="cpu",
            depth=10,
            col_names=("input_size", "output_size", "num_params"),
        )
    )
