from typing import Literal, Union

import torch
from monai.data import MetaTensor
from torch import Tensor, nn

from ascent.models.components.decoders.unet_decoder import UNetDecoder
from ascent.models.components.encoders.convnext import ConvNeXt
from ascent.models.components.encoders.unet_encoder import UNetEncoder


class UNet(nn.Module):
    """A generic U-Net that can be instantiated dynamically with different types of encoders and
    decoders."""

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
        patch_size: Union[list[int], tuple[int, ...]],
        encoder: Literal[UNetEncoder, ConvNeXt],
        decoder: Literal[UNetDecoder],
    ) -> None:
        """Initialize class instance.

        Args:
            patch_size: Input patch size. Can be either 2D or 3D.
            encoder: A U-Net type convolutional encoder. Can be either `UNetEncoder` or `ConvNeXt`.
            decoder: A U-Net type convolutional decoder. Can be `UNetDecoder`.

        Raises:
            NotImplementedError: Error when input patch size is neither 2D nor 3D.
        """
        super().__init__()
        if not len(patch_size) in [2, 3]:
            raise NotImplementedError("Only 2D and 3D patches are supported right now!")

        self.patch_size = patch_size
        self.encoder = encoder
        self.decoder = decoder(encoder=encoder)

    def forward(
        self, input_data: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor]:  # noqa: D102
        skips = self.encoder(input_data)
        return self.decoder(skips)

    def compute_pixels_in_output_feature_map(self, input_size: tuple[int, ...]) -> int:
        """Compute total number of pixels/voxels in the output feature map after convolutions.

        Args:
            input_size: Size of the input image.

        Returns:
            Number of pixels/voxels in the output feature map after convolution.
        """
        if not len(input_size) == len(self.encoder.strides[0]):
            raise ValueError(
                "`input_size` should be (H, W(, D)) without channel or batch dimension!"
            )
        return self.encoder.compute_pixels_in_output_feature_map(
            input_size
        ) + self.decoder.compute_pixels_in_output_feature_map(input_size)


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
        dim=len(patch_size),
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

    unet = UNet(patch_size, encoder, unet_decoder)

    print(unet)
    dummy_input = torch.rand((2, in_channels, *patch_size))
    out = unet(dummy_input)
    print(unet.compute_pixels_in_output_feature_map(patch_size))
    print(
        summary(
            unet,
            input_size=(2, in_channels, *patch_size),
            device="cpu",
            depth=10,
            col_names=("input_size", "output_size", "num_params"),
        )
    )
