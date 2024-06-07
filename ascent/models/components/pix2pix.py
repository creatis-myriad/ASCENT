from typing import Union

import torch
from monai.data import MetaTensor
from torch import Tensor, nn

from ascent.models.components.encoders.patch_gan import PatchGAN
from ascent.models.components.unet import UNet


class Pix2PixGAN(nn.Module):
    """A generic Pix2Pix that can be instantiated dynamically with different types of generators
    paired with PatchGAN discriminators."""

    def __init__(
        self,
        generator: UNet,
        discriminator: PatchGAN,
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
        patch_size = generator.patch_size
        if not len(patch_size) in [2, 3]:
            raise NotImplementedError("Only 2D and 3D patches are supported right now!")

        self.patch_size = patch_size
        self.generator = generator
        self.discriminator = discriminator

        # store some attributes to be used in nnUNetLitModule
        self.in_channels = self.generator.encoder.in_channels
        self.num_classes = self.generator.decoder.num_classes
        self.deep_supervision = self.generator.decoder.deep_supervision
        self.discriminator_output_shape = self.discriminator.get_output_shape(patch_size)

    def forward(
        self, input_data: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor]:  # noqa: D102
        return self.generator(input_data)


if __name__ == "__main__":
    from torchinfo import summary

    from ascent.models.components.decoders.unet_decoder import UNetDecoder
    from ascent.models.components.encoders.unet_encoder import UNetEncoder

    patch_size = (256, 256)
    in_channels = 1
    kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

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
    print(
        summary(
            unet,
            input_size=(2, in_channels, *patch_size),
            device="cpu",
            depth=10,
            col_names=("input_size", "output_size", "num_params"),
        )
    )

    pix2pix_gan = Pix2PixGAN(generator=unet, discriminator=PatchGAN(2, 2))
    print(pix2pix_gan)
