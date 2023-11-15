import warnings
from typing import Literal, Optional, Union

import numpy as np
import torch
from monai.data import MetaTensor
from torch import Tensor, nn

from ascent.models.components.utils.blocks import ConvNeXtBlock
from ascent.models.components.utils.initialization import get_initialization
from ascent.models.components.utils.layers import get_conv


class ConvNeXt(nn.Module):
    """A generic ConvNeXt backbone that can be instantiated dynamically.

    References:
        Liu et al. "A ConvNet for the 2020s". IEEE/CVF CVPR 2022.
    """

    # define some static class attributes
    MAX_NUM_FILTERS_2D = 480
    MAX_NUM_FILTERS_3D = 320

    def __init__(
        self,
        in_channels: int,
        num_stages: int,
        dim: Literal[2, 3],
        stem_kernel: Union[int, list[int], tuple[int, ...]],
        kernels: Union[int, list[Union[int, list[int]]], tuple[Union[int, tuple[int, ...]], ...]],
        strides: Union[int, list[Union[int, list[int]]], tuple[Union[int, tuple[int, ...]], ...]],
        num_conv_per_stage: Union[int, list[int], tuple[int, ...]] = (3, 3, 9, 3),
        num_features_per_stage: Union[int, list[int], tuple[int, ...]] = (96, 192, 384, 768),
        conv_bias: bool = True,
        expansion_rate: Union[int, list[int], tuple[int, ...]] = 4,
        stochastic_depth_p: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        conv_kwargs: Optional[dict] = None,
        norm_layer: Optional[Literal["instance", "batch", "group", "layer"]] = "group",
        norm_kwargs: Optional[dict] = None,
        activation: Optional[Literal["relu", "leakyrelu", "gelu", "tanh", "sigmoid"]] = "gelu",
        activation_kwargs: Optional[dict] = None,
        drop_block: bool = False,
        drop_kwargs: Optional[dict] = None,
        return_skip: bool = True,
        initialization: Literal[
            "kaiming_normal", "xavier_uniform", "trunc_normal"
        ] = "trunc_normal",
    ) -> None:
        """Initialize class instance.

        Args:
            in_channels: Number of input channels.
            num_stages: Number of stages in the encoder.
            dim: Spatial dimension of the input data.
            stem_kernel: Kernel size of the stem convolutional layer.
            kernels: Size of the convolving kernel. If the value is an integer, then the same
                value is used for all spatial dimensions and all stages. If the value is a sequence
                of length `num_stages`, then each value is used for each stage. If the value is a
                sequence of length `num_stages` and each element is a sequence of length `dim`,
                then each value is used for each stage and each spatial dimension.
            strides: Stride of the convolution. If the value is an integer, then the same value is
                used for all spatial dimensions and all stages. If the value is a sequence of
                length `num_stages`, then each value is used for each stage. If the value is a
                sequence of length `num_stages` and each element is a sequence of length `dim`,
                then each value is used for each stage and each spatial dimension.
            num_conv_per_stage: Number of convolutional layers per stage.
            conv_bias: Whether to use bias in the convolutional layers.
            expansion_rate: Expansion rate of the convolutional layers.
            stochastic_depth_p: Stochastic depth rate.
            layer_scale_init_value: Init value for Layer Scale.
            conv_kwargs: Keyword arguments for the convolution layers.
            norm_layer: Normalization to use. Can be either `instance`, `batch`,
                `group` or `layer`.
            norm_kwargs: Keyword arguments for the normalization layers.
            activation: Activation function to use. Can be either `relu`, `leakyrelu`, `gelu`,
                `tanh` or `sigmoid`.
            activation_kwargs: Keyword arguments for the activation functions.
            drop_block: Whether to use drop out layers.
            drop_kwargs: Keyword arguments for the drop out layers: `nn.Dropout2d` or
                `nn.Dropout3d`.
            return_skip: Whether to return the skip connections.
            initialization: Weight initialization technique. Can be either `kaiming_normal` or
                `xavier_uniform`.

        Raises:
            ValueError: When `len(kernels)` is not equal to `num_stages`.
            ValueError: When `len(strides)` is not equal to `num_stages`.
            ValueError: When `len(expansion_rate)` is not equal to `num_stages`.
            ValueError: When `len(n_conv_per_stage)` is not equal to `num_stages`.
            ValueError: When `len(num_features_per_stage)` is not equal to `num_stages`.
        """
        super().__init__()

        if isinstance(stem_kernel, int):
            stem_kernel = (stem_kernel,) * dim

        if isinstance(kernels, int):
            kernels = (kernels,) * num_stages

        if isinstance(strides, int):
            strides = (strides,) * num_stages

        if isinstance(expansion_rate, int):
            expansion_rate = (expansion_rate,) * num_stages

        if isinstance(num_conv_per_stage, int):
            num_conv_per_stage = (num_conv_per_stage,) * num_stages

        if isinstance(num_features_per_stage, int):
            num_features_per_stage = (num_features_per_stage,) * num_stages

        if not len(kernels) == num_stages:
            raise ValueError(f"len(kernels) must be equal to num_stages: {num_stages}")

        if not len(strides) == num_stages:
            raise ValueError(f"len(strides) must be equal to num_stages: {num_stages}")

        if not len(expansion_rate) == num_stages:
            raise ValueError(f"len(expansion_rate) must be equal to num_stages: {num_stages}")

        if not len(num_conv_per_stage) == num_stages:
            raise ValueError(f"len(num_conv_per_stage) must be equal to num_stages: {num_stages}")

        if not len(num_features_per_stage) == num_stages:
            raise ValueError(
                f"len(num_features_per_stage) must be equal to num_stages: {num_stages}"
            )

        self.in_channels = in_channels
        self.filters_per_stage = num_features_per_stage

        stages = []
        layers = []
        # stem
        stem = get_conv(
            in_channels, num_features_per_stage[0], stem_kernel, strides[0], dim, conv_bias
        )
        layers.append(stem)

        # total number of ConvNeXt blocks
        total_stage_blocks = sum(num_conv_per_stage)
        stage_block_id = 0

        convnext_kwargs = {
            "dim": dim,
            "conv_bias": conv_bias,
            "conv_kwargs": conv_kwargs,
            "norm_layer": norm_layer,
            "norm_kwargs": norm_kwargs,
            "activation": activation,
            "activation_kwargs": activation_kwargs,
            "drop_block": drop_block,
            "drop_kwargs": drop_kwargs,
            "layer_scale_init_value": layer_scale_init_value,
        }

        # first stage
        for num_blocks in range(num_conv_per_stage[0]):
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_p * stage_block_id / (total_stage_blocks - 1.0)
            layers.append(
                ConvNeXtBlock(
                    num_features_per_stage[0],
                    num_features_per_stage[0],
                    kernels[0],
                    1,
                    expansion_rate=expansion_rate[0],
                    stochastic_depth_p=sd_prob,
                    **convnext_kwargs,
                )
            )
            stage_block_id += 1

        stages.append(nn.Sequential(*layers))

        # remaining stages
        for stage in range(1, num_stages):
            layers = []
            # downsample
            layers.append(
                get_conv(
                    num_features_per_stage[stage - 1],
                    num_features_per_stage[stage],
                    kernels[stage],
                    strides[stage],
                    dim,
                    conv_bias,
                )
            )
            for num_blocks in range(num_conv_per_stage[stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_p * stage_block_id / (total_stage_blocks - 1.0)
                layers.append(
                    ConvNeXtBlock(
                        num_features_per_stage[stage],
                        num_features_per_stage[stage],
                        kernels[stage],
                        1,
                        expansion_rate=expansion_rate[stage],
                        stochastic_depth_p=sd_prob,
                        **convnext_kwargs,
                    )
                )
                stage_block_id += 1

            stages.append(nn.Sequential(*layers))

        self.stages = nn.ModuleList(stages)
        self.return_skip = return_skip
        self.strides = [[i] * dim if isinstance(i, int) else list(i) for i in strides]

        if return_skip and 2 in self.strides[0]:
            warnings.warn(
                "The stem performs a pooling operation and `return_skip` is set to True. The "
                "resolution of the skip connections might be wrong and will cause an error if "
                "paired with `UNetDecoder`."
            )

        # initialize weights
        init_kwargs = {}
        if activation == "leakyrelu":
            if activation_kwargs is not None and "negative_slope" in activation_kwargs:
                init_kwargs["neg_slope"] = activation_kwargs["negative_slope"]
        self.apply(get_initialization(initialization, **init_kwargs))

        # store some attributes that a potential decoder needs
        self.dim = dim
        # here, we assume that stem does not perform pooling
        self.filters_per_stage = num_features_per_stage
        self.kernels = kernels
        self.conv_bias = conv_bias
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.activation = activation
        self.activation_kwargs = activation_kwargs
        self.drop_block = drop_block
        self.drop_kwargs = drop_kwargs
        self.initialization = initialization
        self.init_kwargs = init_kwargs

    def forward(
        self, input_data: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor, list[Union[Tensor, MetaTensor]]]:  # noqa: D102
        out = input_data
        encoder_outputs = []
        for stage in self.stages:
            out = stage(out)
            encoder_outputs.append(out)
        if self.return_skip:
            return encoder_outputs
        else:
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
        if not len(input_size) == len(self.strides[0]):
            raise ValueError(
                "`input_size` should be (H, W(, D)) without channel or batch dimension!"
            )
        output = self.in_channels * np.prod(input_size, dtype=np.int64)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    input_size = [i // j for i, j in zip(input_size, self.strides[s])]
                    if hasattr(sq, "compute_pixels_in_output_feature_map"):
                        output += self.stages[s][-1].compute_pixels_in_output_feature_map(
                            input_size
                        )
            else:
                input_size = [i // j for i, j in zip(input_size, self.strides[s])]
                output += self.stages[s].compute_pixels_in_output_feature_map(input_size)
        return output


if __name__ == "__main__":
    from torchinfo import summary

    kernels = [[3, 1], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    strides = [[1, 1], [1, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

    encoder = ConvNeXt(
        in_channels=1,
        num_stages=len(kernels),
        dim=2,
        stem_kernel=[3, 1],
        kernels=kernels,
        strides=strides,
        num_conv_per_stage=2,
        num_features_per_stage=(32, 64, 128, 256, 480, 480, 480),
        conv_bias=True,
        expansion_rate=2,
        stochastic_depth_p=0.0,
        layer_scale_init_value=1e-6,
        conv_kwargs=None,
        norm_layer="group",
        norm_kwargs=None,
        activation="gelu",
        activation_kwargs=None,
        drop_block=False,
        drop_kwargs=None,
        return_skip=True,
        initialization="trunc_normal",
    )

    print(encoder)
    dummy_input = torch.rand((2, 1, 640, 1024))
    out = encoder(dummy_input)
    print(encoder.compute_pixels_in_output_feature_map((640, 1024)))
    print(
        summary(
            encoder,
            input_size=(2, 1, 640, 1024),
            device="cpu",
            depth=6,
            col_names=("input_size", "output_size", "num_params"),
        )
    )
