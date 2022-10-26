from typing import Union

import torch
import torch.nn as nn
from monai.data import MetaTensor
from torch import Tensor

from ascent.models.components.spyrit_related.utils import round_differentiable


class SpyritNet(nn.Module):
    """A network that implements deep unfolding to dealias color Doppler echocardiographic images.

    SpyritNet is composed of:
        1. A data consistency layer that includes:
            - A forward operator for robust phase unwraping
            - Solving of linear inverse problem using torch.linalg.solve.
        2. A U-Net denoiser.
    """

    def __init__(self, fwd_op: nn.Module, dc_layer: nn.Module, denoiser: nn.Module):
        """Initialize SpyritNet.

        Args:
            fwd_op: Forward operator (Doppler_operator in this case).
            dc_layer: Data consistency (DC) layer.
            denoiser: Denoiser.
        """

        super().__init__()
        self.fwd_op = fwd_op
        self.dc_layer = dc_layer  # must be Tikhonov solve
        self.denoiser = denoiser

    def postprocess(
        self,
        x: Union[Tensor, MetaTensor],
        z_hat: Union[Tensor, MetaTensor],
        W: Union[Tensor, MetaTensor],
    ) -> Union[Tensor, MetaTensor]:
        """Retrieve the Nyquist numbers from smoothed output Doppler velocities given by DC layer
        and denoiser.

        Args:
            x: Input aliased tensor.
            z_hat: Output dealiased tensor given bt Dc layer or denoiser.
            W: Weight tensor (Dopler power in this case).

        Returns:
            Postprocessed output Doppler velocities with the computed Nyquist numbers.
        """

        if W is not None:
            n = round_differentiable((z_hat - W * x) / 2.0)
        else:
            n = round_differentiable((z_hat - x) / 2.0)
        # only single aliasing is considered
        n[torch.abs(n) > 1] = 0
        return x + 2 * n

    def forward(self, x: Union[Tensor, MetaTensor]) -> Union[Tensor, MetaTensor]:
        """Forward propagate the aliased input Doppler velocities through the deep unfolding
        pipeline.

        Args:
            x: Input aliased tensor.

        Returns:
            Dealiased tensor.
        """

        y = self.forward_tikh(x)
        y = self.denoiser(torch.concat([x[:, :-1], x[:, -1:], y], dim=1))
        y = self.postprocess(x[:, :-1], y, None)
        return y

    def forward_tikh(self, x: Union[Tensor, MetaTensor]):
        y = self.reconstruct_tick(x)
        return y

    def reconstruct(self, x: Union[Tensor, MetaTensor]):
        y = self.reconstruct_tick(x)
        # Image domain denoising
        y = self.denoiser(torch.concat([x[:, :-1], x[:, -1:], y], dim=1))
        y = self.postprocess(x[:, :-1], y, None)
        return x

    def reconstruct_tick(self, x: MetaTensor):
        # Data consistency layer
        y = self.dc_layer(x[:, :-1] * x[:, -1:], self.fwd_op)
        y = self.postprocess(x[:, :-1], y, x[:, -1:])
        return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pyrootutils
    from monai.data import DataLoader
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from torchvision import datasets, transforms

    from ascent.models.components.spyrit_related.utils import (
        Forward_operator,
        Tikhonov_solve,
    )
