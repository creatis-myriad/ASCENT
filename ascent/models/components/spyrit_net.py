from typing import Optional, Union

import torch
import torch.nn as nn
from monai.data import MetaTensor
from torch import Tensor

from ascent.models.components.spyrit_related.utils import round_differentiable


class SpyritNet(nn.Module):
    """A network that implements deep unfolding for phase unwrapping.

    SpyritNet is composed of:
        1. A classical phase unwrapping algorithm.
        2. A U-Net denoiser.
    """

    def __init__(self, unwrap: nn.Module, denoiser: nn.Module, postprocess: bool = True):
        """Initialize SpyritNet.

        Args:
            unwrap: Phase unwrapping algorithm.
            denoiser: Denoiser.
            postprocess: Whether to run postprocessing before feeding data to the denoiser.
        """

        super().__init__()
        self.unwrap = unwrap
        self.denoiser = denoiser
        self.do_postprocess = postprocess

    def postprocess(
        self,
        x: Union[Tensor, MetaTensor],
        z_hat: Union[Tensor, MetaTensor],
        W: Optional[Union[Tensor, MetaTensor]] = None,
    ) -> Union[Tensor, MetaTensor]:
        """Retrieve the Nyquist numbers from smoothed output Doppler velocities given by phase
        unwrapping algorithm and denoiser.

        Args:
            x: Input aliased tensor.
            z_hat: Output dealiased tensor given by phase unwrapping algorithm or denoiser.
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
        """Forward propagate the aliased input, x, through the deep unfolding pipeline.

        Args:
            x: Input aliased tensor. (b, c, w, h)

        Returns:
            Dealiased tensor. (b, c, w, h)
        """

        y = self.unwrap(x[:, :-1] * x[:, -1:])
        if self.do_postprocess:
            y = self.postprocess(x[:, :-1], y, x[:, -1:])
        y = self.denoiser(torch.concat([x[:, :-1], x[:, -1:], y], dim=1))
        # y = self.postprocess(x[:, :-1], y, None)
        return y
