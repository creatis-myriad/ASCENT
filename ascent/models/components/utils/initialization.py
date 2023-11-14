from typing import Optional

import torch
from torch import nn


class InitWeightsKaimingNormal:
    """Initialize weights of convolution modules with Kaiming normal distribution."""

    def __init__(self, neg_slope: float = 1e-2) -> None:
        """Initialize class instance.

        Args:
            neg_slope: Negative slope of the rectifier used after this layer.
        """
        self.neg_slope = neg_slope

    def __call__(self, module: nn.Module) -> None:  # noqa: D102
        if (
            isinstance(module, nn.Conv3d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.ConvTranspose3d)
            or isinstance(module, nn.Linear)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class InitWeightsXavierUniform:
    """Initialize weights of convolution modules with Xavier uniform distribution."""

    def __init__(self, gain: int = 1):
        """Initialize class instance.

        Args:
            gain: Scaling factor for the weights.
        """
        self.gain = gain

    def __call__(self, module):  # noqa: D102
        if (
            isinstance(module, nn.Conv3d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.ConvTranspose3d)
            or isinstance(module, nn.Linear)
        ):
            module.weight = nn.init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class InitWeightsTruncNormal:
    """Initialize weights of convolution modules with a truncated normal distribution."""

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        a: float = -2.0,
        b: float = 2.0,
        generator: Optional[torch.Generator] = None,
    ):
        """Initialize class instance.

        Args:
            mean: Mean of the normal distribution.
            std: Standard deviation of the normal distribution.
            a: Minimum cutoff value.
            b: Maximum cutoff value.
            generator: Generator used for random number generation.
        """
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b
        self.generator = generator

    def __call__(self, module):  # noqa: D102
        if (
            isinstance(module, nn.Conv3d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.ConvTranspose3d)
            or isinstance(module, nn.Linear)
        ):
            module.weight = nn.init.trunc_normal_(
                module.weight, self.mean, self.std, self.a, self.b, self.generator
            )
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


def get_initialization(name: str, **kwargs):
    """Get initialization function by name.

    Args:
        name: Name of the initialization function.
        **kwargs: Keyword arguments for the initialization function.

    Returns:
        Initialization function.

    Raises:
        NotImplementedError: If the initialization function is not found.
    """
    if name not in initializations:
        raise NotImplementedError(f"Initialization function '{name}' not found.")
    return initializations[name](**kwargs)


initializations = {
    "kaiming_normal": InitWeightsKaimingNormal,
    "xavier_uniform": InitWeightsXavierUniform,
    "trunc_normal": InitWeightsTruncNormal,
}
