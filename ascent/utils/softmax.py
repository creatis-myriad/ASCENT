import torch.nn.functional as F
from torch import Tensor


def softmax_helper(x: Tensor) -> Tensor:
    """Utility function to perform softmax along the second dimension.

    Args:
        x: Input tensor.

    Returns:
        Tensor applied with softmax function along the second dimension.
    """
    return F.softmax(x, 1)
