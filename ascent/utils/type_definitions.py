import os
from typing import Sequence, TypeVar, Union

import numpy as np
import torch

__all__ = [
    "NdarrayTensor",
    "NdarrayOrTensor",
    "TensorOrList",
    "PathLike",
]

#: NdarrayOrTensor: Union of numpy.ndarray and torch.Tensor to be used for typing
NdarrayOrTensor = Union[np.ndarray, torch.Tensor]

#: NdarrayTensor
#
# Generic type which can represent either a numpy.ndarray or a torch.Tensor
# Unlike Union can create a dependence between parameter(s) / return(s)
NdarrayTensor = TypeVar("NdarrayTensor", bound=NdarrayOrTensor)

#: TensorOrList: The TensorOrList type is used for defining `batch-first Tensor` or `list of channel-first Tensor`.
TensorOrList = Union[torch.Tensor, Sequence[torch.Tensor]]

#: PathLike: The PathLike type is used for defining a file path.
PathLike = Union[str, os.PathLike]
