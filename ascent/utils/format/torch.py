from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence


def _apply(obj, func):
    if isinstance(obj, (list, tuple)):
        if isinstance(obj, PackedSequence):
            return type(obj)(
                *(
                    _apply(getattr(obj, el), func) if el != "batch_sizes" else getattr(obj, el)
                    for el in obj._fields
                )
            )
        return type(obj)(_apply(el, func) for el in obj)
    if isinstance(obj, dict):
        return {k: _apply(el, func) for k, el in obj.items()}
    return func(obj)


def torch_apply(obj: Union[Tuple, List, Dict], func: Callable) -> Union[Tuple, List, Dict]:
    """Applies a function to all tensors inside a Python object composed of the supported types.

    References:
        - This function is copied from the 'poutyne' framework:
        https://github.com/GRAAL-Research/poutyne/blob/aeb78c2b26edaa30663a88522d39a187baeec9cd/poutyne/utils.py#L83-L101

    Args:
        obj: The Python object to convert.
        func: The function to apply.

    Returns:
        A new Python object with the same structure as `obj` but where the tensors have been applied the function
        `func`. Not supported types are left as reference in the new object.
    """

    def fn(t):
        return func(t) if torch.is_tensor(t) else t

    return _apply(obj, fn)


def torch_to_numpy(obj: Union[Tuple, List, Dict], copy: bool = False) -> Union[Tuple, List, Dict]:
    """Converts to Numpy arrays all tensors inside a Python object composed of the supported types.

    References:
        - This function is copied from the 'poutyne' framework:
        https://github.com/GRAAL-Research/poutyne/blob/aeb78c2b26edaa30663a88522d39a187baeec9cd/poutyne/utils.py#L35-L76

    Args:
        obj: The Python object to convert.
        copy: Whether to copy the memory. By default, if a tensor is already on CPU, the Numpy array will be a view of
            the tensor.

    Returns:
        A new Python object with the same structure as `obj` but where the tensors are now Numpy arrays. Not supported
        types are left as reference in the new object.

    Example:
        .. code-block:: python
            >>> from vital.utils.format.torch import torch_to_numpy
            >>> torch_to_numpy({
            ...     'first': torch.tensor([1, 2, 3]),
            ...     'second':[torch.tensor([4,5,6]), torch.tensor([7,8,9])],
            ...     'third': 34
            ... })
            {
                'first': array([1, 2, 3]),
                'second': [array([4, 5, 6]), array([7, 8, 9])],
                'third': 34
            }
    """
    if copy:

        def func(t):
            return t.detach().cpu().numpy().copy()

    else:

        def func(t):
            return t.detach().cpu().numpy()

    return torch_apply(obj, func)


def numpy_to_torch(obj: Union[Tuple, List, Dict]) -> Union[Tuple, List, Dict]:
    """Converts to tensors all Numpy arrays inside a Python object composed of the supported types.

    References:
        - This function, and the private `_apply` recursive function it calls, are copied from the 'poutyne' framework:
        https://github.com/GRAAL-Research/poutyne/blob/aeb78c2b26edaa30663a88522d39a187baeec9cd/poutyne/utils.py#L132-L164

    Args:
        obj: The Python object to convert.

    Returns:
        A new Python object with the same structure as `obj` but where the Numpy arrays are now tensors. Not supported
        types are left as reference in the new object.

    Example:
        .. code-block:: python
            >>> from vital.utils.format.torch import numpy_to_torch
            >>> numpy_to_torch({
            ...     'first': np.array([1, 2, 3]),
            ...     'second':[np.array([4,5,6]), np.array([7,8,9])],
            ...     'third': 34
            ... })
            {
                'first': tensor([1, 2, 3]),
                'second': [tensor([4, 5, 6]), tensor([7, 8, 9])],
                'third': 34
            }
    """

    def fn(a):
        return torch.from_numpy(a) if isinstance(a, np.ndarray) else a

    return _apply(obj, fn)
