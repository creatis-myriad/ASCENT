from functools import wraps
from typing import Callable

import torch
from monai.data import MetaTensor
from torch import Tensor

from ascent.utils.format.torch import torch_to_numpy


def _has_method(o: object, name: str) -> bool:
    return callable(getattr(o, name, None))


def auto_cast_data(func: Callable) -> Callable:
    """Decorator to allow functions relying on numpy arrays to accept other input data types.

    Args:
        func: Function for which to automatically convert all the torch.Tensor/monai.data.MetaTensor
        arguments and keyword arguments to numpy arrays.

    Returns:
        Function that accepts input data types other than numpy arrays by converting between them
        and numpy arrays.
    """
    cast_types = (Tensor, MetaTensor)

    @wraps(func)
    def _call_func_with_cast_data(*args, **kwargs):
        dtype_args = [type(arg) for arg in args]
        device_args = [arg.device if isinstance(arg, cast_types) else None for arg in args]
        dtype_kwargs = [type(v) for _, v in kwargs.items()]
        device_kwargs = [
            v.device if isinstance(v, cast_types) else None for _, v in kwargs.items()
        ]

        # Cast all the args to numpy arrays
        args = torch_to_numpy(args)

        # Cast all the kwargs to numpy arrays
        kwargs = torch_to_numpy(kwargs)

        if _has_method(args[0], func.__name__):
            # If `func` is a method, pass over the implicit `self` as first argument
            self_or_empty, args = args[0:1], args[1:]
        else:
            self_or_empty, args = (), args[0:]

        result = func(*self_or_empty, *args, **kwargs)

        if any(t in cast_types for t in (dtype_args + dtype_kwargs)):
            cuda_device = [d for d in (device_args + device_kwargs) if "cuda" in d.type]
            mps_device = [d for d in (device_args + device_kwargs) if "mps" in d.type]
            cpu_device = [d for d in (device_args + device_kwargs) if "cpu" in d.type]

            device = max([cuda_device, mps_device, cpu_device], key=len)[0]
            result = torch.tensor(result, device=device)
        return result

    return _call_func_with_cast_data
