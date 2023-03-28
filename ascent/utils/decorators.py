from functools import wraps
from typing import Callable

import numpy as np
import torch
from torch import Tensor

def _has_method(o: object, name: str) -> bool:
    return callable(getattr(o, name, None))

def auto_cast_data(func: Callable) -> Callable:
    """Decorator to allow functions relying on numpy arrays to accept other input data types.
    
    Args:
        func: Function for which to automatically convert the first argument to a numpy array.
        
    Returns:
        Function that accepts input data types other than numpy arrays by converting between them 
        and numpy arrays.
    
    Raises:
        ValueError: If the data is not a numpy or torch.Tensor array.
        
    Retrieved from:
        https://github.com/vitalab/vital/blob/dev/vital/utils/decorators.py
    """
    cast_types = [Tensor]
    dtypes = [np.ndarray, *cast_types]

    @wraps(func)
    def _call_func_with_cast_data(*args, **kwargs):
        if _has_method(args[0], func.__name__):
            # If `func` is a method, pass over the implicit `self` as first argument
            self_or_empty, data, args = args[0:1], args[1], args[2:]
        else:
            self_or_empty, data, args = (), args[0], args[1:]

        dtype = type(data)
        if dtype not in dtypes:
            raise ValueError(
                f"Decorator 'auto_cast_data' used by function '{func.__name__}' does not support "
                f"casting inputs of type '{dtype}' to numpy arrays. Either provide the implementation "
                f"for casting to numpy arrays from '{cast_types}' in 'auto_cast_data' decorator, "
                f"or manually convert the input of '{func.__name__}' to one of the following "
                f"supported types: {dtypes}."
            )
        if dtype == Tensor:
            data_device = data.device
            data = data.detach().cpu().numpy()
        result = func(*self_or_empty, data, *args, **kwargs)
        if dtype == Tensor:
            result = torch.tensor(result, device=data_device)
        return result

    return _call_func_with_cast_data