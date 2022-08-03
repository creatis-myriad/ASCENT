from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class DataParameters:
    """Class for defining parameters related to the nature of the data.
    Args:
        in_shape: Shape of the input data (e.g. channels, height, width).
        out_shape: Shape of the target data (e.g. channels, height, width).
    """

    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]

