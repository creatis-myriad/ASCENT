import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.

    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape
    (B, H, W, (D,) C) while channels_first corresponds to inputs with shape (B, C, H, W(, D)).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """Initialize class instance.

        Args:
            normalized_shape: Shape(s) of the dimension(s) to be normalized.
            eps: Value added to the denominator for numerical stability.
            data_format: Whether the input is channel first or channel last.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if len(x.shape) == 4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif len(x.shape) == 5:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            else:
                raise NotImplementedError()
            return x
