from torch import Tensor, nn


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """Robust cross entropy loss copied from https://github.com/MIC-
    DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/crossentropy.py.

    This serves as a compatibility layer as the target tensor is float and has an extra dimension.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # noqa: D102
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())
