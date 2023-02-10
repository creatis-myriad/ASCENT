from torch import Tensor, nn


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """Robust cross entropy loss.

    This serves as a compatibility layer as the target tensor is float and has an extra dimension.

    Retrieved from:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/crossentropy.py
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # noqa: D102
        if len(target.shape) == len(input.shape):
            if not (shape := target.shape[1]) == 1:
                raise ValueError(f"target should have only one channel, got {shape} instead.")
            target = target[:, 0]
        return super().forward(input, target.long())
