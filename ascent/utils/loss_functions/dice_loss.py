from typing import Optional, Union

import monai
import torch
from torch import Tensor, nn

from ascent.utils.loss_functions.crossentropy import RobustCrossEntropyLoss
from ascent.utils.softmax import softmax_helper
from ascent.utils.tensor_utils import sum_tensor


def get_tp_fp_fn_tn(
    net_output: Tensor,
    target: Tensor,
    axes: Optional[Union[int, tuple[int]]] = None,
    mask: Optional[Tensor] = None,
    square: bool = False,
) -> tuple[Tensor]:
    """Computes the number of true positives, false positives, true negatives, false negatives.

    Note:
        - net_output must be (b, c, x, y(, z)))
        - target must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) OR one hot
          encoding (b, c, x, y(, z)).
        - If mask is provided it must have shape (b, 1, x, y(, z))).

    Args:
        net_output: Tensor output by a network.
        target: Ground truth reference.
        axes: Axes to reduce.
        mask: Mask in case certain classes want to be ignored.
        square: Set true to square the tp, fp, fn, and tn before summation.

    Returns:
        Tuple containing the computed tp, fp, fn, and tn.

    Retrieved from:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/dice_loss.py
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = target.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            target = target.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, target.shape)]):
            # if this is the case then target is probably already a one hot encoding
            y_onehot = target
        else:
            target = target.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, target, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp**2
        fp = fp**2
        fn = fn**2
        tn = tn**2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    """Soft Dice loss.

    Retrieved from:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/dice_loss.py
    """

    def __init__(
        self,
        apply_nonlin: Optional[nn.Module] = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
    ):
        """Initialize class instance.

        Args:
            apply_nonlin: Non linearity to be applied.
            batch_dice: Whether to compute batch dice.
            do_bg: Whether to include background.
            smooth: Epsillon to be added on denominator to avoid dividing by zero.
        """
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x: Tensor, y: Tensor, loss_mask: Optional[Tensor] = None) -> Tensor:
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        # monai MetaTensor indexing, e.g. [1:] will fail on Jean Zay cluster
        if type(dc) == monai.data.meta_tensor.MetaTensor:
            dc = dc.as_tensor()

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.0):
        """squares the terms in the denominator as proposed by Milletari et al."""
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then target is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x**2 + y_onehot**2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        # monai MetaTensor indexing, e.g. [1:] will fail on Jean Zay cluster
        if type(dc) == monai.data.meta_tensor.MetaTensor:
            dc = dc.as_tensor()

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class DC_and_CE_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        aggregate="sum",
        square_dice=False,
        weight_ce=1,
        weight_dice=1,
        log_dice=False,
        ignore_label=None,
    ):
        """CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you
        want.

        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            assert not square_dice, "not implemented"
            ce_kwargs["reduction"] = "none"
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """target must be b, c, x, y(, z) with c=1.

        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, "not implemented for one hot encoding"
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son")  # reserved for other stuff (later)
        return result
