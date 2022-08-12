import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.functional import dice, stat_scores
from torchmetrics.utilities import check_forward_full_state_property

from covid.models.components.unet_related.utils import softmax_helper, sum_tensor


class Dice(Metric):
    full_state_update = True

    def __init__(self, num_classes):
        super().__init__(dist_sync_on_step=True)
        self.add_state("tp_hard", default=list())
        self.add_state("fp_hard", default=list())
        self.add_state("fn_hard", default=list())
        self.add_state("dice", default=list())

    def update(self, preds, target):
        tp_hard, fp_hard, fn_hard, dice = self.compute_stats(preds, target)
        self.tp_hard.append(tp_hard)
        self.fp_hard.append(fp_hard)
        self.fn_hard.append(fn_hard)
        self.dice.append(dice)

    def compute(self):
        tp = np.sum(self.tp_hard, 0)
        fp = np.sum(self.fp_hard, 0)
        fn = np.sum(self.fn_hard, 0)

        global_dc_per_class = [
            i if not np.isnan(i) else 0.0
            for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        ]

        return np.mean(global_dc_per_class)

    @staticmethod
    def compute_stats(preds, target):
        num_classes = preds.shape[1]
        preds_softmax = softmax_helper(preds)
        preds_seg = preds_softmax.argmax(1)
        target = target[:, 0]
        axes = tuple(range(1, len(target.shape)))
        tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(preds_seg.device.index)
        fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(preds_seg.device.index)
        fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(preds_seg.device.index)
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor(
                (preds_seg == c).float() * (target == c).float(), axes=axes
            )
            fp_hard[:, c - 1] = sum_tensor(
                (preds_seg == c).float() * (target != c).float(), axes=axes
            )
            fn_hard[:, c - 1] = sum_tensor(
                (preds_seg != c).float() * (target == c).float(), axes=axes
            )

        tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()
        hard_dice = (2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)

        return list(tp_hard), list(fp_hard), list(fn_hard), list(hard_dice)


if __name__ == "__main__":
    dc = dice(
        torch.rand((2, 3, 128, 128)),
        torch.randint(0, 3, (2, 1, 128, 128)),
        average="macro",
        mdmc_average="global",
        num_classes=3,
    )
    # check_forward_full_state_property(
    #     Dice,
    #     init_args={"num_classes": 3},
    #     input_args={
    #         "preds": torch.rand((2, 3, 128, 128), device="cuda"),
    #         "target": torch.rand((2, 3, 128, 128), device="cuda"),
    #     },
    # )
