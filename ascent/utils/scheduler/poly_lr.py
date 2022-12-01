import torch
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step."""

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0, power=0.9):
        """Initialize class instance.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            max_decay_steps: after this step, we stop decreasing learning rate
            end_learning_rate: scheduler stopping learning rate decay, value of learning rate must be this value
            power: The power of the polynomial.
        """

        self.max_decay_steps = max(max_decay_steps, 1)
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):  # noqa: D102
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [
            (base_lr - self.end_learning_rate)
            * ((1 - self.last_step / self.max_decay_steps) ** (self.power))
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]

    def step(self, step=None):  # noqa: D102
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [
                (base_lr - self.end_learning_rate)
                * ((1 - self.last_step / self.max_decay_steps) ** (self.power))
                + self.end_learning_rate
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group["lr"] = lr


if __name__ == "__main__":
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.01)
    scheduler = PolynomialLR(optim, max_decay_steps=999)

    for epoch in range(1, 10):
        scheduler.step(epoch)

        print(epoch, optim.param_groups[0]["lr"])
