"""Small serializable learning-rate schedulers used by detector training."""

import math


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.min_lrs = [lr * min_lr_ratio for lr in self.base_lrs]
        self.epoch = -1

    def step_epoch(self, epoch):
        self.epoch = epoch
        if epoch < self.warmup_epochs:
            alpha = (epoch + 1) / max(self.warmup_epochs, 1)
            lrs = [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            lrs = [
                min_lr
                + 0.5
                * (base_lr - min_lr)
                * (1 + math.cos(math.pi * progress))
                for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
            ]
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def step_iter(self):
        return None

    def state_dict(self):
        return {"epoch": self.epoch}

    def load_state_dict(self, state):
        self.epoch = int(state.get("epoch", -1))


class WarmupStepDecayScheduler:
    def __init__(
        self,
        optimizer,
        warmup_iters=500,
        warmup_start_factor=0.001,
        milestones=(8, 11),
        gamma=0.1,
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.warmup_start_factor = warmup_start_factor
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.iteration = 0
        self.epoch = -1

    def _decay(self):
        return self.gamma ** sum(self.epoch >= milestone for milestone in self.milestones)

    def step_epoch(self, epoch):
        self.epoch = epoch
        if self.iteration >= self.warmup_iters:
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * self._decay()

    def step_iter(self):
        self.iteration += 1
        if self.iteration <= self.warmup_iters:
            alpha = self.iteration / max(self.warmup_iters, 1)
            factor = self.warmup_start_factor + (1 - self.warmup_start_factor) * alpha
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * self._decay() * factor

    def state_dict(self):
        return {"iteration": self.iteration, "epoch": self.epoch}

    def load_state_dict(self, state):
        self.iteration = int(state.get("iteration", 0))
        self.epoch = int(state.get("epoch", -1))
