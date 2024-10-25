# ==========================================================================
# Copyright (c) 2012-2024 Taiki Miyagawa

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==========================================================================

import math
import warnings

import torch

EPSILON = 1e-12


def get_CosineAnnealingWarmRestartsLinearStart(
        optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0., start_factor: float = EPSILON, milestone: int = 5,
        verbose: bool = False,
        *args, **kwargs):
    """
    Scheduler such that:
    1. Linear warmup (till the "milestone-1"-th epoch (first epoch is set to the 0th epoch))
    2. Cosine annealing with warm restarts (cos down, abrupt up, cos down, abrupt up, ...)

    T_0 = 1 means no cosine annealing (constant LR).
    """
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, total_iters=milestone, verbose=verbose)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, verbose=verbose)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[milestone])

    return scheduler


def get_CosineAnnealingWarmRestartsLinearStartWeightDecay(
        optimizer, T_0: int, num_iter: int, T_mult: int = 1, eta_min: float = 0., start_factor: float = EPSILON,
        gamma: float = 0.9, milestone: int = 5,
        verbose: bool = False,
        *args, **kwargs):
    """
    Scheduler such that:
    1. Linear warmup (till the "milestone-1"-th epoch (first epoch is set to the 0th epoch))
    2. Cosine annealing with warm restarts (cos down, abrupt up, cos down, abrupt up, ...)

    T_0 = 1 means no cosine annealing (constant LR).
    """
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, total_iters=milestone, verbose=verbose)
    scheduler2 = CosineAnnealingWarmRestartsWeightDecay(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, verbose=verbose, gamma=gamma, num_iter=num_iter)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[milestone])

    return scheduler


class CosineAnnealingWarmRestartsWeightDecay(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, num_iter: int, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, gamma: float = 0.9):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(
                "Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(
                "Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.gamma = gamma

        # Calc milestones
        self.milestones = []
        mma = self.milestones.append
        i = 1
        it_period = 0
        if T_mult > 1:
            while it_period <= num_iter - 1:
                if i == 1:
                    it_period = i * T_0 - 1
                else:
                    it_period += T_0 * (T_mult * (i-1))
                if it_period <= num_iter - 1:
                    mma(it_period)
                i += 1
        else:
            while i * T_0 <= num_iter - 1:
                mma(i * T_0 - 1)
                i += 1

        tmp = [0] * num_iter
        for it_idx in self.milestones:
            tmp[it_idx] += 1

        self.power = []
        pa = self.power.append
        for it_idx, _ in enumerate(tmp):
            pa(sum(tmp[:it_idx]))

        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return [
            (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2) *
            self.gamma**self.power[max(self.last_epoch, 0)]
            for base_lr in self.base_lrs
        ]
